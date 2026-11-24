# 1. Imports and FastAPI init
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import json
import whisper
import tempfile

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from fuzzyMatch import extract_safest_coordinate

app = FastAPI(title="AMR Voice Command Node")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Model Configuration & Loading (Tokenizer, Base Model, LoRA, Whisper)
BASE_MODEL = "Qwen/Qwen3-1.7B-Base"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LORA_ADAPTER = os.path.join(BASE_DIR, "model", "qwen3-1.7b-amr-lora")
WHISPER_MODEL = "base"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=DTYPE, device_map=DEVICE, low_cpu_mem_usage=True
)
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER)
model.eval()

whisper_model = whisper.load_model(WHISPER_MODEL)


# 3. Define Helper Function HERE (so tokenizer and model are available)
def extract_amr_command(command: str):
    instruction = "Extract the action and destination from the AMR command and return them as JSON."
    prompt = f"### Instruction:\n{instruction}\n\n### AMR Command:\n{command}\n\n### Output:\n"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    generated_tokens = outputs[0, inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=422,
            detail={"message": "Fine-tuned model returned invalid JSON.", "raw_output": response}
        )

    if "action" not in parsed or "destination" not in parsed:
        raise HTTPException(status_code=422, detail="Model output missing 'action' or 'destination'.")

    return parsed


# 4. Pydantic Model
class CommandRequest(BaseModel):
    text: str


# 5. API Routes
@app.get("/")
async def home():
    if not os.path.exists("index.html"):
        raise HTTPException(status_code=404, detail="index.html not found.")
    return FileResponse("index.html")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/send-command")
async def process_command(request: CommandRequest):
    command = request.text.strip()
    if not command:
        raise HTTPException(status_code=400, detail="Command cannot be empty.")

    try:
        result = extract_amr_command(command)
        action = result.get("action", "none")
        destination = result.get("destination", "none")

        coordinates = extract_safest_coordinate(destination)

        goal_path = os.path.join(os.path.dirname(__file__), "..", "current_goal.json")

        if coordinates is None and destination != "none":
            with open(goal_path, "w") as f:
                json.dump({"destination": destination, "coordinates": None, "status": "not_found"}, f)
            return {
                "action": action,
                "destination": destination,
                "target": "none",
                "coordinates": None,
                "message": "Location/object not found"
            }

        with open(goal_path, "w") as f:
            json.dump({"destination": destination, "coordinates": coordinates, "status": "navigating"}, f)

        return {
            "action": action,
            "destination": destination,
            "target": destination,
            "coordinates": coordinates,
            "message": "Success"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)