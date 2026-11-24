# AMR NLP Integration

This package contains the NLP and voice-command components for the Autonomous Mobile Robot (AMR).

## Components

- `main.py` - FastAPI server, Whisper speech-to-text, and Qwen3-1.7B inference
- `fuzzyMatch.py` - entity/destination matching utility
- `index.html` - AMR command interface
- `model/qwen3-1.7b-amr-lora/` - trained LoRA adapter
- `requirements.txt` - Python dependencies

## Model

Base model:
Qwen/Qwen3-1.7B-Base

Fine-tuning:
QLoRA / LoRA

Quantization during training:
4-bit NF4

The LoRA adapter must be used together with the Qwen3-1.7B base model.

## Run

Create a virtual environment:

python3 -m venv venv

Activate it:

source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Run the FastAPI server:

uvicorn main:app --host 0.0.0.0 --port 8000

API documentation:

http://<robot-ip>:8000/docs

## Text Command API

POST /send-command

Example request:

{
    "text": "Go to the robotics lab."
}

Expected response:

{
    "action": "navigate",
    "destination": "robotics lab",
    "target": "robotics lab"
}

## Voice API

POST /transcribe

The Whisper model converts spoken input into text before NLP processing.

## Important

The Qwen3-1.7B base model is not included in this package.
It will be downloaded from Hugging Face when required.

The final lightweight Raspberry Pi deployment version will use a quantized GGUF model.
