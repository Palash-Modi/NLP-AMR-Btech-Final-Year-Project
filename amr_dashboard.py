"""AMR live dashboard for Windows. Needs only Python 3 (no installs).
Run:   python amr_dashboard.py
Open:  http://localhost:8000
Optional: python amr_dashboard.py --source https://amr.nlpamr.pp.ua/sensors/json --port 8000
It serves the dashboard and relays the robot JSON (this avoids browser CORS errors).
Nothing on the Pi is changed."""
import argparse, json, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="https://amr.nlpamr.pp.ua/sensors/json")
ap.add_argument("--port", type=int, default=8000)
args = ap.parse_args()

PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AMR Live Telemetry</title>
<style>
:root{--bg:#f4f6fb;--card:#fff;--tx:#1f2937;--mut:#6b7280;--g:#16a34a;--b:#2563eb;--r:#dc2626;--p:#7c3aed;--k:#111827;--a:#d97706}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font-family:Segoe UI,system-ui,sans-serif;padding:14px}
header{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;flex-wrap:wrap;gap:8px}
h1{font-size:20px;margin:0}
.pill{padding:5px 12px;border-radius:99px;font-weight:700;font-size:12px;color:#fff;background:#9ca3af}
.pill.live{background:var(--g)}.pill.off{background:var(--r)}.pill.stale{background:var(--a)}
.grid{display:grid;gap:12px}
.kpis{grid-template-columns:repeat(auto-fit,minmax(150px,1fr));margin-bottom:12px}
.card{background:var(--card);border-radius:12px;padding:12px 14px;border:2px solid #e5e7eb;box-shadow:0 1px 3px #0001}
.card h3{margin:0 0 4px;font-size:13px;color:var(--mut);font-weight:600}
.val{font-size:30px;font-weight:800;line-height:1.1}.val small{font-size:14px;font-weight:600;color:var(--mut)}
.k1{border-color:#bbf7d0;background:#f0fdf4}.k1 .val{color:var(--g)}
.k2{border-color:#fecaca;background:#fef2f2}.k2 .val{color:var(--r)}
.k3{border-color:#fde68a;background:#fffbeb}.k3 .val{color:var(--a);font-size:22px;padding-top:6px}
.k4{border-color:#bfdbfe;background:#eff6ff}.k4 .val{color:var(--b)}
.k5{border-color:#ddd6fe;background:#f5f3ff}.k5 .val{color:var(--p)}
.k6{border-color:#d1d5db;background:#f9fafb}.k6 .val{color:var(--k)}
.main{grid-template-columns:minmax(260px,1fr) 2fr;margin-bottom:12px}
@media(max-width:800px){.main{grid-template-columns:1fr}}
canvas{width:100%;display:block}
.charts{grid-template-columns:repeat(auto-fit,minmax(280px,1fr));margin-bottom:12px}
.strip{display:flex;gap:18px;flex-wrap:wrap;background:var(--k);color:#fff;border-radius:12px;padding:10px 16px;font-size:13px}
.strip b{color:#fbbf24}
</style></head><body>
<header><h1>🤖 AMR Live Telemetry</h1><span id="conn" class="pill">CONNECTING</span></header>

<div class="grid kpis">
 <div class="card k1"><h3> Battery</h3><div class="val" id="bat">--<small>%</small></div></div>
 <div class="card k4"><h3> Voltage</h3><div class="val" id="volt">--<small>V</small></div></div>
 <div class="card k2"><h3> Obstacle distance</h3><div class="val" id="dist">--<small>cm</small></div></div>
 <div class="card k3"><h3> Motion state</h3><div class="val" id="state">--</div></div>
 <div class="card k5"><h3> Accel Z</h3><div class="val" id="az">--<small>m/s²</small></div></div>
 <div class="card k6"><h3> Gyro Z</h3><div class="val" id="gz">--<small>°/s</small></div></div>
</div>

<div class="grid main">
 <div class="card"><h3> Sonar radar (front)</h3><canvas id="radar" width="400" height="300"></canvas></div>
 <div class="card"><h3> Distance over time (cm)</h3><canvas id="c1" width="800" height="300"></canvas></div>
</div>

<div class="grid charts">
 <div class="card"><h3> Battery over time (%)</h3><canvas id="c2" width="500" height="200"></canvas></div>
 <div class="card"><h3> Accel Z over time (m/s²)</h3><canvas id="c3" width="500" height="200"></canvas></div>
 <div class="card"><h3> Gyro Z over time (°/s)</h3><canvas id="c4" width="500" height="200"></canvas></div>
</div>

<div class="strip">
 <span>Status: <b id="s1">--</b></span>
 <span>Command: <b id="s2">--</b></span>
 <span>Stop threshold: <b id="s3">--</b></span>
 <span>Accel X/Y: <b id="s4">--</b></span>
 <span>Robot time: <b id="s5">--</b></span>
 <span>Last update: <b id="s6">--</b></span>
</div>

<script>
const URL_ = '/sensors/json', EVERY = 500, MAXP = 120;
const $ = id => document.getElementById(id);
const H = {t:[], dist:[], bat:[], az:[], gz:[]};
let last = {}, lastOk = 0, stopCm = 25;

function push(k, v){ H[k].push(v); if (H[k].length > MAXP) H[k].shift(); }

function line(cv, data, color, opts){
  opts = opts || {};
  const c = cv.getContext('2d'), w = cv.width, h = cv.height, pad = 30;
  c.clearRect(0,0,w,h);
  const pts = data.filter(v => v !== null && isFinite(v));
  let lo = opts.min !== undefined ? opts.min : Math.min.apply(null, pts.concat([0]));
  let hi = opts.max !== undefined ? opts.max : Math.max.apply(null, pts.concat([1]));
  if (opts.min === undefined && pts.length && Math.min.apply(null, pts) > 0 && opts.tight){ lo = Math.min.apply(null, pts) - 1; }
  if (hi === lo) hi = lo + 1;
  c.strokeStyle = '#e5e7eb'; c.fillStyle = '#6b7280'; c.font = '11px sans-serif'; c.lineWidth = 1;
  for (let i = 0; i <= 4; i++){
    const y = pad/2 + (h - pad*1.5) * i/4, v = hi - (hi-lo)*i/4;
    c.beginPath(); c.moveTo(pad+10, y); c.lineTo(w-6, y); c.stroke();
    c.fillText(v.toFixed(1), 2, y+4);
  }
  if (opts.ref !== undefined && opts.ref >= lo && opts.ref <= hi){
    const y = pad/2 + (h - pad*1.5) * (1 - (opts.ref-lo)/(hi-lo));
    c.strokeStyle = '#dc2626'; c.setLineDash([5,4]); c.beginPath(); c.moveTo(pad+10,y); c.lineTo(w-6,y); c.stroke(); c.setLineDash([]);
  }
  if (!data.length) return;
  const X = i => pad+10 + (w-pad-16) * (data.length > 1 ? i/(MAXP-1) : 0) + (w-pad-16)*(1 - data.length/MAXP);
  const Y = v => pad/2 + (h - pad*1.5) * (1 - (v-lo)/(hi-lo));
  c.beginPath(); let started = false, fx = 0, lx = 0;
  data.forEach((v,i) => {
    if (v === null || !isFinite(v)){ started = false; return; }
    const x = X(i), y = Y(v);
    if (!started){ c.moveTo(x,y); started = true; fx = x; } else c.lineTo(x,y);
    lx = x;
  });
  c.strokeStyle = color; c.lineWidth = 2.5; c.stroke();
  if (opts.fill){
    c.lineTo(lx, Y(lo)); c.lineTo(fx, Y(lo)); c.closePath();
    c.globalAlpha = .15; c.fillStyle = color; c.fill(); c.globalAlpha = 1;
  }
}

function radar(d){
  const cv = $('radar'), c = cv.getContext('2d'), w = cv.width, h = cv.height;
  c.clearRect(0,0,w,h);
  const cx = w/2, cy = h-20, R = h-50, MAX = 350;
  c.strokeStyle = '#d1d5db'; c.fillStyle = '#6b7280'; c.font = '11px sans-serif'; c.lineWidth = 1;
  [0.25,0.5,0.75,1].forEach(f => {
    c.beginPath(); c.arc(cx,cy,R*f,Math.PI,2*Math.PI); c.stroke();
    c.fillText(Math.round(MAX*f)+'cm', cx+R*f-26, cy-3);
  });
  [-60,-30,0,30,60].forEach(a => {
    const r = (a-90)*Math.PI/180;
    c.beginPath(); c.moveTo(cx,cy); c.lineTo(cx+R*Math.cos(r), cy+R*Math.sin(r)); c.stroke();
  });
  // stop zone
  c.fillStyle = 'rgba(220,38,38,.12)'; c.beginPath(); c.moveTo(cx,cy);
  c.arc(cx,cy,R*Math.min(stopCm,MAX)/MAX,Math.PI,2*Math.PI); c.closePath(); c.fill();
  // robot
  c.fillStyle = '#111827'; c.fillRect(cx-12,cy-6,24,14);
  const v = d.sonar_cm;
  if (v >= 2 && v <= MAX){
    const r = Math.min(v,MAX)/MAX*R, danger = v <= stopCm;
    c.strokeStyle = danger ? '#dc2626' : '#16a34a'; c.lineWidth = 4;
    c.beginPath(); c.arc(cx,cy,r,Math.PI*1.2,Math.PI*1.8); c.stroke();
    c.fillStyle = c.strokeStyle; c.beginPath(); c.arc(cx,cy-r,7,0,7); c.fill();
    c.fillStyle = '#1f2937'; c.font = 'bold 14px sans-serif'; c.fillText(v.toFixed(1)+' cm', cx+12, cy-r+5);
  } else {
    c.fillStyle = '#6b7280'; c.font = '14px sans-serif'; c.textAlign = 'center';
    c.fillText('No echo / clear path', cx, cy-R/2); c.textAlign = 'left';
  }
}

function render(d){
  const noEcho = !(d.sonar_cm >= 2 && d.sonar_cm <= 350);
  $('bat').innerHTML = d.battery_pct + '<small>%</small>';
  $('volt').innerHTML = d.voltage.toFixed(2) + '<small>V</small>';
  $('dist').innerHTML = noEcho ? '--<small>cm</small>' : d.sonar_cm.toFixed(1) + '<small>cm</small>';
  $('state').textContent = d.motion_state;
  $('az').innerHTML = d.accel_z.toFixed(2) + '<small>m/s²</small>';
  $('gz').innerHTML = (d.gyro_z !== undefined ? d.gyro_z.toFixed(1) : '--') + '<small>°/s</small>';
  const danger = !noEcho && d.sonar_cm <= stopCm;
  $('s1').textContent = danger ? '⚠ OBSTACLE' : (d.motion_state === 'STOPPED' ? 'IDLE' : 'MOVING');
  $('s2').textContent = d.command || '--';
  $('s3').textContent = stopCm + ' cm';
  $('s4').textContent = d.accel_x !== undefined ? d.accel_x.toFixed(2) + ' / ' + d.accel_y.toFixed(2) : '--';
  $('s5').textContent = (d.timestamp || '').replace('T',' ').replace('Z','');
  $('s6').textContent = new Date().toLocaleTimeString();
  push('dist', noEcho ? null : d.sonar_cm); push('bat', d.battery_pct);
  push('az', d.accel_z); push('gz', d.gyro_z !== undefined ? d.gyro_z : null);
  line($('c1'), H.dist, '#dc2626', {min:0, fill:true, ref:stopCm});
  line($('c2'), H.bat, '#16a34a', {min:0, max:100, fill:true});
  line($('c3'), H.az, '#7c3aed', {tight:true});
  line($('c4'), H.gz, '#2563eb', {min:-30, max:30});
  radar(d);
}

function setConn(cls, txt){ const e = $('conn'); e.className = 'pill ' + cls; e.textContent = txt; }

async function poll(){
  try{
    const r = await fetch(URL_, {cache:'no-store'});
    if (!r.ok) throw new Error(r.status);
    const d = await r.json();
    if (d.stop_cm !== undefined) stopCm = d.stop_cm;
    last = d; lastOk = Date.now();
    render(d); setConn('live', '● LIVE');
  }catch(e){
    setConn(Date.now() - lastOk > 5000 ? 'off' : 'stale', Date.now() - lastOk > 5000 ? 'OFFLINE' : 'STALE');
  }
  setTimeout(poll, EVERY);
}
radar({sonar_cm:-1});
poll();
</script></body></html>
"""

class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html", "/dashboard"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/sensors/json":
            try:
                req = urllib.request.Request(args.source, headers={"User-Agent": "amr-dashboard"})
                with urllib.request.urlopen(req, timeout=5) as r:
                    data = r.read()
                json.loads(data)
                self._send(200, data, "application/json")
            except Exception as e:
                self._send(502, json.dumps({"error": str(e)}).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def log_message(self, *a):
        pass

print("Dashboard running -> http://localhost:%d   (source: %s)" % (args.port, args.source))
print("Press Ctrl+C to stop.")
try:
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()
except KeyboardInterrupt:
    pass