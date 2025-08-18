# main.py  — flat setup, banners, and working buttons

import os, base64, datetime, re, random, string
from decimal import Decimal, ROUND_DOWN, getcontext
from typing import Optional
from contextlib import asynccontextmanager

import psycopg
import httpx
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from itsdangerous import URLSafeSerializer, BadSignature
from pydantic import BaseModel

# ---- import your game logic ----
from mines import (
    mines_start, mines_pick, mines_cashout, mines_state, mines_history
)
from crash import (
    ensure_betting_round, place_bet, load_round, begin_running,
    finish_round, create_next_betting, last_busts, your_bet,
    your_history, cashout_now, current_multiplier
)

# ---------- Config ----------
getcontext().prec = 28
TWO = Decimal("0.01")
def D(x): 
    return x if isinstance(x, Decimal) else Decimal(str(x))
def q2(x: Decimal): 
    return D(x).quantize(TWO, rounding=ROUND_DOWN)

UTC = datetime.timezone.utc
def now_utc(): return datetime.datetime.now(UTC)
def iso(dt: Optional[datetime.datetime]):
    if not dt: return None
    return dt.astimezone(UTC).isoformat()

DATABASE_URL = os.getenv("DATABASE_URL")
SECRET_KEY   = os.getenv("SECRET_KEY", "dev-secret")
PORT         = int(os.getenv("PORT", "8080"))
DISCORD_INVITE = os.getenv("DISCORD_INVITE", "")  # optional

# ---------- App & lifespan ----------
def _ensure_dir(p):
    try: os.makedirs(p, exist_ok=True)
    except: pass

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
IMG_DIR    = os.path.join(STATIC_DIR, "img")
_ensure_dir(IMG_DIR)

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    apply_migrations()
    yield

app = FastAPI(lifespan=lifespan)

# static mount (you can put CSS/JS later if you want)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Serve banners from /img/<file> looking in ./static/img/<file>
_TRANSPARENT_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
)
@app.get("/img/{filename}")
async def serve_img(filename: str):
    p = os.path.join(IMG_DIR, filename)
    if os.path.isfile(p):
        return FileResponse(p)
    return Response(content=_TRANSPARENT_PNG, media_type="image/png")

# ---------- Sessions ----------
SER = URLSafeSerializer(SECRET_KEY, salt="session-v1")
def _set_session(resp, data: dict):
    resp.set_cookie("session", SER.dumps(data), max_age=30*86400, httponly=True, samesite="lax")
def _clear_session(resp):
    resp.delete_cookie("session")
def _require_session(request: Request) -> dict:
    raw = request.cookies.get("session")
    if not raw: raise HTTPException(401, "Not logged in")
    try:
        sess = SER.loads(raw)
        if not sess.get("id"): raise BadSignature("no id")
        return sess
    except BadSignature:
        raise HTTPException(401, "Invalid session")

# Demo login for local testing (click “Sign in demo” on the page)
@app.get("/demo_login")
async def demo_login(name: str = "demo_user", user_id: str = "100000000000000001"):
    resp = RedirectResponse("/")
    _set_session(resp, {"id": user_id, "username": name, "avatar_url": ""})
    return resp
@app.get("/logout")
async def logout():
    resp = RedirectResponse("/")
    _clear_session(resp)
    return resp

# ---------- DB helpers / schema ----------
def with_conn(fn):
    def wrapper(*args, **kwargs):
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL not set")
        with psycopg.connect(DATABASE_URL) as con:
            with con.cursor() as cur:
                res = fn(cur, *args, **kwargs)
                con.commit()
                return res
    return wrapper

@with_conn
def init_db(cur):
    cur.execute("""CREATE TABLE IF NOT EXISTS balances(
        user_id TEXT PRIMARY KEY,
        balance NUMERIC(18,2) NOT NULL DEFAULT 0
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS balance_log(
        id BIGSERIAL PRIMARY KEY,
        actor_id TEXT NOT NULL,
        target_id TEXT NOT NULL,
        amount NUMERIC(18,2) NOT NULL,
        reason TEXT,
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS profiles(
        user_id TEXT PRIMARY KEY,
        display_name TEXT NOT NULL,
        name_lower TEXT NOT NULL UNIQUE,
        xp INTEGER NOT NULL DEFAULT 0,
        role TEXT NOT NULL DEFAULT 'member',
        is_anon BOOLEAN NOT NULL DEFAULT FALSE,
        referred_by TEXT,
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    )""")
    # crash tables (if not present already)
    cur.execute("""CREATE TABLE IF NOT EXISTS crash_rounds(
        id BIGSERIAL PRIMARY KEY,
        status TEXT NOT NULL,
        bust NUMERIC(10,2),
        betting_opens_at TIMESTAMPTZ NOT NULL,
        betting_ends_at TIMESTAMPTZ NOT NULL,
        started_at TIMESTAMPTZ,
        expected_end_at TIMESTAMPTZ,
        ended_at TIMESTAMPTZ
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS crash_bets(
        round_id BIGINT NOT NULL,
        user_id TEXT NOT NULL,
        bet NUMERIC(18,2) NOT NULL,
        cashout NUMERIC(8,2) NOT NULL,
        cashed_out NUMERIC(8,2),
        cashed_out_at TIMESTAMPTZ,
        win NUMERIC(18,2) NOT NULL DEFAULT 0,
        resolved BOOLEAN NOT NULL DEFAULT FALSE,
        PRIMARY KEY(round_id, user_id)
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS crash_games(
        id BIGSERIAL PRIMARY KEY,
        user_id TEXT NOT NULL,
        bet NUMERIC(18,2) NOT NULL,
        cashout NUMERIC(8,2) NOT NULL,
        bust NUMERIC(10,2) NOT NULL,
        win NUMERIC(18,2) NOT NULL,
        xp_gain INTEGER NOT NULL,
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    )""")
    # mines table (aligned with mines.py)
    cur.execute("""CREATE TABLE IF NOT EXISTS mines_games(
        id BIGSERIAL PRIMARY KEY,
        user_id TEXT NOT NULL,
        bet NUMERIC(18,2) NOT NULL,
        mines INTEGER NOT NULL,
        board TEXT NOT NULL,
        picks BIGINT NOT NULL DEFAULT 0,
        started_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        ended_at TIMESTAMPTZ,
        status TEXT NOT NULL DEFAULT 'active',
        seed TEXT NOT NULL,
        commit_hash TEXT NOT NULL,
        win NUMERIC(18,2) NOT NULL DEFAULT 0
    )""")

@with_conn
def apply_migrations(cur):
    cur.execute("ALTER TABLE profiles ADD COLUMN IF NOT EXISTS is_anon BOOLEAN NOT NULL DEFAULT FALSE")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_crash_games_created_at ON crash_games (created_at)")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_mines_games_started_at ON mines_games (started_at)")

@with_conn
def get_balance(cur, user_id: str) -> Decimal:
    cur.execute("SELECT balance FROM balances WHERE user_id=%s", (user_id,))
    r = cur.fetchone()
    return q2(r[0]) if r else Decimal("0.00")

@with_conn
def ensure_profile_row(cur, user_id: str):
    default_name = f"user_{user_id[-4:]}"
    cur.execute("""
        INSERT INTO profiles(user_id, display_name, name_lower)
        VALUES (%s,%s,%s)
        ON CONFLICT (user_id) DO NOTHING
    """, (user_id, default_name, default_name))

# ---------- Basic profile API ----------
@app.get("/api/me")
async def api_me(request: Request):
    try:
        s = _require_session(request)
        return {"id": s["id"], "username": s["username"], "avatar_url": s.get("avatar_url")}
    except:
        return {"id": None}

@app.get("/api/balance")
async def api_balance(request: Request):
    s = _require_session(request)
    return {"balance": float(get_balance(s["id"]))}

# ---------- Crash API ----------
class CrashBetIn(BaseModel):
    bet: str
    cashout: float = 2.0

@app.get("/api/crash/state")
async def api_crash_state(request: Request):
    try:
        s = _require_session(request)
        uid = s["id"]
    except:
        uid = None
    rid, info = ensure_betting_round()
    now = now_utc()
    if info["status"] == "betting" and now >= info["betting_ends_at"]:
        begin_running(rid); info = load_round()
    if info and info["status"] == "running" and info["expected_end_at"] and now >= info["expected_end_at"]:
        finish_round(rid); create_next_betting(); info = load_round()
    out = {
        "phase": info["status"],
        "bust": info["bust"],
        "betting_opens_at": iso(info["betting_opens_at"]),
        "betting_ends_at": iso(info["betting_ends_at"]),
        "started_at": iso(info["started_at"]),
        "expected_end_at": iso(info["expected_end_at"]),
        "last_busts": last_busts()
    }
    if info["status"] == "running":
        out["current_multiplier"] = current_multiplier(info["started_at"], info["expected_end_at"], info["bust"], now)
    if uid:
        y = your_bet(rid, uid)
        if y: out["your_bet"] = y
    return out

@app.post("/api/crash/place")
async def api_crash_place(request: Request, body: CrashBetIn):
    s = _require_session(request)
    return place_bet(s["id"], q2(D(body.bet or "0")), float(body.cashout or 2.0))

@app.post("/api/crash/cashout")
async def api_crash_cashout(request: Request):
    s = _require_session(request)
    cur = load_round()
    if not cur or cur["status"] != "running":
        raise HTTPException(400, "No running round")
    return cashout_now(s["id"])

@app.get("/api/crash/history")
async def api_crash_history(request: Request):
    s = _require_session(request)
    return {"rows": your_history(s["id"], 10)}

# ---------- Mines API ----------
class MinesStartIn(BaseModel):
    bet: str
    mines: int

@app.post("/api/mines/start")
async def api_mines_start(request: Request, body: MinesStartIn):
    s = _require_session(request)
    return mines_start(s["id"], q2(D(body.bet or "0")), int(body.mines))

@app.post("/api/mines/pick")
async def api_mines_pick(request: Request, index: int = Query(..., ge=0, le=24)):
    s = _require_session(request)
    return mines_pick(s["id"], index)

@app.post("/api/mines/cashout")
async def api_mines_cashout(request: Request):
    s = _require_session(request)
    return mines_cashout(s["id"])

@app.get("/api/mines/state")
async def api_mines_state(request: Request):
    s = _require_session(request)
    st = mines_state(s["id"])
    return st or {}

@app.get("/api/mines/history")
async def api_mines_history(request: Request):
    s = _require_session(request)
    return {"rows": mines_history(s["id"], 15)}

# ---------- HTML (UI) ----------
HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>GROWCB</title>
<style>
  :root{--bg:#0b0f14;--card:#121820;--muted:#8aa0b6;--acc:#57b6ff;--ok:#19c37d;--bad:#ff5c5c;}
  html,body{margin:0;padding:0;background:var(--bg);color:#e8f0f7;font-family:Inter,system-ui,Segoe UI,Arial,sans-serif}
  a{color:var(--acc);text-decoration:none}
  .wrap{max-width:1100px;margin:0 auto;padding:20px}
  header{display:flex;align-items:center;gap:12px;margin-bottom:16px}
  header img{height:36px}
  header .sp{flex:1}
  header .btn{padding:8px 12px;border-radius:8px;background:#1b2430;border:1px solid #223145}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:16px}
  .card{background:var(--card);border:1px solid #1d2a3a;border-radius:16px;overflow:hidden}
  .card img{width:100%;height:140px;object-fit:cover;background:#0c1117}
  .card .body{padding:12px}
  .row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
  .row input,.row select{background:#0d131b;border:1px solid #223145;color:#e8f0f7;border-radius:8px;padding:8px}
  .row .btn{background:#162233;border:1px solid #223145;padding:8px 12px;border-radius:8px;cursor:pointer}
  .row .btn.primary{background:#14324a;border-color:#25537a}
  .pill{padding:4px 8px;border-radius:999px;background:#0e1722;border:1px solid #1d2a3a;font-size:12px}
  .section{margin-top:26px}
  .hidden{display:none}
  .grid-5{display:grid;grid-template-columns:repeat(5,56px);gap:8px;justify-content:center}
  .tile{width:56px;height:56px;border-radius:10px;border:1px solid #223145;background:#0f1722;display:flex;align-items:center;justify-content:center;font-weight:700}
  .tile.g{background:#0f2b1b;border-color:#1e5a3d}
  .tile.b{background:#2b0f12;border-color:#6b1f2b}
  .hint{color:var(--muted);font-size:12px}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <img src="/img/GrowCBnobackground.png" onerror="this.style.display='none'"/>
    <div class="sp"></div>
    <span id="me" class="pill">Not signed in</span>
    <a class="btn" href="/demo_login">Sign in demo</a>
    <a class="btn" href="/logout">Logout</a>
  </header>

  <div class="section" id="home">
    <h2>Games</h2>
    <div class="grid">
      <div class="card">
        <img src="/img/mines.png" alt="Mines" onerror="this.style.display='none'"/>
        <div class="body">
          <h3>Mines</h3>
          <p class="hint">Pick tiles. Avoid bombs. Cash out anytime.</p>
          <div class="row"><button class="btn primary" onclick="nav('mines')">Open Mines</button></div>
        </div>
      </div>
      <div class="card">
        <img src="/img/crash.png" alt="Crash" onerror="this.style.display='none'"/>
        <div class="body">
          <h3>Crash</h3>
          <p class="hint">Place a bet and cash out before it crashes.</p>
          <div class="row"><button class="btn primary" onclick="nav('crash')">Open Crash</button></div>
        </div>
      </div>
      <div class="card">
        <img src="/img/coinflip.png" alt="Coinflip" onerror="this.style.display='none'"/>
        <div class="body"><h3>Coinflip</h3><p class="hint">Coming soon.</p></div>
      </div>
      <div class="card">
        <img src="/img/blackjack.png" alt="Blackjack" onerror="this.style.display='none'"/>
        <div class="body"><h3>Blackjack</h3><p class="hint">Coming soon.</p></div>
      </div>
      <div class="card">
        <img src="/img/pump.png" alt="Pump" onerror="this.style.display='none'"/>
        <div class="body"><h3>Pump</h3><p class="hint">Coming soon.</p></div>
      </div>
    </div>
  </div>

  <div class="section hidden" id="mines">
    <h2>Mines</h2>
    <div class="row">
      <input id="m_bet" type="number" step="0.01" value="10.00" placeholder="Bet (DL)"/>
      <select id="m_mines">
        <option value="3">3 mines</option>
        <option value="5">5 mines</option>
        <option value="10">10 mines</option>
        <option value="15">15 mines</option>
      </select>
      <button class="btn primary" onclick="minesStart()">Start</button>
      <button class="btn" onclick="minesCash()">Cash out</button>
      <span class="pill" id="m_stat">—</span>
      <span class="pill" id="m_pot">Potential: —</span>
    </div>
    <div class="grid-5" id="m_grid"></div>
    <div class="hint" style="margin-top:10px">Commit: <span id="m_hash">—</span></div>
  </div>

  <div class="section hidden" id="crash">
    <h2>Crash</h2>
    <div class="row">
      <input id="c_bet" type="number" step="0.01" value="10.00" placeholder="Bet (DL)"/>
      <input id="c_auto" type="number" step="0.01" value="2.00" placeholder="Auto cashout x"/>
      <button class="btn primary" onclick="crashPlace()">Place bet</button>
      <button class="btn" onclick="crashCash()">Cash out now</button>
      <span class="pill" id="c_phase">—</span>
      <span class="pill" id="c_mult">x1.00</span>
    </div>
    <div class="hint">Recent: <span id="c_hist">—</span></div>
  </div>

</div>

<script>
const qs = id => document.getElementById(id);

// ✅ Fixed JSON checker (this bug blanked your whole site before)
const j = async (url, init) => {
  const r = await fetch(url, init);
  if(!r.ok){
    let t = await r.text().catch(()=> '');
    try{ const js = JSON.parse(t); throw new Error(js.detail || js.message || t || r.statusText); }
    catch{ throw new Error(t || r.statusText); }
  }
  const ct = (r.headers.get('content-type')||'').toLowerCase();
  return (ct.includes('application/json') || ct.includes('+json')) ? r.json() : r.text();
};

function nav(id){
  for(const el of document.querySelectorAll('.section')) el.classList.add('hidden');
  qs(id).classList.remove('hidden');
  if(id==='crash'){ pollCrash(true); }
  if(id==='mines'){ pollMines(true); }
}

async function boot(){
  try{
    const me = await j('/api/me');
    if(me.id){
      qs('me').textContent = me.username || me.id;
    }else{
      qs('me').textContent = 'Not signed in';
    }
  }catch(e){}
}
boot();

// -------- Mines --------
let m_timer=null;
function drawMinesGrid(st){
  const g = qs('m_grid'); g.innerHTML='';
  const reveals = st.reveals || Array.from({length:25},()=> 'u');
  reveals.forEach((rv,i)=>{
    const d = document.createElement('div');
    d.className = 'tile ' + (rv==='g'?'g':rv==='b'?'b':'');
    d.textContent = rv==='g'?'✓':rv==='b'?'✗':'';
    d.onclick = ()=> minesPick(i);
    g.appendChild(d);
  });
  qs('m_stat').textContent = `Status: ${st.status}`;
  qs('m_hash').textContent = st.commit_hash || st.hash || '—';
  qs('m_pot').textContent  = st.potential_win ? `Potential: 💎 ${Number(st.potential_win).toFixed(2)} DL` : 'Potential: —';
}
async function pollMines(kick=false){
  clearTimeout(m_timer);
  try{
    const st = await j('/api/mines/state');
    if(st && st.id){ drawMinesGrid(st); }
  }catch(e){}
  m_timer = setTimeout(pollMines, 1500);
}
async function minesStart(){
  const bet = qs('m_bet').value || '0';
  const mines = Number(qs('m_mines').value||'3');
  await j('/api/mines/start', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({bet, mines})});
  pollMines(true);
}
async function minesPick(idx){
  await j('/api/mines/pick?index='+idx, {method:'POST'});
  pollMines(true);
}
async function minesCash(){
  await j('/api/mines/cashout', {method:'POST'});
  pollMines(true);
}

// -------- Crash --------
let c_timer=null;
async function pollCrash(kick=false){
  clearTimeout(c_timer);
  try{
    const st = await j('/api/crash/state');
    qs('c_phase').textContent = `Phase: ${st.phase}`;
    const m = st.current_multiplier || 1.00;
    qs('c_mult').textContent = 'x'+Number(m).toFixed(2);
    qs('c_hist').textContent = (st.last_busts||[]).slice(0,10).map(x=>'x'+Number(x).toFixed(2)).join('  ');
  }catch(e){}
  c_timer = setTimeout(pollCrash, 1000);
}
async function crashPlace(){
  const bet = qs('c_bet').value || '0';
  const cashout = Number(qs('c_auto').value || '2.00');
  await j('/api/crash/place', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({bet, cashout})});
  pollCrash(true);
}
async function crashCash(){
  await j('/api/crash/cashout', {method:'POST'});
  pollCrash(true);
}

// default view
nav('home');
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(HTML)

# ---------- Run locally ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
