# app/main.py — single-file FastAPI app with balance + profile UI and game stubs
import os, json, asyncio, re, random, string, datetime
from decimal import Decimal, ROUND_DOWN, getcontext
from typing import Optional, Dict, Any, List

import uvicorn
import httpx
from fastapi import FastAPI, Request, HTTPException, Query, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, Response, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from itsdangerous import URLSafeSerializer, BadSignature

# ---------------- Config ----------------
getcontext().prec = 28

PORT = int(os.getenv("PORT", "8080"))
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")
DISCORD_API = "https://discord.com/api"
CLIENT_ID = os.getenv("DISCORD_CLIENT_ID", "")
CLIENT_SECRET = os.getenv("DISCORD_CLIENT_SECRET", "")
OAUTH_REDIRECT = os.getenv("OAUTH_REDIRECT", "http://localhost:8080/callback")

TOKEN_COOKIE = "sid"
ser = URLSafeSerializer(SECRET_KEY)

# ---------------- Fake DB (replace with real DB later) ----------------
# Minimal in-memory store: balances, profiles, guild membership, leaderboards, promos
FAKE_USERS: Dict[str, Dict[str, Any]] = {}
LEADERBOARD: List[Dict[str, Any]] = []
PROMOS: Dict[str, Decimal] = {"WELCOME10": Decimal("10.00")}
GUILD_ID = os.getenv("DISCORD_GUILD_ID", "")

def seed_user(uid: str, username: str, avatar_url: str = "", in_guild: bool = True):
    if uid not in FAKE_USERS:
        FAKE_USERS[uid] = {
            "id": uid,
            "username": username,
            "avatar_url": avatar_url,
            "in_guild": in_guild,
            "balance": Decimal("100.00"),
            "created_at": datetime.datetime.utcnow().isoformat(),
            "roles": ["member"],
            "level": 3,
            "wins": 12,
            "losses": 9
        }

seed_user("1", "DemoUser", "https://cdn.discordapp.com/embed/avatars/1.png", True)

# ---------------- Helpers ----------------
def fmt_dl(x: Decimal) -> str:
    q = x.quantize(Decimal(".01"), rounding=ROUND_DOWN)
    return f"{q.normalize():f}"

def current_user(request: Request) -> Optional[Dict[str, Any]]:
    token = request.cookies.get(TOKEN_COOKIE)
    if not token:
        return None
    try:
        payload = ser.loads(token)
        uid = payload.get("uid")
        return FAKE_USERS.get(uid)
    except BadSignature:
        return None

def require_user(request: Request) -> Dict[str, Any]:
    user = current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in")
    return user

# ---------------- App ----------------
app = FastAPI(title="GROWCB")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# ---------------- Auth (very simplified demo) ----------------
@app.get("/login")
async def login_demo():
    # In a real app, redirect to Discord OAuth — here we just log in as DemoUser
    resp = RedirectResponse(url="/")
    resp.set_cookie(TOKEN_COOKIE, ser.dumps({"uid": "1"}), httponly=True, samesite="lax")
    return resp

@app.get("/logout")
async def logout():
    resp = RedirectResponse(url="/")
    resp.delete_cookie(TOKEN_COOKIE)
    return resp

# ---------------- API: User/Me/Profile/Balance ----------------
@app.get("/api/me")
async def api_me(request: Request):
    user = current_user(request)
    if not user:
        return JSONResponse({"authenticated": False}, status_code=200)
    return {
        "authenticated": True,
        "id": user["id"],
        "username": user["username"],
        "avatar_url": user["avatar_url"],
        "in_guild": user.get("in_guild", False)
    }

@app.get("/api/balance")
async def api_balance(request: Request):
    user = require_user(request)
    return {"balance": float(user["balance"])}

@app.post("/api/balance/add")
async def api_balance_add(request: Request, amount: Decimal = Query(..., gt=0)):
    user = require_user(request)
    user["balance"] += amount
    return {"ok": True, "balance": float(user["balance"])}

@app.get("/api/profile")
async def api_profile(request: Request):
    user = require_user(request)
    # Expand with more fields as needed
    return {
        "id": user["id"],
        "username": user["username"],
        "avatar_url": user["avatar_url"],
        "in_guild": user.get("in_guild", False),
        "balance": float(user["balance"]),
        "level": user.get("level", 1),
        "roles": user.get("roles", []),
        "wins": user.get("wins", 0),
        "losses": user.get("losses", 0),
        "created_at": user.get("created_at")
    }

# ---------------- API: Leaderboard / Promo ----------------
@app.get("/api/leaderboard")
async def api_leaderboard():
    # Simple fake board based on wins
    board = sorted(
        [
            {
                "username": u["username"],
                "wins": u.get("wins", 0),
                "losses": u.get("losses", 0),
                "balance": float(u["balance"])
            }
            for u in FAKE_USERS.values()
        ],
        key=lambda r: (r["wins"], r["balance"]),
        reverse=True
    )[:25]
    return {"items": board}

@app.post("/api/promo/apply")
async def api_promo_apply(request: Request, code: str = Query(...)):
    user = require_user(request)
    up = code.strip().upper()
    if up not in PROMOS:
        return JSONResponse({"ok": False, "error": "Invalid code"}, status_code=400)
    val = PROMOS.pop(up)
    user["balance"] += val
    return {"ok": True, "added": float(val), "balance": float(user["balance"])}

# ---------------- API: Games (stubs) ----------------
# Crash: minimal skeleton
CRASH_ROUND = {
    "id": 1,
    "status": "betting",   # betting -> running -> finished
    "multiplier": 1.0,
    "bets": {}  # uid -> amount
}

@app.get("/api/crash/state")
async def api_crash_state():
    return {
        "round_id": CRASH_ROUND["id"],
        "status": CRASH_ROUND["status"],
        "multiplier": CRASH_ROUND["multiplier"]
    }

@app.post("/api/crash/bet")
async def api_crash_bet(request: Request, amount: Decimal = Query(..., gt=0)):
    user = require_user(request)
    if CRASH_ROUND["status"] != "betting":
        return JSONResponse({"ok": False, "error": "Betting closed"}, status_code=400)
    if user["balance"] < amount:
        return JSONResponse({"ok": False, "error": "Insufficient balance"}, status_code=400)
    user["balance"] -= amount
    CRASH_ROUND["bets"][user["id"]] = CRASH_ROUND["bets"].get(user["id"], Decimal("0")) + amount
    return {"ok": True, "balance": float(user["balance"])}

@app.post("/api/crash/cashout")
async def api_crash_cashout(request: Request):
    user = require_user(request)
    if CRASH_ROUND["status"] != "running":
        return JSONResponse({"ok": False, "error": "Round not running"}, status_code=400)
    bet = CRASH_ROUND["bets"].get(user["id"], Decimal("0"))
    if bet <= 0:
        return JSONResponse({"ok": False, "error": "No active bet"}, status_code=400)
    win = bet * Decimal(str(CRASH_ROUND["multiplier"]))
    user["balance"] += win
    CRASH_ROUND["bets"][user["id"]] = Decimal("0")
    return {"ok": True, "won": float(win), "balance": float(user["balance"])}

# Mines: 5x5 board, 5 mines — minimal demo
MINES_GAMES: Dict[str, Dict[str, Any]] = {}

def new_mines_game(uid: str) -> Dict[str, Any]:
    mines = set(random.sample(range(25), 5))
    return {
        "board": ["hidden"] * 25,
        "mines": mines,
        "revealed": set(),
        "bet": Decimal("0"),
        "active": False
    }

@app.post("/api/mines/start")
async def api_mines_start(request: Request, bet: Decimal = Query(..., gt=0)):
    user = require_user(request)
    if user["balance"] < bet:
        return JSONResponse({"ok": False, "error": "Insufficient balance"}, status_code=400)
    g = MINES_GAMES.get(user["id"]) or new_mines_game(user["id"])
    g["board"] = ["hidden"] * 25
    g["mines"] = set(random.sample(range(25), 5))
    g["revealed"] = set()
    g["bet"] = bet
    g["active"] = True
    user["balance"] -= bet
    MINES_GAMES[user["id"]] = g
    return {"ok": True, "balance": float(user["balance"])}

@app.get("/api/mines/state")
async def api_mines_state(request: Request):
    user = require_user(request)
    g = MINES_GAMES.get(user["id"])
    if not g:
        g = new_mines_game(user["id"])
        MINES_GAMES[user["id"]] = g
    return {
        "active": g["active"],
        "board": g["board"],
        "revealed": list(g["revealed"]),
        "bet": float(g["bet"]),
        "count_mines": len(g["mines"])
    }

# IMPORTANT: fixed path (no stray dot) and typed query bounds
@app.post("/api/mines/pick")
async def api_mines_pick(request: Request, index: int = Query(..., ge=0, le=24)):
    user = require_user(request)
    g = MINES_GAMES.get(user["id"])
    if not g or not g["active"]:
        return JSONResponse({"ok": False, "error": "No active game"}, status_code=400)
    if index in g["revealed"]:
        return {"ok": True, "board": g["board"], "hit": False, "already": True}

    if index in g["mines"]:
        # hit mine — lose
        g["board"][index] = "mine"
        g["active"] = False
        return {"ok": True, "hit": True, "board": g["board"], "payout": 0.0}
    else:
        g["revealed"].add(index)
        g["board"][index] = "safe"
        # tiny progressive payout preview (don’t pay yet; pay on cashout)
        revealed = len(g["revealed"])
        payout_preview = float((g["bet"] * Decimal("0.25")) + (g["bet"] * Decimal(revealed) * Decimal("0.05")))
        return {"ok": True, "hit": False, "board": g["board"], "payout_preview": payout_preview}

@app.post("/api/mines/cashout")
async def api_mines_cashout(request: Request):
    user = require_user(request)
    g = MINES_GAMES.get(user["id"])
    if not g or not g["active"]:
        return JSONResponse({"ok": False, "error": "No active game"}, status_code=400)
    revealed = len(g["revealed"])
    payout = (g["bet"] * Decimal("0.25")) + (g["bet"] * Decimal(revealed) * Decimal("0.05"))
    user["balance"] += payout
    g["active"] = False
    return {"ok": True, "payout": float(payout), "balance": float(user["balance"])}

# ---------------- HTML + JS ----------------
HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>GROWCB</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    :root { --bg:#0a0f1e; --card:#121a2f; --muted:#a9b1c7; --accent:#6ea0ff; --good:#2ecc71; --bad:#ff5c5c; }
    *{box-sizing:border-box}
    body{margin:0;background:linear-gradient(180deg,#0b1226,#090e1b);color:#eaf0ff;font-family:Inter,system-ui,Segoe UI,Roboto,Arial}
    header{position:sticky;top:0;background:rgba(10,15,30,.72);backdrop-filter:blur(8px);border-bottom:1px solid #1b2544;z-index:10}
    .wrap{max-width:1100px;margin:0 auto;padding:14px 16px;display:flex;align-items:center;gap:12px}
    .brand{font-weight:800;letter-spacing:.5px}
    nav{margin-left:16px;display:flex;gap:8px}
    .btn, .tab{border:1px solid #26335c;background:#162042;color:#eaf0ff;padding:8px 12px;border-radius:10px;cursor:pointer;user-select:none;transition:.15s}
    .btn:hover, .tab:hover{transform:translateY(-1px);border-color:#3550a5}
    .btn.primary{background:#2a3d7a;border-color:#4462cc}
    .chip{border:1px solid #2b3c73;background:#111a34;color:#dbe6ff;padding:6px 10px;border-radius:999px;font-size:13px}
    .grow{flex:1}
    .right{margin-left:auto;display:flex;align-items:center;gap:10px}
    .avatar-wrap{position:relative}
    .avatar{width:32px;height:32px;border-radius:50%;border:2px solid #2b3c73;cursor:pointer}
    .menu{position:absolute;right:0;top:42px;background:#101832;border:1px solid #223059;border-radius:12px;display:none;min-width:180px;overflow:hidden}
    .menu .item{padding:10px 12px;cursor:pointer}
    .menu .item:hover{background:#142047}
    main{max-width:1100px;margin:22px auto;padding:0 16px}
    .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px}
    .card{background:#121a2f;border:1px solid #1b2544;border-radius:14px;padding:12px}
    .card h3{margin:6px 0 8px 0}
    .hidden{display:none!important}
    .good{color:var(--good)} .bad{color:var(--bad)}
    .row{display:flex;gap:10px;align-items:center}
    .lb{width:100%;border-collapse:collapse}
    .lb th,.lb td{border-bottom:1px solid #1b2544;padding:8px;text-align:left;font-size:14px}
    .mono{font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace}
    a{color:#9bb8ff;text-decoration:none}
    a:hover{text-decoration:underline}
  </style>
</head>
<body>
<header>
  <div class="wrap">
    <div class="brand"><a id="homeLink" href="/">GROWCB</a></div>
    <nav>
      <button class="tab" id="tab-games">Games</button>
      <button class="tab" id="tab-ref">Referral</button>
      <button class="tab" id="tab-promo">Promo</button>
      <button class="tab" id="tab-lb">Leaderboard</button>
      <button class="tab" id="tab-profile">Profile</button>
    </nav>
    <div class="grow"></div>
    <div class="right" id="authArea">
      <!-- filled by JS: join/login | balance chip | avatar menu -->
    </div>
  </div>
</header>

<main>
  <!-- Games page -->
  <section id="page-games">
    <h2>Games</h2>
    <div class="cards">
      <div class="card">
        <h3>Crash</h3>
        <div class="row">
          <button class="btn" id="crashBet5">Bet 5</button>
          <button class="btn" id="crashCash">Cashout</button>
        </div>
        <div id="crashState" class="mono" style="margin-top:8px"></div>
      </div>

      <div class="card">
        <h3>Mines</h3>
        <div class="row">
          <button class="btn" id="minesStart5">Start (5)</button>
          <button class="btn" id="minesCash">Cashout</button>
        </div>
        <div id="minesGrid" style="display:grid;grid-template-columns:repeat(5,40px);gap:6px;margin-top:10px"></div>
        <div id="minesInfo" class="mono" style="margin-top:8px"></div>
      </div>
    </div>
  </section>

  <!-- Referral page -->
  <section id="page-ref" class="hidden">
    <h2>Referral</h2>
    <div class="card">
      <p>Invite friends and earn a bonus. (Demo content)</p>
      <div class="mono">Your code: <span id="refCode">DEMO-1234</span></div>
    </div>
  </section>

  <!-- Promo page -->
  <section id="page-promo" class="hidden">
    <h2>Promo</h2>
    <div class="row">
      <input id="promoCode" placeholder="Enter code" style="padding:8px;border-radius:8px;border:1px solid #1b2544;background:#0f1730;color:#eaf0ff"/>
      <button id="applyPromo" class="btn primary">Apply</button>
    </div>
    <div id="promoMsg" class="mono" style="margin-top:8px"></div>
  </section>

  <!-- Leaderboard page -->
  <section id="page-lb" class="hidden">
    <h2>Leaderboard</h2>
    <table class="lb" id="lbTable">
      <thead><tr><th>User</th><th>Wins</th><th>Losses</th><th>Balance</th></tr></thead>
      <tbody></tbody>
    </table>
  </section>

  <!-- Profile page -->
  <section id="page-profile" class="hidden">
    <h2>Profile</h2>
    <div class="card" id="profileBox">
      <div class="row">
        <img id="pAvatar" src="" style="width:56px;height:56px;border-radius:50%;border:2px solid #2b3c73"/>
        <div>
          <div id="pName" style="font-weight:700"></div>
          <div id="pMeta" class="mono" style="font-size:13px;color:#a9b1c7"></div>
        </div>
      </div>
      <div class="row" style="margin-top:10px">
        <span class="chip">Balance: <strong id="pBalance">0.00</strong></span>
        <span class="chip">Level: <strong id="pLevel">1</strong></span>
        <span class="chip">Roles: <strong id="pRoles">member</strong></span>
      </div>
      <div class="mono" id="pStats" style="margin-top:10px"></div>
    </div>
  </section>
</main>

<script>
  // ------- Small helpers -------
  const qs = (id)=>document.getElementById(id);
  async function j(url, opts={}){
    const r = await fetch(url, opts);
    const ct = r.headers.get('content-type')||'';
    return ct.includes('application/json') ? r.json() : r.text();
  }
  function showOnly(id){
    for(const sec of document.querySelectorAll('main > section')) sec.classList.add('hidden');
    qs(id).classList.remove('hidden');
  }

  // ------- Header: show balance + profile menu -------
  async function renderHeader(){
    const el = qs('authArea');
    try{
      const me = await j('/api/me');
      if(!me.authenticated){
        el.innerHTML = '<a class="btn primary" href="/login">Login with Discord</a>';
        return;
      }
      const bal = await j('/api/balance');
      el.innerHTML = `
        <button class="btn primary" id="btnJoinSmall">${me.in_guild ? 'In Discord' : 'Join Discord'}</button>
        <span class="chip">Balance: <strong id="hdrBalance">${(bal.balance||0).toFixed(2)}</strong></span>
        <div class="avatar-wrap">
          <img class="avatar" id="avatarBtn" src="${me.avatar_url||''}" title="${me.username||'user'}"/>
          <div id="userMenu" class="menu">
            <div class="item" id="menuProfile">Profile</div>
            <div class="item" id="menuSettings">Settings</div>
            <a class="item" href="/logout">Logout</a>
          </div>
        </div>
      `;
      // avatar menu
      const menu = qs('userMenu');
      qs('avatarBtn').onclick = ()=>{ menu.style.display = (menu.style.display==='block'?'none':'block'); };
      document.addEventListener('click', (e)=>{ if(!e.target.closest('.avatar-wrap')) menu.style.display='none'; });
      qs('menuProfile').onclick = ()=>{ showOnly('page-profile'); renderProfile(); };
      qs('menuSettings').onclick = ()=> alert('Settings (demo)');

    }catch(e){
      el.innerHTML = '<a class="btn primary" href="/login">Login with Discord</a>';
    }
  }

  // ------- Tabs / Nav -------
  qs('homeLink').onclick = (e)=>{ e.preventDefault(); showOnly('page-games'); };
  qs('tab-games').onclick = ()=> showOnly('page-games');
  qs('tab-ref').onclick   = ()=> showOnly('page-ref');
  qs('tab-promo').onclick = ()=> showOnly('page-promo');
  qs('tab-lb').onclick    = ()=> { showOnly('page-lb'); refreshLeaderboard(); };
  qs('tab-profile').onclick = ()=> { showOnly('page-profile'); renderProfile(); };

  // ------- Games: Crash -------
  async function refreshCrash(){
    try{
      const st = await j('/api/crash/state');
      qs('crashState').textContent = `Round ${st.round_id} | ${st.status} | x${st.multiplier.toFixed ? st.multiplier.toFixed(2) : st.multiplier}`;
    }catch(_){}
  }
  async function crashBet(v){
    try{
      const r = await j('/api/crash/bet?amount='+encodeURIComponent(v), {method:'POST'});
      if(r.ok){ await refreshHeaderBalance(); refreshCrash(); } else alert(r.error||'Bet failed');
    }catch(e){ alert('Error'); }
  }
  async function crashCash(){
    try{
      const r = await j('/api/crash/cashout', {method:'POST'});
      if(r.ok){ await refreshHeaderBalance(); refreshCrash(); } else alert(r.error||'Cashout failed');
    }catch(e){ alert('Error'); }
  }
  qs('crashBet5').onclick = ()=> crashBet(5);
  qs('crashCash').onclick = ()=> crashCash();

  // ------- Games: Mines -------
  function drawMinesGrid(board){
    const m = qs('minesGrid'); m.innerHTML = '';
    for(let i=0;i<25;i++){
      const b = document.createElement('button');
      b.className = 'btn';
      b.style.width='40px'; b.style.height='40px'; b.textContent = board[i]==='safe'?'·':'?';
      b.onclick = ()=> pickMine(i);
      m.appendChild(b);
    }
  }
  async function refreshMines(){
    const s = await j('/api/mines/state');
    drawMinesGrid(s.board||[]);
    qs('minesInfo').textContent = s.active ? `Bet: ${s.bet.toFixed(2)} | Mines: ${s.count_mines}` : 'No active game';
  }
  async function startMines(v){
    const r = await j('/api/mines/start?bet='+encodeURIComponent(v), {method:'POST'});
    if(r.ok){ await refreshHeaderBalance(); refreshMines(); } else alert(r.error||'Start failed');
  }
  async function pickMine(i){
    const r = await j('/api/mines/pick?index='+i, {method:'POST'});
    if(r.ok){
      drawMinesGrid(r.board||[]);
      if(r.hit){ alert('Boom! You hit a mine.'); }
      if(r.payout_preview) qs('minesInfo').textContent = `Preview payout: ${r.payout_preview.toFixed(2)}`;
    } else alert(r.error||'Pick failed');
  }
  async function cashoutMines(){
    const r = await j('/api/mines/cashout', {method:'POST'});
    if(r.ok){ alert('You cashed out: '+r.payout.toFixed(2)); await refreshHeaderBalance(); refreshMines(); }
    else alert(r.error||'Cashout failed');
  }
  qs('minesStart5').onclick = ()=> startMines(5);
  qs('minesCash').onclick = ()=> cashoutMines();

  // ------- Promo -------
  qs('applyPromo').onclick = async ()=>{
    const code = (qs('promoCode').value||'').trim();
    if(!code) return;
    const r = await j('/api/promo/apply?code='+encodeURIComponent(code), {method:'POST'});
    if(r.ok){ qs('promoMsg').textContent = `Added ${r.added.toFixed(2)}!`; refreshHeaderBalance(); }
    else qs('promoMsg').textContent = r.error||'Error';
  };

  // ------- Leaderboard -------
  async function refreshLeaderboard(){
    const r = await j('/api/leaderboard');
    const tb = qs('lbTable').querySelector('tbody'); tb.innerHTML = '';
    for(const it of (r.items||[])){
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${it.username}</td><td>${it.wins}</td><td>${it.losses}</td><td>${it.balance.toFixed(2)}</td>`;
      tb.appendChild(tr);
    }
  }

  // ------- Profile -------
  async function renderProfile(){
    const p = await j('/api/profile');
    qs('pAvatar').src = p.avatar_url||'';
    qs('pName').textContent = p.username||'user';
    qs('pMeta').textContent = `ID ${p.id} • Joined: ${p.created_at?.slice(0,10)||'—'}`;
    qs('pBalance').textContent = (p.balance||0).toFixed(2);
    qs('pLevel').textContent = p.level||1;
    qs('pRoles').textContent = (p.roles||[]).join(', ') || 'member';
    qs('pStats').textContent = `Wins: ${p.wins||0} | Losses: ${p.losses||0}`;
  }

  async function refreshHeaderBalance(){
    try{
      const bal = await j('/api/balance');
      const hb = document.getElementById('hdrBalance');
      if(hb) hb.textContent = (bal.balance||0).toFixed(2);
      const pb = document.getElementById('pBalance');
      if(pb) pb.textContent = (bal.balance||0).toFixed(2);
    }catch(_){}
  }

  // ------- Boot -------
  (async function boot(){
    await renderHeader();
    await refreshCrash();
    await refreshMines();
    showOnly('page-games');
  })();
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_TEMPLATE
# ---------------- Background tasks (Crash engine) ----------------
# Very lightweight demo loop that cycles rounds:
# - 6s betting phase
# - running phase where multiplier rises until a random bust
CRASH_TASK: asyncio.Task | None = None

async def crash_engine():
    global CRASH_ROUND
    rid = CRASH_ROUND["id"]
    while True:
        try:
            # --- Betting window ---
            CRASH_ROUND["status"] = "betting"
            CRASH_ROUND["multiplier"] = 1.0
            # Keep existing bets for current round; players may add during betting
            await asyncio.sleep(6)  # seconds of betting

            # If nobody bet, just skip to next round quickly
            if not any(v > 0 for v in CRASH_ROUND["bets"].values()):
                # advance round id
                rid += 1
                CRASH_ROUND = {"id": rid, "status": "betting", "multiplier": 1.0, "bets": {}}
                await asyncio.sleep(1)
                continue

            # --- Running phase ---
            CRASH_ROUND["status"] = "running"
            # exponential-ish growth with a random bust threshold
            # bust at a random point between x1.2 and x5.0
            bust_at = random.uniform(1.2, 5.0)
            m = 1.0
            while m < bust_at and CRASH_ROUND["status"] == "running":
                await asyncio.sleep(0.25)
                m *= 1.03 + random.uniform(0.00, 0.02)  # gentle increase
                CRASH_ROUND["multiplier"] = round(m, 2)

            # --- Bust ---
            CRASH_ROUND["status"] = "finished"
            # Any remaining players who didn't cash out lose their bet; bets reset
            CRASH_ROUND["bets"] = {}
            await asyncio.sleep(2)

            # --- Next round ---
            rid += 1
            CRASH_ROUND = {"id": rid, "status": "betting", "multiplier": 1.0, "bets": {}}
            await asyncio.sleep(1)

        except asyncio.CancelledError:
            # graceful shutdown
            break
        except Exception:
            # keep engine alive on unexpected errors
            await asyncio.sleep(1)

@app.on_event("startup")
async def _startup():
    global CRASH_TASK
    if CRASH_TASK is None or CRASH_TASK.done():
        CRASH_TASK = asyncio.create_task(crash_engine())

@app.on_event("shutdown")
async def _shutdown():
    global CRASH_TASK
    if CRASH_TASK and not CRASH_TASK.done():
        CRASH_TASK.cancel()
        try:
            await CRASH_TASK
        except asyncio.CancelledError:
            pass

# ---------------- Health & static helpers ----------------
@app.get("/health")
async def health():
    return {"ok": True, "time": datetime.datetime.utcnow().isoformat()}

# Optional: serve image placeholders if you want to drop files in /img
from fastapi.staticfiles import StaticFiles
IMG_DIR = os.getenv("IMG_DIR", "img")
if os.path.isdir(IMG_DIR):
    app.mount("/img", StaticFiles(directory=IMG_DIR), name="img")

# ---------------- Entrypoint ----------------
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=os.getenv("RELOAD", "0") == "1",
        log_level=os.getenv("LOG_LEVEL", "info")
    )
