# app/main.py — full GrowCB site + Discord bot

import os, json, asyncio, re, random, string, datetime, base64
from urllib.parse import urlencode
from typing import Optional, Dict, List
from decimal import Decimal, ROUND_DOWN, getcontext

import httpx
import psycopg
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from itsdangerous import URLSafeSerializer, BadSignature
from pydantic import BaseModel

# Discord bot
import discord
from discord.ext import commands

# ---------- Import games ----------
from crash import (
    ensure_betting_round, place_bet, load_round, begin_running,
    finish_round, create_next_betting, last_busts, your_bet,
    your_history, cashout_now, current_multiplier
)
from mines import (
    mines_start, mines_pick, mines_cashout, mines_state, mines_history
)

# ---------- Config ----------
getcontext().prec = 28

PREFIX = "."
CLIENT_ID = os.getenv("DISCORD_CLIENT_ID", "")
CLIENT_SECRET = os.getenv("DISCORD_CLIENT_SECRET") or os.getenv("CLIENT_SECRET", "")
OAUTH_REDIRECT = os.getenv("OAUTH_REDIRECT", "")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")
PORT = int(os.getenv("PORT", "8080"))
DISCORD_API = "https://discord.com/api"
OWNER_ID = int(os.getenv("OWNER_ID", "1128658280546320426"))
DATABASE_URL = os.getenv("DATABASE_URL")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")

GEM = "💎"
TWO = Decimal("0.01")
def D(x): return Decimal(str(x)) if not isinstance(x, Decimal) else x
def q2(x): return D(x).quantize(TWO, rounding=ROUND_DOWN)

UTC = datetime.timezone.utc
def now_utc(): return datetime.datetime.now(UTC)

# ---------- FastAPI + static ----------
app = FastAPI()
base = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(base, "static")), name="static")

# ---------- Sessions ----------
SER = URLSafeSerializer(SECRET_KEY, salt="session-v1")
def _set_session(resp, data): resp.set_cookie("session", SER.dumps(data), max_age=30*86400, httponly=True, samesite="lax")
def _clear_session(resp): resp.delete_cookie("session")
def _require_session(request: Request):
    raw = request.cookies.get("session")
    if not raw: raise HTTPException(401, "Not logged in")
    try: return SER.loads(raw)
    except BadSignature: raise HTTPException(401, "Invalid session")

# ---------- DB ----------
def with_conn(fn):
    def wrapper(*a, **kw):
        if not DATABASE_URL: raise RuntimeError("DATABASE_URL not set")
        with psycopg.connect(DATABASE_URL) as con, con.cursor() as cur:
            res = fn(cur, *a, **kw)
            con.commit(); return res
    return wrapper

@with_conn
def init_db(cur):
    cur.execute("""CREATE TABLE IF NOT EXISTS balances (
        user_id TEXT PRIMARY KEY,
        balance NUMERIC(18,2) NOT NULL DEFAULT 0
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS balance_log (
        id BIGSERIAL PRIMARY KEY,
        actor_id TEXT NOT NULL,
        target_id TEXT NOT NULL,
        amount NUMERIC(18,2) NOT NULL,
        reason TEXT,
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS profiles (
        user_id TEXT PRIMARY KEY,
        display_name TEXT NOT NULL,
        name_lower TEXT NOT NULL UNIQUE,
        xp INTEGER NOT NULL DEFAULT 0,
        role TEXT NOT NULL DEFAULT 'member',
        is_anon BOOLEAN NOT NULL DEFAULT FALSE,
        referred_by TEXT,
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS chat_messages (
        id BIGSERIAL PRIMARY KEY,
        user_id TEXT NOT NULL,
        username TEXT NOT NULL,
        text TEXT NOT NULL,
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    )""")

@with_conn
def get_balance(cur, uid: str) -> Decimal:
    cur.execute("SELECT balance FROM balances WHERE user_id=%s", (uid,))
    r = cur.fetchone(); return q2(r[0]) if r else Decimal("0.00")

@with_conn
def adjust_balance(cur, actor: str, target: str, amt, reason=None) -> Decimal:
    amt = q2(D(amt))
    cur.execute("INSERT INTO balances(user_id,balance) VALUES(%s,0) ON CONFLICT(user_id) DO NOTHING", (target,))
    cur.execute("UPDATE balances SET balance=balance+%s WHERE user_id=%s", (amt, target))
    cur.execute("INSERT INTO balance_log(actor_id,target_id,amount,reason) VALUES(%s,%s,%s,%s)",
                (actor,target,amt,reason))
    cur.execute("SELECT balance FROM balances WHERE user_id=%s",(target,))
    return q2(cur.fetchone()[0])

@with_conn
def tip_transfer(cur, from_id: str, to_id: str, amount: Decimal):
    amount = q2(D(amount))
    if amount <= 0: raise ValueError("Amount must be > 0")
    if from_id == to_id: raise ValueError("Cannot tip yourself")
    cur.execute("INSERT INTO balances(user_id,balance) VALUES(%s,0) ON CONFLICT(user_id) DO NOTHING",(from_id,))
    cur.execute("INSERT INTO balances(user_id,balance) VALUES(%s,0) ON CONFLICT(user_id) DO NOTHING",(to_id,))
    cur.execute("SELECT balance FROM balances WHERE user_id=%s FOR UPDATE",(from_id,))
    sbal = D(cur.fetchone()[0])
    if sbal < amount: raise ValueError("Insufficient balance")
    cur.execute("UPDATE balances SET balance=balance-%s WHERE user_id=%s",(amount,from_id))
    cur.execute("UPDATE balances SET balance=balance+%s WHERE user_id=%s",(amount,to_id))
    cur.execute("INSERT INTO balance_log(actor_id,target_id,amount,reason) VALUES(%s,%s,%s,%s)",
                (from_id,to_id,-amount,"tip"))
    cur.execute("INSERT INTO balance_log(actor_id,target_id,amount,reason) VALUES(%s,%s,%s,%s)",
                (from_id,to_id,amount,"tip"))
    return True

# ---------- Discord Bot ----------
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=PREFIX, intents=intents, help_command=None)

@bot.event
async def on_ready():
    print(f"🤖 Bot ready as {bot.user} (id={bot.user.id})")

@bot.command()
async def bal(ctx, user: discord.User=None):
    target = user or ctx.author
    balval = get_balance(str(target.id))
    await ctx.send(f"{target.mention} balance: {balval} {GEM}")

@bot.command()
async def tip(ctx, user: discord.User, amount: float):
    try:
        tip_transfer(str(ctx.author.id), str(user.id), D(amount))
        await ctx.send(f"Tipped {amount} {GEM} to {user.mention}")
    except Exception as e:
        await ctx.send(f"❌ {e}")

@bot.command()
async def help(ctx):
    await ctx.send(f"Commands: `{PREFIX}bal [@user]`, `{PREFIX}tip @user amount`")
# ---------- OAuth ----------
@app.get("/login")
async def login():
    if not (CLIENT_ID and OAUTH_REDIRECT):
        return HTMLResponse("OAuth not configured")
    params = {
        "client_id": CLIENT_ID,
        "redirect_uri": OAUTH_REDIRECT,
        "response_type": "code",
        "scope": "identify",
        "prompt": "consent"
    }
    return RedirectResponse(f"{DISCORD_API}/oauth2/authorize?{urlencode(params)}")

@app.get("/callback")
async def callback(code: str):
    if not (CLIENT_ID and CLIENT_SECRET and OAUTH_REDIRECT):
        return HTMLResponse("OAuth not configured")

    async with httpx.AsyncClient(timeout=15) as cx:
        data = {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": OAUTH_REDIRECT
        }
        r = await cx.post(f"{DISCORD_API}/oauth2/token", data=data,
                          headers={"Content-Type": "application/x-www-form-urlencoded"})
        if r.status_code != 200:
            return HTMLResponse(f"OAuth failed: {r.text}", status_code=400)
        tok = r.json()
        access = tok.get("access_token")
        u = await cx.get(f"{DISCORD_API}/users/@me", headers={"Authorization": f"Bearer {access}"})
        if u.status_code != 200:
            return HTMLResponse(f"User fetch failed: {u.text}", status_code=400)
        me = u.json()

    user_id = str(me["id"])
    username = me.get("username", "user")
    avatar_hash = me.get("avatar")
    avatar_url = f"https://cdn.discordapp.com/avatars/{user_id}/{avatar_hash}.png?size=64" if avatar_hash \
                 else "https://cdn.discordapp.com/embed/avatars/0.png"

    ensure_profile_row(user_id)
    resp = RedirectResponse("/")
    _set_session(resp, {"id": user_id, "username": username, "avatar_url": avatar_url})
    return resp

@app.get("/logout")
async def logout():
    resp = RedirectResponse("/")
    _clear_session(resp)
    return resp

# ---------- Profiles ----------
NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{3,20}$")

@with_conn
def ensure_profile_row(cur, user_id: str):
    cur.execute("""INSERT INTO profiles(user_id,display_name,name_lower,role)
                   VALUES(%s,%s,%s,%s)
                   ON CONFLICT(user_id) DO NOTHING""",
                (user_id, f"user_{user_id[-4:]}", f"user_{user_id[-4:]}",
                 "owner" if user_id == str(OWNER_ID) else "member"))

@with_conn
def profile_info(cur, user_id: str):
    ensure_profile_row(user_id)
    cur.execute("SELECT xp, role, is_anon FROM profiles WHERE user_id=%s", (user_id,))
    xp, role, is_anon = cur.fetchone()
    bal = get_balance(user_id)
    return {"id": user_id, "xp": xp, "role": role, "is_anon": is_anon, "balance": float(bal)}

# ---------- API ----------
@app.get("/api/me")
async def api_me(request: Request):
    s = _require_session(request)
    return {"id": s["id"], "username": s["username"], "avatar_url": s.get("avatar_url")}

@app.get("/api/balance")
async def api_balance(request: Request):
    s = _require_session(request)
    return {"balance": float(get_balance(s["id"]))}

@app.get("/api/profile")
async def api_profile(request: Request):
    s = _require_session(request)
    return profile_info(s["id"])

class AdjustIn(BaseModel):
    identifier: str
    amount: str
    reason: Optional[str] = None

@app.post("/api/admin/adjust")
async def api_admin_adjust(request: Request, body: AdjustIn):
    s = _require_session(request)
    if s["id"] != str(OWNER_ID):
        raise HTTPException(403, "Owner only")
    m = re.search(r"\d{5,}", body.identifier or "")
    if not m: raise HTTPException(400, "Invalid ID")
    newbal = adjust_balance(s["id"], m.group(0), D(body.amount), body.reason)
    return {"new_balance": float(newbal)}

# ---------- Crash ----------
class CrashBetIn(BaseModel):
    bet: str
    cashout: Optional[float] = None

@app.get("/api/crash/state")
async def api_crash_state():
    rid, info = ensure_betting_round()
    now = now_utc()
    if info["status"] == "betting" and now >= info["betting_ends_at"]:
        begin_running(rid); info = load_round()
    if info["status"] == "running" and info["expected_end_at"] and now >= info["expected_end_at"]:
        finish_round(rid); create_next_betting(); info = load_round()
    out = {"phase": info["status"], "bust": info["bust"], "last_busts": last_busts()}
    if info["status"] == "running":
        out["current_multiplier"] = current_multiplier(info["started_at"], info["expected_end_at"], info["bust"], now)
    return out

@app.post("/api/crash/place")
async def api_crash_place(request: Request, body: CrashBetIn):
    s = _require_session(request)
    bet = q2(D(body.bet))
    return place_bet(s["id"], bet, float(body.cashout or 2.0))

@app.post("/api/crash/cashout")
async def api_crash_cashout(request: Request):
    s = _require_session(request)
    return cashout_now(s["id"])

# ---------- Mines ----------
class MinesStartIn(BaseModel):
    bet: str
    mines: int

@app.post("/api/mines/start")
async def api_mines_start(request: Request, body: MinesStartIn):
    s = _require_session(request)
    return mines_start(s["id"], q2(D(body.bet)), body.mines)

@app.post("/api/mines/pick")
async def api_mines_pick(request: Request, index: int = Query(...)):
    s = _require_session(request)
    return mines_pick(s["id"], index)

@app.post("/api/mines/cashout")
async def api_mines_cashout(request: Request):
    s = _require_session(request)
    return mines_cashout(s["id"])

@app.get("/api/mines/state")
async def api_mines_state(request: Request):
    s = _require_session(request)
    return mines_state(s["id"]) or {}

# ---------- Chat ----------
class ChatIn(BaseModel):
    text: str

@with_conn
def chat_insert(cur, uid: str, username: str, text: str):
    cur.execute("INSERT INTO chat_messages(user_id,username,text) VALUES(%s,%s,%s) RETURNING id, created_at",
                (uid, username, text))
    r = cur.fetchone()
    return {"id": r[0], "created_at": str(r[1])}

@with_conn
def chat_fetch(cur, since: int, limit: int):
    cur.execute("SELECT id,user_id,username,text,created_at FROM chat_messages WHERE id>%s ORDER BY id ASC LIMIT %s",
                (since, limit))
    return [{"id":r[0],"user_id":r[1],"username":r[2],"text":r[3],"created_at":str(r[4])} for r in cur.fetchall()]

@app.post("/api/chat/send")
async def api_chat_send(request: Request, body: ChatIn):
    s = _require_session(request)
    return chat_insert(s["id"], s["username"], body.text.strip())

@app.get("/api/chat/fetch")
async def api_chat_fetch(since: int = 0, limit: int = 30):
    return {"rows": chat_fetch(since, limit)}

# ---------- Simple UI ----------
HTML_TEMPLATE = """
<!doctype html>
<html><head><title>GrowCB</title></head>
<body style="background:#0a0f1e;color:#fff;font-family:sans-serif">
<h1>💎 GrowCB</h1>
<p>Login with Discord to play Crash & Mines, check balances, and chat.</p>
<p><a href="/login">Login with Discord</a></p>
<div id="me"></div>
<script>
async function j(u){let r=await fetch(u,{credentials:'include'});return r.ok?r.json():{}}
(async()=>{
 try{
  const me=await j('/api/me');
  document.getElementById('me').textContent='Logged in as '+me.username;
 }catch{}
})();
</script>
</body></html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(HTML_TEMPLATE)

# ---------- Lifespan ----------
@app.on_event("startup")
async def startup_event():
    init_db()
    if DISCORD_BOT_TOKEN:
        asyncio.create_task(bot.start(DISCORD_BOT_TOKEN))

@app.on_event("shutdown")
async def shutdown_event():
    if not bot.is_closed():
        await bot.close()

# ---------- Runner ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
