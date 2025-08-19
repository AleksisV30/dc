# app/main.py

import os, json, asyncio, re, random, string, datetime, base64
from urllib.parse import urlencode
from typing import Optional, Dict, List
from decimal import Decimal, ROUND_DOWN, getcontext
from contextlib import asynccontextmanager

import httpx
import psycopg
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from itsdangerous import URLSafeSerializer, BadSignature
from pydantic import BaseModel

# ---------- Discord bot ----------
import discord
from discord.ext import commands

# ---------- Game engines (local modules) ----------
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
GUILD_ID = int(os.getenv("GUILD_ID", "0"))
DISCORD_INVITE = os.getenv("DISCORD_INVITE", "")
SITE_BASE = os.getenv("SITE_BASE", "https://growcb.net")

GEM = "💎"
MIN_BET = Decimal("1.00")
MAX_BET = Decimal("1000000.00")
BETTING_SECONDS = 10

TWO = Decimal("0.01")
def D(x) -> Decimal:
    if isinstance(x, Decimal): return x
    return Decimal(str(x))
def q2(x: Decimal) -> Decimal:
    return D(x).quantize(TWO, rounding=ROUND_DOWN)

UTC = datetime.timezone.utc
def now_utc() -> datetime.datetime: return datetime.datetime.now(UTC)
def iso(dt: Optional[datetime.datetime]) -> Optional[str]:
    if dt is None: return None
    return dt.astimezone(UTC).isoformat()

# ---------- App / Lifespan ----------
def _get_static_dir():
    base = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(base, "static")
    try: os.makedirs(static_dir, exist_ok=True)
    except Exception: pass
    return static_dir

# Discord bot (disable default help, allow message content)
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=PREFIX, intents=intents, help_command=None)

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    apply_migrations()
    bot_task = None
    if DISCORD_BOT_TOKEN:
        bot_task = asyncio.create_task(bot.start(DISCORD_BOT_TOKEN))
    try:
        yield
    finally:
        if bot_task:
            try:
                await bot.close()
            except Exception:
                pass
            bot_task.cancel()

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=_get_static_dir()), name="static")

# ---------- Images from local folder or /static ----------
_TRANSPARENT_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
)

@app.get("/img/{filename}")
async def serve_img(filename: str):
    base = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(base, filename), os.path.join(base, "static", filename)):
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

# ---------- DB helpers ----------
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
    # balances
    cur.execute("""
        CREATE TABLE IF NOT EXISTS balances (
            user_id TEXT PRIMARY KEY,
            balance NUMERIC(18,2) NOT NULL DEFAULT 0
        )
    """)
    # balance log
    cur.execute("""
        CREATE TABLE IF NOT EXISTS balance_log (
            id BIGSERIAL PRIMARY KEY,
            actor_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            amount NUMERIC(18,2) NOT NULL,
            reason TEXT,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # profiles
    cur.execute("""
        CREATE TABLE IF NOT EXISTS profiles (
            user_id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            name_lower TEXT NOT NULL UNIQUE,
            xp INTEGER NOT NULL DEFAULT 0,
            role TEXT NOT NULL DEFAULT 'member',
            is_anon BOOLEAN NOT NULL DEFAULT FALSE,
            referred_by TEXT,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # oauth
    cur.execute("""
        CREATE TABLE IF NOT EXISTS oauth_tokens (
            user_id TEXT PRIMARY KEY,
            access_token TEXT NOT NULL,
            refresh_token TEXT,
            expires_at TIMESTAMPTZ
        )
    """)
    # referral
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ref_names (
            user_id TEXT PRIMARY KEY,
            name_lower TEXT UNIQUE NOT NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ref_visits (
            id BIGSERIAL PRIMARY KEY,
            referrer_id TEXT NOT NULL,
            joined_user_id TEXT,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # promos
    cur.execute("""
        CREATE TABLE IF NOT EXISTS promo_codes (
            code TEXT PRIMARY KEY,
            amount NUMERIC(18,2) NOT NULL,
            max_uses INTEGER NOT NULL DEFAULT 1,
            uses INTEGER NOT NULL DEFAULT 0,
            expires_at TIMESTAMPTZ,
            created_by TEXT,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS promo_redemptions (
            user_id TEXT NOT NULL,
            code TEXT NOT NULL,
            redeemed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(user_id, code)
        )
    """)
    # crash
    cur.execute("""
        CREATE TABLE IF NOT EXISTS crash_rounds (
            id BIGSERIAL PRIMARY KEY,
            status TEXT NOT NULL,
            bust NUMERIC(10,2),
            betting_opens_at TIMESTAMPTZ NOT NULL,
            betting_ends_at TIMESTAMPTZ NOT NULL,
            started_at TIMESTAMPTZ,
            expected_end_at TIMESTAMPTZ,
            ended_at TIMESTAMPTZ
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS crash_bets (
            round_id BIGINT NOT NULL,
            user_id TEXT NOT NULL,
            bet NUMERIC(18,2) NOT NULL,
            cashout NUMERIC(8,2) NOT NULL,
            cashed_out NUMERIC(8,2),
            cashed_out_at TIMESTAMPTZ,
            win NUMERIC(18,2) NOT NULL DEFAULT 0,
            resolved BOOLEAN NOT NULL DEFAULT FALSE,
            PRIMARY KEY(round_id, user_id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS crash_games (
            id BIGSERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            bet NUMERIC(18,2) NOT NULL,
            cashout NUMERIC(8,2) NOT NULL,
            bust NUMERIC(10,2) NOT NULL,
            win NUMERIC(18,2) NOT NULL,
            xp_gain INTEGER NOT NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # chat
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id BIGSERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            username TEXT NOT NULL,
            text TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            private_to TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_timeouts (
            user_id TEXT PRIMARY KEY,
            until TIMESTAMPTZ NOT NULL,
            reason TEXT,
            created_by TEXT,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # mines
    cur.execute("""
        CREATE TABLE IF NOT EXISTS mines_games (
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
        )
    """)
    # transfers (deposit/withdraw)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS transfer_requests (
            id BIGSERIAL PRIMARY KEY,
            kind TEXT NOT NULL,           -- 'deposit' | 'withdraw'
            user_id TEXT,
            discord_identity TEXT,
            grow_id TEXT,
            world TEXT NOT NULL,
            amount NUMERIC(18,2),
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
    """)

# ---- balances / profiles ----
@with_conn
def get_balance(cur, user_id: str) -> Decimal:
    cur.execute("SELECT balance FROM balances WHERE user_id=%s", (user_id,))
    row = cur.fetchone(); return q2(row[0]) if row else Decimal("0.00")

@with_conn
def adjust_balance(cur, actor_id: str, target_id: str, amount, reason: Optional[str]):
    amount = q2(D(amount))
    cur.execute("INSERT INTO balances (user_id, balance) VALUES (%s, 0) ON CONFLICT (user_id) DO NOTHING", (target_id,))
    cur.execute("UPDATE balances SET balance = balance + %s WHERE user_id = %s", (amount, target_id))
    cur.execute("INSERT INTO balance_log(actor_id, target_id, amount, reason) VALUES (%s, %s, %s, %s)",
                (actor_id, target_id, amount, reason))
    cur.execute("SELECT balance FROM balances WHERE user_id = %s", (target_id,))
    return q2(cur.fetchone()[0])

@with_conn
def tip_transfer(cur, from_id: str, to_id: str, amount: Decimal):
    amount = q2(D(amount))
    if amount <= 0: raise ValueError("Amount must be > 0")
    if from_id == to_id: raise ValueError("Cannot tip yourself")
    cur.execute("INSERT INTO balances(user_id,balance) VALUES (%s,0) ON CONFLICT(user_id) DO NOTHING", (from_id,))
    cur.execute("INSERT INTO balances(user_id,balance) VALUES (%s,0) ON CONFLICT(user_id) DO NOTHING", (to_id,))
    cur.execute("SELECT balance FROM balances WHERE user_id=%s FOR UPDATE", (from_id,))
    sbal = D(cur.fetchone()[0])
    if sbal < amount: raise ValueError("Insufficient balance")
    cur.execute("UPDATE balances SET balance=balance-%s WHERE user_id=%s", (amount, from_id))
    cur.execute("UPDATE balances SET balance=balance+%s WHERE user_id=%s", (amount, to_id))
    cur.execute("INSERT INTO balance_log(actor_id,target_id,amount,reason) VALUES (%s,%s,%s,%s)", (from_id, to_id, -amount, "tip"))
    cur.execute("INSERT INTO balance_log(actor_id,target_id,amount,reason) VALUES (%s,%s,%s,%s)", (from_id, to_id, amount, "tip"))
    return True

NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{3,20}$")

@with_conn
def ensure_profile_row(cur, user_id: str):
    role = 'owner' if str(user_id) == str(OWNER_ID) else 'member'
    default_name = f"user_{user_id[-4:]}"
    cur.execute("""
        INSERT INTO profiles(user_id, display_name, name_lower, role, is_anon)
        VALUES (%s,%s,%s,%s,FALSE)
        ON CONFLICT (user_id) DO NOTHING
    """, (user_id, default_name, default_name, role))

@with_conn
def get_profile_name(cur, user_id: str):
    cur.execute("SELECT display_name FROM profiles WHERE user_id = %s", (user_id,))
    r = cur.fetchone(); return r[0] if r else None

@with_conn
def set_profile_name(cur, user_id: str, name: str):
    if not NAME_RE.match(name): raise ValueError("Name must be 3-20 chars [a-zA-Z0-9_-]")
    lower = name.lower()
    cur.execute("SELECT user_id FROM profiles WHERE name_lower=%s AND user_id<>%s", (lower, user_id))
    if cur.fetchone(): raise ValueError("Name is already taken")
    cur.execute("""
        INSERT INTO profiles(user_id, display_name, name_lower)
        VALUES (%s,%s,%s)
        ON CONFLICT (user_id) DO UPDATE SET display_name=EXCLUDED.display_name, name_lower=EXCLUDED.name_lower
    """, (user_id, name, lower))
    return {"ok": True, "name": name}

@with_conn
def set_profile_is_anon(cur, user_id: str, is_anon: bool):
    ensure_profile_row(user_id)
    cur.execute("UPDATE profiles SET is_anon=%s WHERE user_id=%s", (bool(is_anon), user_id))
    return {"ok": True, "is_anon": bool(is_anon)}

@with_conn
def profile_info(cur, user_id: str):
    ensure_profile_row(user_id)
    cur.execute("SELECT xp, role, is_anon FROM profiles WHERE user_id=%s", (user_id,))
    xp, role, is_anon = cur.fetchone()
    level = 1 + int(xp) // 100
    base = (level - 1) * 100; need = level * 100 - base
    progress = int(xp) - base; pct = 0 if need==0 else int(progress*100/need)
    bal = get_balance(user_id)
    return {
        "id": str(user_id),
        "xp": int(xp), "level": level, "progress": progress, "next_needed": need,
        "progress_pct": pct, "balance": float(bal), "role": role, "is_anon": bool(is_anon)
    }

@with_conn
def public_profile(cur, user_id: str) -> Optional[dict]:
    ensure_profile_row(user_id)
    cur.execute("SELECT display_name, xp, role, created_at, is_anon FROM profiles WHERE user_id=%s", (user_id,))
    r = cur.fetchone()
    if not r: return None
    display_name, xp, role, created_at, is_anon = r
    level = 1 + int(xp)//100
    cur.execute("SELECT COUNT(*) FROM crash_games WHERE user_id=%s", (user_id,)); crash_count = int(cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM mines_games WHERE user_id=%s AND status<>'active'", (user_id,)); mines_count = int(cur.fetchone()[0])
    bal = get_balance(user_id)
    return {
        "id": str(user_id), "name": ("Anonymous" if is_anon else display_name), "role": role, "is_anon": bool(is_anon),
        "xp": int(xp), "level": level, "balance": float(bal),
        "crash_games": crash_count, "mines_games": mines_count,
        "created_at": str(created_at)
    }

@with_conn
def set_role(cur, target_id: str, role: str):
    if role not in ("member","admin","owner"): raise ValueError("Invalid role")
    cur.execute("UPDATE profiles SET role=%s WHERE user_id=%s", (role, target_id))
    return {"ok": True, "role": role}

# ---- promos ----
class PromoError(Exception): ...
class PromoAlreadyRedeemed(PromoError): ...
class PromoInvalid(PromoError): ...
class PromoExpired(PromoError): ...
class PromoExhausted(PromoError): ...

@with_conn
def redeem_promo(cur, user_id: str, code: str) -> Decimal:
    code = code.strip().upper()
    cur.execute("SELECT code, amount, max_uses, uses, expires_at FROM promo_codes WHERE code=%s", (code,))
    row = cur.fetchone()
    if not row: raise PromoInvalid("Invalid code")
    _, amount, max_uses, uses, expires_at = row
    if expires_at is not None:
        cur.execute("SELECT NOW()>%s", (expires_at,))
        if cur.fetchone()[0]: raise PromoExpired("Code expired")
    if uses >= max_uses: raise PromoExhausted("Code maxed out")
    cur.execute("SELECT 1 FROM promo_redemptions WHERE user_id=%s AND code=%s", (user_id, code))
    if cur.fetchone(): raise PromoAlreadyRedeemed("You already redeemed this code")
    cur.execute("INSERT INTO balances(user_id,balance) VALUES (%s,0) ON CONFLICT(user_id) DO NOTHING", (user_id,))
    cur.execute("UPDATE balances SET balance=balance+%s WHERE user_id=%s", (amount, user_id))
    cur.execute("UPDATE promo_codes SET uses=uses+1 WHERE code=%s", (code,))
    cur.execute("INSERT INTO promo_redemptions(user_id,code) VALUES (%s,%s)", (user_id, code))
    cur.execute("INSERT INTO balance_log(actor_id,target_id,amount,reason) VALUES (%s,%s,%s,%s)",
                ("promo", user_id, amount, f"promo:{code}"))
    cur.execute("SELECT balance FROM balances WHERE user_id=%s", (user_id,))
    return q2(cur.fetchone()[0])

def _rand_code(n=8): return ''.join(random.choices(string.ascii_uppercase+string.digits, k=n))

@with_conn
def create_promo(cur, actor_id: str, code: Optional[str], amount, max_uses: int = 1, expires_at: Optional[str] = None):
    amt = q2(D(amount))
    code = (code.strip().upper() if code else _rand_code())
    cur.execute("""
        INSERT INTO promo_codes(code,amount,max_uses,expires_at,created_by)
        VALUES (%s,%s,%s,%s,%s)
        ON CONFLICT (code) DO UPDATE SET amount=EXCLUDED.amount, max_uses=EXCLUDED.max_uses, expires_at=EXCLUDED.expires_at
    """, (code, amt, max_uses, expires_at, actor_id))
    return {"ok": True, "code": code}

# ---------- Leaderboard ----------
def _start_of_utc_day(dt: datetime.datetime) -> datetime.datetime:
    return dt.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
def _start_of_utc_month(dt: datetime.datetime) -> datetime.datetime:
    return dt.astimezone(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)

@with_conn
def get_leaderboard_rows_db(cur, period: str, limit: int = 50):
    period = (period or "daily").lower()
    now = now_utc()
    params = []
    where_crash = ""
    where_mines = "WHERE status<>'active'"

    if period == "daily":
        start = _start_of_utc_day(now)
        where_crash = "WHERE created_at >= %s"
        where_mines += " AND started_at >= %s"
        params.extend([start, start])
    elif period == "monthly":
        start = _start_of_utc_month(now)
        where_crash = "WHERE created_at >= %s"
        where_mines += " AND started_at >= %s"
        params.extend([start, start])
    else:
        where_crash = ""
        where_mines = "WHERE status<>'active'"

    sql = f"""
        WITH wagers AS (
            SELECT user_id, COALESCE(SUM(bet),0)::numeric(18,2) AS total
            FROM crash_games
            {where_crash}
            GROUP BY user_id
            UNION ALL
            SELECT user_id, COALESCE(SUM(bet),0)::numeric(18,2) AS total
            FROM mines_games
            {where_mines}
            GROUP BY user_id
        ),
        by_user AS (
            SELECT user_id, SUM(total)::numeric(18,2) AS total_wagered
            FROM wagers GROUP BY user_id
        )
        SELECT 
            bu.user_id,
            COALESCE(p.display_name, 'user_' || RIGHT(bu.user_id, 4)) AS display_name,
            COALESCE(p.is_anon, FALSE) AS is_anon,
            bu.total_wagered::numeric(18,2) AS total_wagered
        FROM by_user bu
        LEFT JOIN profiles p ON p.user_id = bu.user_id
        ORDER BY bu.total_wagered DESC, bu.user_id
        LIMIT %s
    """
    params.append(limit)
    cur.execute(sql, tuple(params))
    rows = cur.fetchall()
    out = []
    for uid, name, is_anon, total in rows:
        out.append({
            "user_id": str(uid),
            "display_name": name,
            "is_anon": bool(is_anon),
            "total_wagered": float(q2(D(total)))
        })
    return out

# ---------- Migrations ----------
@with_conn
def apply_migrations(cur):
    cur.execute("ALTER TABLE profiles ADD COLUMN IF NOT EXISTS is_anon BOOLEAN NOT NULL DEFAULT FALSE")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_crash_games_created_at ON crash_games (created_at)")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_mines_games_started_at ON mines_games (started_at)")
    cur.execute("ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS private_to TEXT")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_transfer_kind ON transfer_requests(kind)")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_ref_names ON ref_names(name_lower)")

# ---------- OAuth token store & Discord join ----------
@with_conn
def save_tokens(cur, user_id: str, access_token: str, refresh_token: Optional[str], expires_in: Optional[int]):
    expires_at = now_utc() + datetime.timedelta(seconds=int(expires_in or 0)) if expires_in else None
    cur.execute("""
        INSERT INTO oauth_tokens(user_id, access_token, refresh_token, expires_at)
        VALUES (%s,%s,%s,%s)
        ON CONFLICT (user_id) DO UPDATE SET access_token=EXCLUDED.access_token, refresh_token=EXCLUDED.refresh_token, expires_at=EXCLUDED.expires_at
    """, (user_id, access_token, refresh_token, expires_at))

@with_conn
def get_tokens(cur, user_id: str):
    cur.execute("SELECT access_token, refresh_token, expires_at FROM oauth_tokens WHERE user_id=%s", (user_id,))
    r = cur.fetchone()
    if not r: return None
    return {"access_token": r[0], "refresh_token": r[1], "expires_at": r[2]}

async def discord_refresh_token(user_id: str):
    rec = get_tokens(user_id)
    if not rec or not rec.get("refresh_token"): return None
    if not CLIENT_ID or not CLIENT_SECRET: return None
    async with httpx.AsyncClient(timeout=15) as cx:
        data = {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "grant_type": "refresh_token",
            "refresh_token": rec["refresh_token"]
        }
        r = await cx.post(f"{DISCORD_API}/oauth2/token", data=data, headers={"Content-Type":"application/x-www-form-urlencoded"})
        if r.status_code != 200: return None
        js = r.json()
        save_tokens(user_id, js.get("access_token",""), js.get("refresh_token"), js.get("expires_in"))
        return js.get("access_token")

async def discord_get_valid_access_token(user_id: str):
    rec = get_tokens(user_id)
    if not rec: return None
    exp = rec.get("expires_at")
    if exp and isinstance(exp, datetime.datetime) and exp.tzinfo is None:
        exp = exp.replace(tzinfo=UTC)
    if (not exp) or (exp - now_utc() < datetime.timedelta(seconds=30)):
        tok = await discord_refresh_token(user_id)
        if tok: return tok
    return rec.get("access_token")

async def guild_add_member(user_id: str, nickname: Optional[str] = None):
    if not (DISCORD_BOT_TOKEN and GUILD_ID):
        raise HTTPException(500, "Discord bot or guild not configured")
    access = await discord_get_valid_access_token(user_id)
    if not access:
        raise HTTPException(400, "Missing OAuth token. Re-login needed.")
    payload = {"access_token": access}
    if nickname: payload["nick"] = nickname
    async with httpx.AsyncClient(timeout=15) as cx:
        url = f"{DISCORD_API}/guilds/{GUILD_ID}/members/{user_id}"
        r = await cx.put(url, json=payload, headers={"Authorization": f"Bot {DISCORD_BOT_TOKEN}"})
        if r.status_code in (201, 204): return {"ok": True}
        if r.status_code == 409: return {"ok": True}
        raise HTTPException(r.status_code, f"Discord join failed: {r.text}")

# ---------- OAuth / Auth ----------
@app.get("/login")
async def login():
    if not (CLIENT_ID and OAUTH_REDIRECT):
        return HTMLResponse("OAuth not configured")
    params = {
        "client_id": CLIENT_ID,
        "redirect_uri": OAUTH_REDIRECT,
        "response_type": "code",
        "scope": "identify guilds.join",
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
        r = await cx.post(f"{DISCORD_API}/oauth2/token", data=data, headers={"Content-Type":"application/x-www-form-urlencoded"})
        if r.status_code != 200:
            return HTMLResponse(f"OAuth failed: {r.text}", status_code=400)
        tok = r.json()
        access = tok.get("access_token")
        async with httpx.AsyncClient(timeout=15) as cx2:
            u = await cx2.get(f"{DISCORD_API}/users/@me", headers={"Authorization": f"Bearer {access}"})
            if u.status_code != 200:
                return HTMLResponse(f"User fetch failed: {u.text}", status_code=400)
            me = u.json()

    user_id = str(me["id"])
    username = f'{me.get("username","user")}#{me.get("discriminator","0")}'.replace("#0","")
    avatar_hash = me.get("avatar")
    avatar_url = f"https://cdn.discordapp.com/avatars/{user_id}/{avatar_hash}.png?size=64" if avatar_hash else "https://cdn.discordapp.com/embed/avatars/0.png"

    ensure_profile_row(user_id)
    save_tokens(user_id, tok.get("access_token",""), tok.get("refresh_token"), tok.get("expires_in"))

    resp = RedirectResponse("/")
    _set_session(resp, {"id": user_id, "username": username, "avatar_url": avatar_url})
    return resp

@app.get("/logout")
async def logout():
    resp = RedirectResponse("/")
    _clear_session(resp)
    return resp

# ---------- Me / Profile / Balance ----------
@app.get("/api/me")
async def api_me(request: Request):
    s = _require_session(request)
    in_guild = False
    if DISCORD_BOT_TOKEN and GUILD_ID:
        try:
            async with httpx.AsyncClient(timeout=8) as cx:
                r = await cx.get(f"{DISCORD_API}/guilds/{GUILD_ID}/members/{s['id']}",
                                 headers={"Authorization": f"Bot {DISCORD_BOT_TOKEN}"})
                in_guild = (r.status_code == 200)
        except:
            in_guild = False
    return {"id": s["id"], "username": s["username"], "avatar_url": s.get("avatar_url"), "in_guild": in_guild}

@app.get("/api/balance")
async def api_balance(request: Request):
    s = _require_session(request)
    return {"balance": float(get_balance(s["id"]))}

@app.get("/api/profile")
async def api_profile(request: Request):
    s = _require_session(request)
    return profile_info(s["id"])

@app.get("/api/profile/public")
async def api_profile_public(user_id: str = Query(...)):
    p = public_profile(user_id)
    if not p: raise HTTPException(404, "Not found")
    return p

# ---------- Settings (Anonymous mode) ----------
class AnonIn(BaseModel):
    is_anon: bool

@app.get("/api/settings/get")
async def api_settings_get(request: Request):
    s = _require_session(request)
    info = profile_info(s["id"])
    return {"is_anon": bool(info["is_anon"])}

@app.post("/api/settings/set_anon")
async def api_settings_set_anon(request: Request, body: AnonIn):
    s = _require_session(request)
    return set_profile_is_anon(s["id"], bool(body.is_anon))

# ---------- Leaderboard API ----------
@app.get("/api/leaderboard")
async def api_leaderboard(period: str = Query("daily"), limit: int = Query(50, ge=1, le=200)):
    rows = get_leaderboard_rows_db(period, limit)
    return {"rows": rows}

# ---------- Referral ----------
@with_conn
def get_ref_state(cur, user_id: str):
    cur.execute("SELECT name_lower FROM ref_names WHERE user_id=%s", (user_id,))
    r = cur.fetchone()
    name = r[0] if r else None
    cur.execute("SELECT COUNT(*) FROM ref_visits WHERE referrer_id=%s AND joined_user_id IS NOT NULL", (user_id,))
    joined = int(cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM ref_visits WHERE referrer_id=%s", (user_id,))
    clicks = int(cur.fetchone()[0])
    return {"name": name, "joined": joined, "clicks": clicks}

@with_conn
def set_ref_name(cur, user_id: str, name: str):
    if not NAME_RE.match(name): raise ValueError("3–20 chars: letters, numbers, _ or -")
    lower = name.lower()
    cur.execute("SELECT user_id FROM ref_names WHERE name_lower=%s AND user_id<>%s", (lower, user_id))
    if cur.fetchone(): raise ValueError("Name is already taken")
    cur.execute("""
        INSERT INTO ref_names(user_id, name_lower)
        VALUES(%s,%s)
        ON CONFLICT (user_id) DO UPDATE SET name_lower=EXCLUDED.name_lower
    """, (user_id, lower))
    return {"ok": True, "name": lower}

def _record_ref_visit(refname: str):
    with psycopg.connect(DATABASE_URL) as con, con.cursor() as cur:
        cur.execute("SELECT user_id FROM ref_names WHERE name_lower=%s", (refname,))
        r = cur.fetchone()
        if r:
            referrer = str(r[0])
            cur.execute("INSERT INTO ref_visits(referrer_id) VALUES (%s)", (referrer,))
            con.commit()

@app.get("/referral/{refname}")
async def referral_landing_new(refname: str):
    rn = (refname or "").lower()
    _record_ref_visit(rn)
    html = f"""
    <script>
      document.cookie = "refname={rn}; path=/; max-age=1209600; samesite=lax";
      location.href = "/";
    </script>
    """
    return HTMLResponse(html)

@app.get("/r/{refname}")
async def referral_landing_old(refname: str):
    return await referral_landing_new(refname)

@app.get("/api/referral/state")
async def api_ref_state(request: Request):
    s = _require_session(request)
    return get_ref_state(s["id"])

class RefIn(BaseModel):
    name: str

@app.post("/api/referral/set")
async def api_ref_set(request: Request, body: RefIn):
    s = _require_session(request)
    return set_ref_name(s["id"], body.name)

@app.get("/api/referral/attach")
async def api_ref_attach(request: Request, refname: str = ""):
    s = _require_session(request)
    rn = (refname or "").lower()
    if not rn: return {"ok": True}
    with psycopg.connect(DATABASE_URL) as con, con.cursor() as cur:
        cur.execute("SELECT user_id FROM ref_names WHERE name_lower=%s", (rn,))
        r = cur.fetchone()
        if not r: return {"ok": True}
        referrer = str(r[0])
        if referrer == s["id"]: return {"ok": True}
        cur.execute("SELECT referred_by FROM profiles WHERE user_id=%s", (s["id"],))
        already = cur.fetchone()
        if already and already[0]: return {"ok": True}
        cur.execute("UPDATE profiles SET referred_by=%s WHERE user_id=%s", (referrer, s["id"]))
        cur.execute("INSERT INTO ref_visits(referrer_id, joined_user_id) VALUES (%s,%s)", (referrer, s["id"]))
        con.commit()
    return {"ok": True}

# ---------- Promo ----------
@app.get("/api/promo/my")
async def api_promo_my(request: Request):
    s = _require_session(request)
    with psycopg.connect(DATABASE_URL) as con, con.cursor() as cur:
        cur.execute("SELECT code, redeemed_at FROM promo_redemptions WHERE user_id=%s ORDER BY redeemed_at DESC LIMIT 50", (s["id"],))
        rows = [{"code": r[0], "redeemed_at": str(r[1])} for r in cur.fetchall()]
    return {"rows": rows}

class PromoIn(BaseModel):
    code: str

@app.post("/api/promo/redeem")
async def api_promo_redeem(request: Request, body: PromoIn):
    s = _require_session(request)
    try:
        bal = redeem_promo(s["id"], body.code)
        return {"ok": True, "new_balance": float(bal)}
    except (PromoInvalid, PromoExpired, PromoExhausted, PromoAlreadyRedeemed) as e:
        raise HTTPException(400, str(e))

# ---------- Crash endpoints ----------
class CrashBetIn(BaseModel):
    bet: str
    cashout: Optional[float] = None

@app.get("/api/crash/state")
async def api_crash_state(request: Request):
    try:
        s = _require_session(request)
        uid = s["id"]
    except:
        s = None
        uid = None

    rid, info = ensure_betting_round()
    now = now_utc()

    if info["status"] == "betting" and now >= info["betting_ends_at"]:
        begin_running(rid)
        info = load_round()
    if info and info["status"] == "running" and info["expected_end_at"] and now >= info["expected_end_at"]:
        finish_round(rid)
        create_next_betting()
        info = load_round()

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
    bet = q2(D(body.bet or "0"))
    cashout = float(body.cashout or 2.0)
    return place_bet(s["id"], bet, max(1.01, cashout))

@app.post("/api/crash/cashout")
async def api_crash_cashout(request: Request):
    s = _require_session(request)
    cur = load_round()
    if not cur or cur["status"] != "running":
        raise HTTPException(400, "No running round")
    return cashout_now(s["id"])
# ---------- Mines endpoints ----------
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

# ---------- Chat endpoints ----------
class ChatIn(BaseModel):
    text: str

@with_conn
def get_role(cur, user_id: str) -> str:
    cur.execute("SELECT role FROM profiles WHERE user_id=%s", (user_id,))
    r = cur.fetchone()
    return r[0] if r else "member"

@with_conn
def chat_timeout_set(cur, actor_id: str, user_id: str, seconds: int, reason: Optional[str]):
    until = now_utc() + datetime.timedelta(seconds=max(1, seconds))
    cur.execute("""INSERT INTO chat_timeouts(user_id, until, reason, created_by)
                   VALUES (%s,%s,%s,%s)
                   ON CONFLICT (user_id) DO UPDATE SET until=EXCLUDED.until, reason=EXCLUDED.reason, created_by=EXCLUDED.created_by""",
                (user_id, until, reason, actor_id))
    return {"ok": True, "until": str(until)}

@with_conn
def chat_timeout_get(cur, user_id: str):
    cur.execute("SELECT until, reason FROM chat_timeouts WHERE user_id=%s", (user_id,))
    r = cur.fetchone()
    if not r: return None
    until, reason = r
    if until <= now_utc():
        cur.execute("DELETE FROM chat_timeouts WHERE user_id=%s", (user_id,))
        return None
    delta = int((until - now_utc()).total_seconds())
    return {"until": str(until), "seconds_left": max(0, delta), "reason": reason or ""}

@with_conn
def chat_insert(cur, user_id: str, username: str, text: str, private_to: Optional[str] = None):
    text = (text or "").strip()
    if not text: raise ValueError("Message is empty")
    if len(text) > 300: raise ValueError("Message is too long (max 300)")
    ensure_profile_row(user_id)
    if private_to is None:
        cur.execute("SELECT until FROM chat_timeouts WHERE user_id=%s", (user_id,))
        r = cur.fetchone()
        if r and r[0] > now_utc(): raise PermissionError("You are timed out")
    cur.execute("INSERT INTO chat_messages(user_id, username, text, private_to) VALUES (%s,%s,%s,%s) RETURNING id, created_at",
                (user_id, username, text, private_to))
    row = cur.fetchone()
    return {"id": int(row[0]), "created_at": str(row[1])}

@with_conn
def chat_fetch(cur, since_id: int, limit: int, for_user_id: Optional[str]):
    if since_id <= 0:
        cur.execute("""SELECT id, user_id, username, text, created_at, private_to
                       FROM chat_messages
                       WHERE private_to IS NULL
                       ORDER BY id DESC LIMIT %s""", (limit,))
        rows_pub = list(reversed(cur.fetchall()))
        rows_priv = []
        if for_user_id:
            cur.execute("""SELECT id, user_id, username, text, created_at, private_to
                           FROM chat_messages
                           WHERE private_to=%s
                           ORDER BY id DESC LIMIT %s""", (for_user_id, limit))
            rows_priv = list(reversed(cur.fetchall()))
        rows = sorted(rows_pub + rows_priv, key=lambda r: r[0])
    else:
        if for_user_id:
            cur.execute("""SELECT id, user_id, username, text, created_at, private_to
                           FROM chat_messages
                           WHERE id>%s AND (private_to IS NULL OR private_to=%s)
                           ORDER BY id ASC LIMIT %s""", (since_id, for_user_id, limit))
        else:
            cur.execute("""SELECT id, user_id, username, text, created_at, private_to
                           FROM chat_messages
                           WHERE id>%s AND private_to IS NULL
                           ORDER BY id ASC LIMIT %s""", (since_id, limit))
        rows = cur.fetchall()

    uids = list({str(r[1]) for r in rows})
    levels: Dict[str, int] = {}
    roles: Dict[str, str] = {}
    if uids:
        cur.execute("SELECT user_id, xp, role FROM profiles WHERE user_id = ANY(%s)", (uids,))
        for uid, xp, role in cur.fetchall():
            levels[str(uid)] = 1 + int(xp) // 100
            roles[str(uid)] = role or "member"
    out = []
    for mid, uid, uname, txt, ts, priv in rows:
        uid = str(uid)
        out.append({"id": int(mid), "user_id": uid, "username": uname,
                    "level": int(levels.get(uid,1)), "role": roles.get(uid,"member"),
                    "text": txt, "created_at": str(ts), "private_to": priv})
    return out

@with_conn
def chat_delete(cur, message_id: int):
    cur.execute("DELETE FROM chat_messages WHERE id=%s", (message_id,))
    return {"ok": True}

@app.post("/api/chat/send")
async def api_chat_send(request: Request, body: ChatIn):
    s = _require_session(request)
    return chat_insert(s["id"], s["username"], body.text, None)

@app.get("/api/chat/fetch")
async def api_chat_fetch(request: Request, since: int = 0, limit: int = 30):
    uid = None
    try:
        sess = _require_session(request)
        uid = sess["id"]
    except:
        pass
    rows = chat_fetch(since, limit, uid)
    return {"rows": rows}

@app.post("/api/chat/delete")
async def api_chat_del(request: Request, id: int):
    s = _require_session(request)
    role = get_role(s["id"])
    if role not in ("admin","owner"): raise HTTPException(403, "No permission")
    return chat_delete(id)

# ---------- Admin ----------
class AdjustIn(BaseModel):
    identifier: str
    amount: str
    reason: Optional[str] = None

def _id_from_identifier(identifier: str) -> str:
    m = re.search(r"\d{5,}", identifier or "")
    if not m: raise HTTPException(400, "Provide a numeric Discord ID or mention")
    return m.group(0)

@app.post("/api/admin/adjust")
async def api_admin_adjust(request: Request, body: AdjustIn):
    s = _require_session(request)
    role = get_role(s["id"])
    if role not in ("admin","owner"): raise HTTPException(403, "No permission")
    target = _id_from_identifier(body.identifier)
    newbal = adjust_balance(s["id"], target, D(body.amount), body.reason)
    return {"new_balance": float(newbal)}

class RoleIn(BaseModel):
    identifier: str
    role: str

@app.post("/api/admin/role")
async def api_admin_role(request: Request, body: RoleIn):
    s = _require_session(request)
    role = get_role(s["id"])
    if role != "owner": raise HTTPException(403, "Only owner can set roles")
    target = _id_from_identifier(body.identifier)
    return set_role(target, body.role)

class TimeoutIn(BaseModel):
    identifier: str
    seconds: int
    reason: Optional[str] = None

@app.post("/api/admin/timeout_site")
async def api_admin_timeout_site(request: Request, body: TimeoutIn):
    s = _require_session(request)
    role = get_role(s["id"])
    if role not in ("admin","owner"): raise HTTPException(403, "No permission")
    target = _id_from_identifier(body.identifier)
    return chat_timeout_set(s["id"], target, int(body.seconds), body.reason or "")

@app.post("/api/admin/timeout_both")
async def api_admin_timeout_both(request: Request, body: TimeoutIn):
    return await api_admin_timeout_site(request, body)

# ---------- Deposit / Withdraw ----------
class TransferIn(BaseModel):
    kind: str             # 'deposit' | 'withdraw'
    discord_identity: Optional[str] = None
    grow_id: Optional[str] = None
    world: str
    amount: Optional[str] = None

@app.post("/api/transfer/create")
async def api_transfer_create(request: Request, body: TransferIn):
    s = None
    try:
        s = _require_session(request)
    except:
        s = None
    kind = (body.kind or "").lower()
    if kind not in ("deposit","withdraw"):
        raise HTTPException(400, "Invalid kind")
    if kind == "withdraw":
        if not body.amount: raise HTTPException(400, "Amount required for withdraw")
        try:
            _ = q2(D(body.amount))
        except: raise HTTPException(400, "Invalid amount")
    grow_id = (body.grow_id or "").strip() or None
    did = None
    if s:
        did = s["id"]
    with psycopg.connect(DATABASE_URL) as con, con.cursor() as cur:
        cur.execute("""INSERT INTO transfer_requests(kind,user_id,discord_identity,grow_id,world,amount)
                       VALUES (%s,%s,%s,%s,%s,%s) RETURNING id""",
                    (kind, did, (body.discord_identity or None), grow_id, body.world.strip(), (q2(D(body.amount)) if body.amount else None)))
        rid = int(cur.fetchone()[0])
        con.commit()
    return {"ok": True, "id": rid}

# ---------- Discord Join ----------
@app.post("/api/discord/join")
async def api_discord_join(request: Request):
    s = _require_session(request)
    nick = get_profile_name(s["id"]) or s["username"]
    return await guild_add_member(s["id"], nickname=nick)

# ---------- HTML (UI/UX) ----------
HTML_TEMPLATE = """
<!doctype html><html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"/>
<title>GROWCB</title>
<style>
:root{
  --bg:#0a0f1e;--bg2:#0c1428;--card:#111a31;--muted:#9eb3da;--text:#ecf2ff;
  --accent:#6aa6ff;--accent2:#22c1dc;--ok:#34d399;--warn:#f59e0b;--err:#ef4444;--border:#1f2b47;
  --chatW:360px;--input-bg:#0b1430;--input-br:#223457;--input-tx:#e6eeff;--input-ph:#9db4e4
}
*{box-sizing:border-box}html,body{height:100%}body{margin:0;color:var(--text);
  background:radial-gradient(1400px 600px at 20% -10%, #11204d 0%, transparent 60%),linear-gradient(180deg,#0a0f1e,#0a0f1e 60%, #0b1124);
  font-family:Inter,system-ui,Segoe UI,Roboto,Arial,Helvetica,sans-serif}
a{color:inherit;text-decoration:none}.container{max-width:1180px;margin:0 auto;padding:16px}
input,select,textarea{width:100%;appearance:none;background:var(--input-bg);color:var(--input-tx);border:1px solid var(--input-br);border-radius:12px;padding:10px 12px;outline:none}
.field{display:flex;flex-direction:column;gap:6px}.row{display:grid;gap:10px}
.row.cols-2{grid-template-columns:1fr 1fr}.row.cols-3{grid-template-columns:1fr 1fr 1fr}
.card{background:linear-gradient(180deg,#0f1a33,#0b1326);border:1px solid var(--border);border-radius:16px;padding:16px}
.header{position:sticky;top:0;z-index:40;backdrop-filter:blur(8px);background:rgba(10,15,30,.78);border-bottom:1px solid var(--border)}
.header-inner{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:10px 12px}
.left{display:flex;align-items:center;gap:14px;flex:1;min-width:0}.brand{display:flex;align-items:center;gap:12px;font-weight:900;white-space:nowrap}
.brand .logo{width:48px;height:48px;border-radius:12px;object-fit:contain;border:1px solid var(--border);background:linear-gradient(135deg,var(--accent),var(--accent2))}
.tabs{display:flex;gap:6px;align-items:center;padding:4px;border-radius:14px;background:linear-gradient(180deg,#0f1a33,#0b1326);border:1px solid var(--border)}
.tab{padding:10px 14px;border-radius:10px;cursor:pointer;font-weight:700;white-space:nowrap;color:#d8e6ff;opacity:.85;transition:all .15s ease;display:flex;align-items:center;gap:8px}
.tab:hover{opacity:1;transform:translateY(-1px)}.tab.active{background:linear-gradient(135deg,#3b82f6,#22c1dc);color:#051326;box-shadow:0 6px 16px rgba(59,130,246,.25);opacity:1}
.right{display:flex;gap:10px;align-items:center;margin-left:12px}
.btn{display:inline-flex;align-items:center;gap:8px;padding:10px 14px;border-radius:14px;border:1px solid var(--border);
     background:linear-gradient(180deg,#0e1833,#0b1326);cursor:pointer;font-weight:700;letter-spacing:.2px;transform:translateZ(0)}
.btn:active{transform:translateY(1px)}.btn.primary{background:linear-gradient(135deg,#3b82f6,#22c1dc);border-color:transparent;color:#041018}
.btn.soft{background:#132446;border-color:#203458}.btn.ghost{background:#162a52;border:1px solid var(--border);color:#eaf2ff}
.chip{background:#0c1631;border:1px solid var(--border);color:#dce7ff;padding:6px 10px;border-radius:999px;font-size:12px;white-space:nowrap}
.avatar{width:36px;height:36px;border-radius:50%;object-fit:cover;border:1px solid var(--border);cursor:pointer}
.avatar-wrap{position:relative}.menu{position:absolute;right:0;top:42px;background:#0c1631;border:1px solid var(--border);border-radius:12px;padding:6px;display:none;min-width:180px;z-index:60}
.menu.open{display:block}.menu .item{padding:8px 10px;border-radius:8px;cursor:pointer;font-size:14px}.menu .item:hover{background:#11234a}
.big{font-size:22px;font-weight:900}.label{font-size:12px;color:var(--muted);letter-spacing:.2px;text-transform:uppercase}.muted{color:var(--muted)}
.games-grid{display:grid;gap:14px;grid-template-columns:1fr}@media(min-width:700px){.games-grid{grid-template-columns:1fr 1fr}}@media(min-width:1020px){.games-grid{grid-template-columns:1fr 1fr 1fr}}
/* Game cards: image only, sized to native aspect (no overlays) */
.game-card{border:1px solid var(--border);border-radius:16px;overflow:hidden;cursor:pointer;background:#0b1326;padding:0}
.game-card .banner{display:block;width:100%;height:auto;object-fit:contain}

/* Crash area */
.cr-graph-wrap{position:relative;height:240px;background:#0e1833;border:1px solid var(--border);border-radius:16px;overflow:hidden}
canvas{display:block;width:100%;height:100%}
table{width:100%;border-collapse:collapse}th,td{padding:10px;border-bottom:1px solid rgba(255,255,255,.06);text-align:left}
tr.me-row{background:linear-gradient(90deg, rgba(34,197,94,.12), transparent 60%)}tr.anon td.name{color:#9db4e4;font-style:italic}
.grid-2{display:grid;grid-template-columns:1fr;gap:16px}@media(min-width:900px){.grid-2{grid-template-columns:1.1fr .9fr}}
.hero{display:flex;justify-content:space-between;align-items:center;gap:14px;flex-wrap:wrap}
.kpi{display:flex;gap:8px;flex-wrap:wrap}
.kpi .pill{background:#0c1631;border:1px solid var(--border);border-radius:999px;padding:6px 10px;font-size:12px}
.copy{display:flex;gap:8px}.copy input{flex:1}
.sep{height:1px;background:rgba(255,255,255,.06);margin:10px 0}
.card.soft{background:linear-gradient(180deg,#0f1836,#0c152b)}
/* Chat drawer */
.chat-drawer{position:fixed;right:0;top:64px;bottom:0;width:var(--chatW);max-width:92vw;transform:translateX(100%);transition:transform .2s ease-out;background:linear-gradient(180deg,#0f1a33,#0b1326);border-left:1px solid var(--border);display:flex;flex-direction:column;z-index:35}
.chat-drawer.open{transform:translateX(0)}.chat-head{display:flex;align-items:center;justify-content:space-between;padding:10px 12px;border-bottom:1px solid var(--border)}
.chat-body{flex:1;overflow:auto;padding:10px 12px}.chat-input{display:flex;gap:8px;padding:10px 12px;border-top:1px solid var(--border)}.chat-input input{flex:1}
.msg{margin-bottom:12px;padding-bottom:8px;border-bottom:1px dashed rgba(255,255,255,.04);position:relative}
.msghead{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.msghead .time{margin-left:auto;color:#9eb3da;font-size:12px}
.badge{font-size:10px;padding:3px 7px;border-radius:999px;border:1px solid var(--border);letter-spacing:.2px}
.badge.member{background:#0c1631;color:#cfe6ff}.badge.admin{background:linear-gradient(135deg,#f59e0b,#fb923c);color:#1a1206;border-color:rgba(0,0,0,.2);font-weight:900}.badge.owner{background:linear-gradient(135deg,#3b82f6,#22c1dc);color:#041018;border-color:transparent;font-weight:900}
.level{font-size:10px;padding:3px 7px;border-radius:999px;background:#0b1f3a;color:#cfe6ff;border:1px solid var(--border)}
.user-link{cursor:pointer;font-weight:800;padding:2px 6px;border-radius:8px;background:#0b1f3a;border:1px solid var(--border)}

/* Floating Actions (header only now) */
.header-actions{display:flex;gap:10px;align-items:center}
.balance-pill{background:#0b1f3a;border:1px solid var(--border);border-radius:999px;padding:8px 12px;font-weight:800}

/* Modal */
.modal{position:fixed;inset:0;background:rgba(0,0,0,.55);display:none;align-items:center;justify-content:center;z-index:50}
.modal.open{display:flex}
.modal .box{width:min(520px,92vw);background:linear-gradient(180deg,#0f1a33,#0b1326);border:1px solid var(--border);border-radius:16px;padding:14px}
.modal .tabs{margin-top:8px}
.inline{display:flex;gap:8px;align-items:center}

/* Toasts */
#toasts{position:fixed;right:14px;bottom:14px;display:flex;flex-direction:column;gap:8px;z-index:60}
.toast{background:#0c1631;border:1px solid var(--border);border-left:4px solid #3b82f6;color:#e6eeff;border-radius:10px;padding:10px 12px;min-width:240px;max-width:360px;box-shadow:0 10px 24px rgba(0,0,0,.35)}
.toast.err{border-left-color:#ef4444}.toast.ok{border-left-color:#22c55e}

/* Loader */
#loader{position:fixed;inset:0;display:grid;place-items:center;background:#070c19;z-index:100}
.spin{width:56px;height:56px;border-radius:50%;border:6px solid rgba(255,255,255,.12);border-top-color:#3b82f6;animation:sp 1s linear infinite}
@keyframes sp{to{transform:rotate(360deg)}}
</style>
</head>
<body>
  <div id="loader"><div class="spin"></div></div>

  <div class="header">
    <div class="header-inner container">
      <div class="left">
        <a class="brand" href="/" id="homeLink">
          <img class="logo" src="/img/GrowCBnobackground.png" alt="GROWCB" onerror="this.style.display='none'"/>
          <span>GROWCB</span>
        </a>
        <div class="tabs">
          <a class="tab" data-path="/">Games</a>
          <a class="tab" data-path="/referral">Referral</a>
          <a class="tab" data-path="/promocodes">Promo Codes</a>
          <a class="tab" data-path="/leaderboard">Leaderboard</a>
        </div>
      </div>
      <div class="right header-actions" id="authArea"></div>
    </div>
  </div>

  <div class="container" style="padding-top:16px">
    <!-- Games -->
    <div id="page-games" data-route="/" style="display:none">
      <div class="card">
        <div class="games-grid" style="margin-top:6px">
          <div class="game-card" data-path="/crash"><img class="banner" src="/img/crash.png" alt="Crash"/></div>
          <div class="game-card" data-path="/mines"><img class="banner" src="/img/mines.png" alt="Mines"/></div>
          <div class="game-card" data-path="/coinflip"><img class="banner" src="/img/coinflip.png" alt="Coinflip"/></div>
          <div class="game-card" data-path="/blackjack"><img class="banner" src="/img/blackjack.png" alt="Blackjack"/></div>
          <div class="game-card" data-path="/pump"><img class="banner" src="/img/pump.png" alt="Pump"/></div>
        </div>
      </div>
    </div>

    <!-- Crash -->
    <div id="page-crash" data-route="/crash" style="display:none">
      <div class="card">
        <div class="hero">
          <div style="display:flex;align-items:baseline;gap:10px"><div class="big" id="crNow">0.00×</div><div class="muted" id="crHint">Loading…</div></div>
          <button class="btn soft" data-path="/">← Games</button>
        </div>
        <div class="cr-graph-wrap" style="margin-top:10px"><canvas id="crCanvas"></canvas></div>
        <div style="margin-top:12px"><div class="label" style="margin-bottom:4px">Previous Busts</div><div id="lastBusts" class="muted">Loading last rounds…</div></div>
        <div class="games-grid" style="grid-template-columns:1fr 1fr;gap:12px;margin-top:8px">
          <div class="field"><div class="label">Bet (DL)</div><input id="crBet" type="number" min="1" step="0.01" placeholder="min 1.00"/></div>
          <div class="field"><div class="label">Auto Cashout (×) — optional</div><input id="crCash" type="number" min="1.01" step="0.01" placeholder="e.g. 2.00"/></div>
        </div>
        <div class="inline" style="margin-top:8px">
          <button class="btn primary" id="crPlace">Place Bet</button>
          <button class="btn" id="crCashout" style="display:none">💸 Cash Out</button>
          <span id="crMsg" class="muted"></span>
        </div>
        <div class="card soft" style="margin-top:14px"><div class="label">Your recent rounds</div><div id="crLast" class="muted">—</div></div>
      </div>
    </div>

    <!-- Mines -->
    <div id="page-mines" data-route="/mines" style="display:none">
      <div class="card">
        <div class="hero"><div class="big">💣 Mines</div><button class="btn soft" data-path="/">← Games</button></div>
        <div class="grid-2" style="margin-top:12px">
          <div>
            <div id="mSetup">
              <div class="field"><div class="label">Bet (DL)</div><input id="mBet" type="number" min="1" step="0.01" placeholder="min 1.00"/></div>
              <div class="field" style="margin-top:10px"><div class="label">Mines (1–24)</div><input id="mMines" type="number" min="1" max="24" step="1" value="3"/></div>
              <div class="inline" style="margin-top:12px">
                <button class="btn primary" id="mStart">Start Game</button>
                <span id="mMsg" class="muted"></span>
              </div>
            </div>
            <div class="inline" style="margin-top:8px">
              <button class="btn ok" id="mCash" style="display:none">💸 Cash Out</button>
              <span class="chip" id="mMult">Multiplier: 1.0000×</span><span class="chip" id="mPotential">Potential: —</span>
            </div>
            <div class="kpi" style="margin-top:8px"><span class="pill" id="mHash">Commit: —</span><span class="pill" id="mStatus">Status: —</span><span class="pill" id="mPicks">Picks: 0</span><span class="pill" id="mBombs">Mines: 3</span></div>
            <div class="card soft" style="margin-top:14px"><div class="label">Recent Mines Games</div><div id="mHist" class="muted">—</div></div>
          </div>
          <div>
            <div class="card soft" style="min-height:420px;display:grid;place-items:center">
              <div id="mGrid" style="display:grid;gap:10px;grid-template-columns:repeat(5,64px)"></div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Simple placeholders -->
    <div id="page-coinflip" data-route="/coinflip" style="display:none"><div class="card"><div class="big">🪙 Coinflip</div><div class="muted" style="margin-top:8px">Coming soon.</div></div></div>
    <div id="page-blackjack" data-route="/blackjack" style="display:none"><div class="card"><div class="big">🃏 Blackjack</div><div class="muted" style="margin-top:8px">Coming soon.</div></div></div>
    <div id="page-pump" data-route="/pump" style="display:none"><div class="card"><div class="big">📈 Pump</div><div class="muted" style="margin-top:8px">Coming soon.</div></div></div>

    <!-- Referral -->
    <div id="page-ref" data-route="/referral" style="display:none">
      <div class="card">
        <div class="big">🙌 Referral Program</div>
        <div class="sep"></div>
        <div class="grid-2">
          <div class="card soft">
            <div class="label">Your Referral Handle</div>
            <div class="row cols-2" style="margin-top:6px">
              <div class="field"><input id="refName" placeholder="choose-handle (3-20 chars)"/></div>
              <button class="btn primary" id="refSave">Save</button>
            </div>
            <div class="muted" id="refMsg" style="margin-top:6px"></div>
            <div class="sep"></div>
            <div class="label">Share Link</div>
            <div class="copy" style="margin-top:6px">
              <input id="refLink" readonly value=""/>
              <button class="btn ghost" id="copyRef">Copy</button>
            </div>
          </div>
          <div class="card soft">
            <div class="label">Stats</div>
            <div class="kpi" style="margin-top:8px">
              <span class="pill">Clicks: <strong id="refClicks">0</strong></span>
              <span class="pill">Joins: <strong id="refJoins">0</strong></span>
            </div>
            <div class="muted" style="margin-top:6px">Clicks count when someone opens your link. Joins count when they sign in.</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Promo -->
    <div id="page-promo" data-route="/promocodes" style="display:none">
      <div class="card">
        <div class="big">🎁 Promo Codes</div>
        <div class="sep"></div>
        <div class="grid-2">
          <div class="card soft">
            <div class="label">Redeem</div>
            <div class="row cols-2" style="margin-top:6px">
              <div class="field"><input id="promoInput" placeholder="e.g. WELCOME10"/></div>
              <button class="btn primary" id="redeemBtn">Redeem</button>
            </div>
            <div id="promoMsg" class="muted" style="margin-top:6px"></div>
          </div>
          <div class="card soft">
            <div class="label">Your Redemptions</div>
            <div id="myCodes" class="muted" style="margin-top:8px">—</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Leaderboard -->
    <div id="page-lb" data-route="/leaderboard" style="display:none">
      <div class="card">
        <div class="hero"><div class="big">🏆 Leaderboard — Top Wagered</div><div class="chip" id="lbCountdown">—</div></div>
        <div class="inline" style="margin-top:10px">
          <div class="tabs" id="lbSeg"><a class="tab active" data-period="daily">Daily</a><a class="tab" data-period="monthly">Monthly</a><a class="tab" data-period="alltime">All-time</a></div>
          <span class="muted">Anonymous players show as “Anonymous”.</span>
        </div>
        <div id="lbWrap" class="muted" style="margin-top:8px">Loading…</div>
      </div>
    </div>

    <!-- Settings / Profile -->
    <div id="page-settings" data-route="/settings" style="display:none">
      <div class="card">
        <div class="label">Settings</div>
        <div style="margin-top:8px">
          <label style="display:flex;align-items:center;gap:10px">
            <input type="checkbox" id="anonToggle" style="width:auto"/>
            <span><strong>Anonymous Mode</strong> — hide your name & wager amounts from others.</span>
          </label>
          <div id="setMsg" class="muted" style="margin-top:8px"></div>
          <div class="sep"></div>
          <a class="btn ghost" href="/logout" style="margin-top:6px">Logout</a>
        </div>
      </div>
    </div>

    <div id="page-profile" data-route="/profile" style="display:none">
      <div class="card">
        <div class="label">Profile</div><div id="profileBox">Loading…</div>
        <div id="ownerPanel" class="card soft" style="display:none;margin-top:12px">
          <div class="label">Owner / Admin Panel</div>
          <div class="row cols-3" style="margin-top:6px">
            <div class="field"><div class="label">Discord ID or &lt;@mention&gt;</div><input id="tIdent" placeholder="ID or <@id>"/></div>
            <div class="field"><div class="label">Amount (+/- DL)</div><input id="tAmt" type="text" placeholder="10 or -5.25"/></div>
            <div style="align-self:end"><button class="btn primary" id="tApply">Apply</button></div>
          </div>
          <div id="tMsg" class="muted" style="margin-top:6px"></div>
        </div>
      </div>
    </div>
  </div>

  <!-- Chat Drawer -->
  <button class="btn primary" style="position:fixed;right:16px;bottom:16px;z-index:34" id="fabChat" title="Open chat">Chat</button>
  <div class="chat-drawer" id="chatDrawer">
    <div class="chat-head"><div>Global Chat <span id="chatNote" class="muted"></span></div><button class="btn soft" id="chatClose">Close</button></div>
    <div class="chat-body" id="chatBody"></div>
    <div class="chat-input"><input id="chatText" placeholder="Say something…"/><button class="btn primary" id="chatSend">Send</button></div>
  </div>

  <!-- Deposit/Withdraw Modal -->
  <div class="modal" id="dwModal">
    <div class="box">
      <div class="inline" style="justify-content:space-between">
        <div class="big">Deposit / Withdraw</div>
        <button class="btn soft" id="dwClose">Close</button>
      </div>
      <div class="tabs" style="margin-top:8px"><a class="tab active" id="tabDep">Deposit</a><a class="tab" id="tabWit">Withdraw</a></div>
      <div id="depPane" style="margin-top:10px">
        <div class="muted">If not logged in with Discord, provide your Discord Username or ID. Provide your GrowID — if incorrect, your transfer may not work.</div>
        <div class="row cols-2" style="margin-top:10px">
          <div class="field"><div class="label">Discord (optional if logged in)</div><input id="depDiscord" placeholder="username#1234 or 123456789012345678"/></div>
          <div class="field"><div class="label">GrowID</div><input id="depGrowID" placeholder="Your GrowID"/></div>
        </div>
        <div class="field" style="margin-top:10px"><div class="label">World</div><input id="depWorld" placeholder="WORLDNAME"/></div>
        <button class="btn primary" id="depSubmit" style="margin-top:10px">Submit Deposit Request</button>
      </div>
      <div id="witPane" style="display:none;margin-top:10px">
        <div class="row cols-2">
          <div class="field"><div class="label">Amount (DL)</div><input id="witAmt" type="number" min="1" step="0.01" placeholder="10.00"/></div>
          <div class="field"><div class="label">World</div><input id="witWorld" placeholder="WORLDNAME"/></div>
        </div>
        <button class="btn primary" id="witSubmit" style="margin-top:10px">Submit Withdraw Request</button>
      </div>
    </div>
  </div>

  <!-- Toasts -->
  <div id="toasts"></div>

<script>
const qs = id => document.getElementById(id);
const $$ = sel => Array.from(document.querySelectorAll(sel));
const j = async (url, init) => {
  const r = await fetch(url, init);
  if(!r.ok){
    const t = await r.text().catch(()=> '');
    try { const js = JSON.parse(t); throw new Error(js.detail || js.message || t || r.statusText); }
    catch { throw new Error(t || r.statusText); }
  }
  const ct = (r.headers.get('content-type')||'').toLowerCase();
  return ct.includes('application/json') ? r.json() : r.text();
};
const GEM = "💎"; const fmtDL = n => `${GEM} ${(Number(n)||0).toFixed(2)} DL`;

/* Toasts */
function toast(msg, type='info', ms=3800){
  const box = document.createElement('div');
  box.className = 'toast' + (type==='err' ? ' err' : type==='ok' ? ' ok' : '');
  box.textContent = msg;
  qs('toasts').appendChild(box);
  setTimeout(()=>{ box.style.opacity='0'; setTimeout(()=> box.remove(), 300); }, ms);
}

/* Router */
const routes = {
  "/": "page-games",
  "/crash": "page-crash",
  "/mines": "page-mines",
  "/coinflip": "page-coinflip",
  "/blackjack": "page-blackjack",
  "/pump": "page-pump",
  "/referral": "page-ref",
  "/promocodes": "page-promo",
  "/leaderboard": "page-lb",
  "/profile": "page-profile",
  "/settings": "page-settings"
};
function showRoute(path){
  for(const id of Object.values(routes)){ const el = qs(id); if(el) el.style.display = 'none'; }
  const page = routes[path] || "page-games";
  qs(page).style.display = '';
  // Tabs active
  $$('.tabs a.tab').forEach(a => {
    const p = a.getAttribute('data-path');
    if(!p) return;
    a.classList.toggle('active', p===path || (path==='/' && p==='/'));
  });
}
function nav(path){
  if(location.pathname !== path){
    history.pushState({}, '', path);
  }
  showRoute(path);
  // page-specific boots
  if(path==='/crash'){ openCrash(); }
  if(path==='/mines'){ refreshMines(); }
  if(path==='/leaderboard'){ refreshLeaderboard(); }
  if(path==='/profile'){ renderProfile(); }
  if(path==='/promocodes'){ renderPromo(); }
  if(path==='/referral'){ loadReferral(); }
}
window.addEventListener('popstate', ()=> showRoute(location.pathname));

// link wiring
document.addEventListener('click', (e)=>{
  const t = e.target.closest('[data-path]');
  if(t){ e.preventDefault(); nav(t.getAttribute('data-path')); }
});

/* Header / Auth */
async function renderHeader(){
  try{
    const me = await j('/api/me');
    const bal = await j('/api/balance');
    qs('authArea').innerHTML = `
      <button class="btn" id="btnDW">Deposit / Withdraw</button>
      <button class="btn primary" id="btnJoin">${me.in_guild ? 'In Discord' : 'Join Discord'}</button>
      <span class="balance-pill">Balance: <strong>${fmtDL(bal.balance)}</strong></span>
      <div class="avatar-wrap">
        <img class="avatar" id="avatarBtn" src="${me.avatar_url||''}" title="${me.username||'user'}"/>
        <div id="userMenu" class="menu">
          <div class="item" data-path="/profile">Profile</div>
          <div class="item" data-path="/settings">Settings</div>
          <a class="item" href="/logout">Logout</a>
        </div>
      </div>
    `;
    qs('btnJoin').onclick = joinDiscord;
    qs('btnDW').onclick = ()=> openDW();
    const menu = qs('userMenu'); const av = qs('avatarBtn');
    av.onclick = (e)=>{ e.stopPropagation(); menu.classList.toggle('open'); };
    document.body.addEventListener('click', ()=> menu.classList.remove('open'));
  }catch(_){
    qs('authArea').innerHTML = `
      <button class="btn" id="btnDW">Deposit / Withdraw</button>
      <a class="btn primary" href="/login">Login with Discord</a>`;
    qs('btnDW').onclick = ()=> openDW();
  }
}
async function joinDiscord(){
  try{ await j('/api/discord/join', { method:'POST' }); toast('Joined Discord!', 'ok'); renderHeader(); }
  catch(e){ toast(e.message || 'Could not join. Try relogin.', 'err'); }
}

/* Deposit / Withdraw Modal */
function openDW(){ qs('dwModal').classList.add('open'); }
function closeDW(){ qs('dwModal').classList.remove('open'); }
qs('dwClose').onclick = closeDW;
qs('tabDep').onclick = ()=>{ qs('tabDep').classList.add('active'); qs('tabWit').classList.remove('active'); qs('depPane').style.display=''; qs('witPane').style.display='none'; };
qs('tabWit').onclick = ()=>{ qs('tabWit').classList.add('active'); qs('tabDep').classList.remove('active'); qs('depPane').style.display='none'; qs('witPane').style.display=''; };
qs('depSubmit').onclick = async ()=>{
  try{
    const data = { kind:'deposit', discord_identity: qs('depDiscord').value.trim() || null, grow_id: qs('depGrowID').value.trim() || null, world: qs('depWorld').value.trim() };
    if(!data.world) throw new Error('World is required');
    await j('/api/transfer/create', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(data) });
    toast('Deposit request submitted. Check #tickets.', 'ok');
    closeDW();
  }catch(e){ toast(e.message, 'err'); }
};
qs('witSubmit').onclick = async ()=>{
  try{
    const data = { kind:'withdraw', amount: qs('witAmt').value.trim(), world: qs('witWorld').value.trim() };
    if(!data.world) throw new Error('World is required');
    if(!data.amount) throw new Error('Amount required');
    await j('/api/transfer/create', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(data) });
    toast('Withdraw request submitted. Staff will DM you.', 'ok');
    closeDW();
  }catch(e){ toast(e.message, 'err'); }
};

/* Referral */
async function loadReferral(){
  try{
    const st = await j('/api/referral/state');
    if(st && st.name){ qs('refName').value = st.name; qs('refLink').value = `${location.origin}/referral/` + st.name; }
    qs('refClicks').textContent = st.clicks||0; qs('refJoins').textContent = st.joined||0;
  }catch(err){ toast(err.message, 'err'); }
}
qs('refSave').onclick = async()=>{
  const name = qs('refName').value.trim();
  try{
    await j('/api/referral/set', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ name })});
    qs('refLink').value = `${location.origin}/referral/` + name.toLowerCase();
    toast('Referral saved.', 'ok');
  }catch(e){ toast(e.message, 'err'); }
};
qs('copyRef').onclick = ()=>{
  const inp = qs('refLink'); inp.select(); inp.setSelectionRange(0, 99999); document.execCommand('copy'); toast('Copied!', 'ok');
};

/* Promo */
async function renderPromo(){
  try{
    const my = await j('/api/promo/my');
    qs('myCodes').innerHTML = (my.rows && my.rows.length)
      ? '<table><thead><tr><th>Code</th><th>Redeemed</th></tr></thead><tbody>' +
        my.rows.map(r=>`<tr><td>${r.code}</td><td>${new Date(r.redeemed_at).toLocaleString()}</td></tr>`).join('') +
        '</tbody></table>' : '—';
  }catch(e){ toast(e.message, 'err'); }
}
qs('redeemBtn').onclick = async ()=>{
  const code = qs('promoInput').value.trim();
  try{
    const r = await j('/api/promo/redeem', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ code }) });
    toast('Redeemed! New balance: '+fmtDL(r.new_balance), 'ok');
    renderHeader(); renderPromo();
  }catch(e){ toast(e.message, 'err'); }
};

/* Profile */
async function renderProfile(){
  try{
    const p = await j('/api/profile');
    const role = p.role || 'member';
    const isOwner = (role==='owner') || (String(p.id||'') === String('__OWNER_ID__'));
    qs('profileBox').innerHTML = `
      <div class="games-grid" style="grid-template-columns:1fr 1fr 1fr">
        <div class="card soft"><div class="label">Level</div><div class="big">Lv ${p.level}</div><div class="muted">${p.xp} XP • ${p.progress_pct}% to next</div></div>
        <div class="card soft"><div class="label">Balance</div><div class="big">${fmtDL(p.balance)}</div></div>
        <div class="card soft"><div class="label">Role</div><div class="big" style="text-transform:uppercase">${role}</div></div>
      </div>
    `;
    qs('ownerPanel').style.display = isOwner ? '' : 'none';
    if(isOwner){
      qs('tApply').onclick = async ()=>{
        try{
          const r = await j('/api/admin/adjust',{ method:'POST', headers:{'Content-Type':'application/json'},
            body: JSON.stringify({ identifier: qs('tIdent').value.trim(), amount: qs('tAmt').value.trim(), reason: 'manual' })});
          toast('OK. New balance: ' + fmtDL(r.new_balance), 'ok');
          renderHeader();
        }catch(e){ toast(e.message, 'err'); }
      };
    }
  }catch(e){ toast(e.message, 'err'); }
}

/* Settings */
async function loadSettings(){
  try{ const r = await j('/api/settings/get'); qs('anonToggle').checked = !!(r && r.is_anon); }
  catch(_){}
}
qs('anonToggle')?.addEventListener('change', async (e)=>{
  try{
    const r = await j('/api/settings/set_anon', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ is_anon: !!e.target.checked }) });
    toast('Saved.', 'ok');
  }catch(err){ toast(err.message, 'err'); }
});

/* Leaderboard */
let lbPeriod = 'daily';
function nextUtcMidnight(){ const now = new Date(); return new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()+1,0,0,0,0)); }
function endOfUtcMonth(){ const now = new Date(); return new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth()+1,1,0,0,0,0)); }
async function refreshLeaderboard(){
  const wrap = qs('lbWrap'); wrap.textContent = 'Loading…';
  const res = await j('/api/leaderboard?period='+lbPeriod+'&limit=50');
  const rows = res.rows||[];
  const me = await j('/api/me').catch(()=>null);
  const uid = me?.id || '';
  const html = rows.length ? `
    <table>
      <thead><tr><th>#</th><th>Name</th><th>Wagered</th></tr></thead>
      <tbody>
        ${rows.map((r, i)=>{
          const isMe = String(r.user_id)===String(uid);
          const name = r.is_anon ? 'Anonymous' : r.display_name;
          const amt = r.is_anon ? '—' : fmtDL(r.total_wagered);
          return `<tr class="${isMe?'me-row':''} ${r.is_anon?'anon':''}"><td>${i+1}</td><td class="name">${name}</td><td>${amt}</td></tr>`;
        }).join('')}
      </tbody>
    </table>` : '—';
  wrap.innerHTML = html;
  const tgt = lbPeriod==='daily' ? nextUtcMidnight() : lbPeriod==='monthly' ? endOfUtcMonth() : null;
  qs('lbCountdown').textContent = tgt ? 'Resets soon…' : 'All-time';
  const seg = qs('lbSeg');
  seg.querySelectorAll('a.tab').forEach(b=>{
    b.classList.toggle('active', b.dataset.period===lbPeriod);
    b.onclick = ()=>{ lbPeriod = b.dataset.period; refreshLeaderboard(); };
  });
}

/* Crash UI */
const crCanvas = ()=> qs('crCanvas');
let crPollTimer=null, crBust=null, crPhase='betting';
function drawCrash(mult){
  const c = crCanvas(); if(!c) return; const ctx = c.getContext('2d');
  const w = c.width = c.clientWidth || 600; const h = c.height = c.clientHeight || 240;
  ctx.fillStyle='#0e1833'; ctx.fillRect(0,0,w,h);
  ctx.strokeStyle='#9eb3da'; ctx.lineWidth=2; ctx.beginPath();
  const maxX = w-40, base=20, maxY=h-20;
  const maxMult = Math.max(2, mult);
  for(let x=0; x<=maxX; x++){
    const t = x/maxX; const m = 1 + (maxMult-1)*t*t;
    const y = maxY - (m-1)/(maxMult-1+1e-9) * (maxY-base);
    if(x===0) ctx.moveTo(base, y); else ctx.lineTo(base+x, y);
  }
  ctx.stroke();
}
async function pollCrash(){
  try{
    const st = await j('/api/crash/state');
    crPhase = st.phase; crBust = st.bust;
    qs('lastBusts').textContent = (st.last_busts||[]).map(v=> (Number(v)||0).toFixed(2)+'×').join(' • ') || '—';
    const cashBtn = qs('crCashout'); const you = st.your_bet;
    cashBtn.style.display = (you && crPhase==='running' && !you.cashed_out) ? '' : 'none';
    if(crPhase==='running' && st.current_multiplier){
      const m = Number(st.current_multiplier)||1.0;
      qs('crNow').textContent = m.toFixed(2)+'×'; qs('crHint').textContent = 'In flight…'; drawCrash(m);
    }else if(crPhase==='betting'){
      qs('crNow').textContent = '0.00×';
      const ends = st.betting_ends_at? new Date(st.betting_ends_at): null;
      if(ends){
        const left = Math.max(0, Math.floor((ends - new Date())/1000));
        qs('crHint').textContent = `Betting… ${left}s`;
      }else qs('crHint').textContent = 'Betting…';
      drawCrash(1);
    }else if(crPhase==='ended'){
      qs('crNow').textContent = (Number(crBust)||0).toFixed(2)+'×';
      qs('crHint').textContent = 'Round ended'; drawCrash(Number(crBust)||1);
    }
  }catch(e){ toast(e.message, 'err'); }
  finally{ crPollTimer = setTimeout(pollCrash, 900); }
}
qs('crPlace').onclick = async ()=>{
  const bet = qs('crBet').value || '0';
  const cash = qs('crCash').value || null;
  try{
    await j('/api/crash/place', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ bet, cashout: cash? Number(cash): null })});
    toast('Bet placed.', 'ok');
  }catch(e){ toast(e.message, 'err'); }
};
qs('crCashout').onclick = async ()=>{
  try{ await j('/api/crash/cashout', { method:'POST' }); toast('Cashed out!', 'ok'); }
  catch(e){ toast(e.message, 'err'); }
};
function openCrash(){
  if(crPollTimer) clearTimeout(crPollTimer);
  pollCrash();
}

/* Mines */
function buildMinesGrid(){
  const grid = qs('mGrid'); grid.innerHTML='';
  for(let i=0;i<25;i++){
    const b = document.createElement('button');
    b.textContent = '?';
    b.style.width='64px'; b.style.height='64px'; b.style.borderRadius='12px';
    b.style.border='1px solid var(--border)'; b.style.background='#0f1a33'; b.style.color='#cfe6ff';
    b.dataset.index = i;
    b.onclick = ()=> pickCell(i);
    grid.appendChild(b);
  }
}
async function pickCell(i){
  try{ await j('/api/mines/pick?index='+i, { method:'POST' }); await refreshMines(); }
  catch(e){ toast(e.message, 'err'); }
}
async function startMines(){
  const bet = qs('mBet').value || '0';
  const mines = parseInt(qs('mMines').value||'3',10);
  try{ await j('/api/mines/start', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ bet, mines })}); await refreshMines(); }
  catch(e){ toast(e.message, 'err'); }
}
async function cashoutMines(){
  try{ await j('/api/mines/cashout', { method:'POST' }); await refreshMines(); }
  catch(e){ toast(e.message, 'err'); }
}
async function refreshMines(){
  try{
    const st = await j('/api/mines/state');
    const grid = qs('mGrid'); const status = st?.status || 'idle';
    qs('mStatus').textContent = 'Status: ' + status;
    qs('mPicks').textContent = 'Picks: ' + (st?.picks||0);
    qs('mBombs').textContent = 'Mines: ' + (st?.mines|| (qs('mMines').value||3));
    qs('mHash').textContent = 'Commit: ' + (st?.commit_hash || '—');
    qs('mMult').textContent = 'Multiplier: ' + (st?.multiplier ? (Number(st.multiplier)||1).toFixed(4)+'×' : '1.0000×');
    qs('mPotential').textContent = 'Potential: ' + (st?.potential_win ? fmtDL(st.potential_win) : '—');
    const playing = status==='active'; qs('mCash').style.display = playing ? '' : 'none'; qs('mSetup').style.display = playing ? 'none' : '';
    if(grid.children.length!==25) buildMinesGrid();
    if(st?.reveals && Array.isArray(st.reveals)){
      st.reveals.forEach((cell, idx)=>{
        const b = grid.children[idx]; if(!b) return;
        if(cell === 'u'){ b.textContent='?'; b.disabled=false; b.style.background='#0f1a33'; }
        else if(cell==='g'){ b.textContent='✅'; b.disabled=true; b.style.background='#163a2a'; }
        else if(cell==='b'){ b.textContent='💣'; b.disabled=true; b.style.background='#3a1620'; }
      });
    }
    const h = await j('/api/mines/history');
    qs('mHist').innerHTML = (h.rows && h.rows.length)
      ? '<table><thead><tr><th>Time</th><th>Bet</th><th>Mines</th><th>Win</th><th>Status</th></tr></thead><tbody>' +
        h.rows.map(r=>`<tr><td>${new Date(r.started_at).toLocaleString()}</td><td>${fmtDL(r.bet)}</td><td>${r.mines}</td><td>${fmtDL(r.win)}</td><td>${r.status}</td></tr>`).join('') +
        '</tbody></table>' : '—';
  }catch(e){ /* silent */ }
}
qs('mStart').onclick = startMines;
qs('mCash').onclick = cashoutMines;

/* Chat UI */
let chatOpen = false, chatTimer=null, lastChatId=0;
function toggleChat(open){
  chatOpen = open;
  qs('chatDrawer').classList.toggle('open', open);
  if(open){ pollChat(); } else { if(chatTimer) clearTimeout(chatTimer); }
}
qs('fabChat').onclick = ()=> toggleChat(true);
qs('chatClose').onclick = ()=> toggleChat(false);

async function pollChat(){
  try{
    const r = await j('/api/chat/fetch?since='+lastChatId+'&limit=50');
    const arr = r.rows||[];
    if(arr.length){
      const body = qs('chatBody');
      for(const m of arr){
        lastChatId = Math.max(lastChatId, m.id||0);
        const row = document.createElement('div'); row.className='msg';
        row.innerHTML = `
          <div class="msghead">
            <span class="user-link" data-uid="${m.user_id}">${m.username}</span>
            <span class="badge ${m.role}">${m.role}</span>
            <span class="level">Lv ${m.level}</span>
            <span class="time">${new Date(m.created_at).toLocaleTimeString()}</span>
          </div>
          <div>${escapeHtml(m.text)}</div>
        `;
        body.appendChild(row);
      }
      qs('chatBody').scrollTop = qs('chatBody').scrollHeight;
    }
  }catch(_){}
  finally{ chatTimer = setTimeout(pollChat, 1200); }
}
function escapeHtml(s){ return String(s||'').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
qs('chatBody').addEventListener('click', async (e)=>{
  const u = e.target.closest('.user-link'); if(!u) return;
  try{
    const p = await j('/api/profile/public?user_id='+encodeURIComponent(u.dataset.uid));
    toast(`${p.name} — Lv ${p.level} • ${fmtDL(p.balance)}`, 'info', 5200);
  }catch(_){}
});
qs('chatSend').onclick = async ()=>{
  const t = qs('chatText').value.trim(); if(!t) return;
  try{ await j('/api/chat/send', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ text: t })}); qs('chatText').value=''; }
  catch(e){ toast(e.message, 'err'); }
};

/* Boot */
(async function boot(){
  buildMinesGrid();
  await renderHeader();
  // attach referral cookie (if present)
  const m = document.cookie.match(/(?:^|; )refname=([^;]+)/); if(m){ try{ await j('/api/referral/attach?refname='+decodeURIComponent(m[1])); }catch(_){ } }
  // route by path
  const path = routes[location.pathname] ? location.pathname : '/';
  showRoute(path);
  if(path==='/crash') openCrash();
  if(path==='/leaderboard') refreshLeaderboard();
  if(path==='/mines') refreshMines();
  // wire top tabs
  $$('.tabs a.tab').forEach(a => a.onclick = (e)=>{ e.preventDefault(); const p=a.getAttribute('data-path'); if(p) nav(p); });
  // games grid links
  $$('.game-card').forEach(c => c.onclick = ()=> nav(c.getAttribute('data-path')));
  // hide loader
  qs('loader').style.display='none';
})();
</script>
</body>
</html>
"""

# ---------- SPA routes for pretty URLs ----------
@app.get("/", response_class=HTMLResponse)
async def index_root():
    html = HTML_TEMPLATE.replace("__OWNER_ID__", str(OWNER_ID))
    return HTMLResponse(html)

@app.get("/crash", response_class=HTMLResponse)
async def spa_crash(): return await index_root()

@app.get("/mines", response_class=HTMLResponse)
async def spa_mines(): return await index_root()

@app.get("/coinflip", response_class=HTMLResponse)
async def spa_cf(): return await index_root()

@app.get("/blackjack", response_class=HTMLResponse)
async def spa_bj(): return await index_root()

@app.get("/pump", response_class=HTMLResponse)
async def spa_pu(): return await index_root()

@app.get("/leaderboard", response_class=HTMLResponse)
async def spa_lb(): return await index_root()

@app.get("/promocodes", response_class=HTMLResponse)
async def spa_pc(): return await index_root()

@app.get("/profile", response_class=HTMLResponse)
async def spa_profile(): return await index_root()

@app.get("/settings", response_class=HTMLResponse)
async def spa_settings(): return await index_root()

@app.get("/referral", response_class=HTMLResponse)
async def spa_referral(): return await index_root()

# ---------- Discord Bot Commands ----------
def _parse_target_id(text: Optional[str]) -> Optional[int]:
    if not text: return None
    m = re.search(r"(\d{5,})", text) or re.search(r"<@!?(\d+)>", text)
    if m:
        try: return int(m.group(1))
        except: return None
    return None

def is_owner_or_webhook():
    async def predicate(ctx: commands.Context):
        if getattr(ctx.message, "webhook_id", None):
            return True
        return int(getattr(ctx.author, "id", 0)) == int(OWNER_ID)
    return commands.check(predicate)

@bot.event
async def on_ready():
    print(f"Bot ready as {bot.user} ({bot.user.id})")

def _embed(title: str, desc: str = "", color: int = 0x3b82f6):
    e = discord.Embed(title=title, description=desc, color=color)
    e.set_footer(text=f"{SITE_BASE}/")
    return e

@bot.command(name="help")
async def help_cmd(ctx: commands.Context):
    e = _embed("GROWCB — Help", color=0x22c1dc)
    e.add_field(name=".bal [@user|id]", value="Show your balance or another user’s balance.", inline=False)
    e.add_field(name=".level [@user|id]", value="Show your level, XP, and progress.", inline=False)
    e.add_field(name=".leaderboard", value="Show top wagered. Use buttons to switch Daily / Monthly / All-time.", inline=False)
    e.add_field(name=".addbal (owner/webhook)", value="`.addbal <@user|id> <amount>` — add DL.", inline=False)
    e.add_field(name=".removebal (owner/webhook)", value="`.removebal <@user|id> <amount>` — remove DL.", inline=False)
    await ctx.send(embed=e)

@bot.command(name="bal")
async def bal_cmd(ctx: commands.Context, target: Optional[str] = None):
    uid = int(ctx.author.id) if not target else (_parse_target_id(target) or int(ctx.author.id))
    bal = float(get_balance(str(uid)))
    name = getattr(ctx.author, "display_name", "you") if uid==ctx.author.id else (target or str(uid))
    e = _embed("Balance", color=0x6aa6ff)
    e.add_field(name="User", value=f"<@{uid}>", inline=True)
    e.add_field(name="Balance", value=f"{GEM} {bal:.2f} DL", inline=True)
    await ctx.send(embed=e)

@bot.command(name="level")
async def level_cmd(ctx: commands.Context, target: Optional[str] = None):
    uid = int(ctx.author.id) if not target else (_parse_target_id(target) or int(ctx.author.id))
    p = profile_info(str(uid))
    # Make it pretty: progress bar
    pct = max(0, min(100, int(p["progress_pct"])))
    bars = int(pct/5)
    bar = "█"*bars + "░"*(20-bars)
    e = _embed("Level & XP", color=0x22c55e)
    e.add_field(name="User", value=f"<@{uid}>", inline=True)
    e.add_field(name="Level", value=f"Lv {p['level']}", inline=True)
    e.add_field(name="XP", value=f"{p['xp']} ({pct}% to next)", inline=True)
    e.add_field(name="Progress", value=f"`{bar}`", inline=False)
    await ctx.send(embed=e)

class LBView(discord.ui.View):
    def __init__(self): super().__init__(timeout=60)
    async def _send(self, interaction: discord.Interaction, period: str):
        rows = get_leaderboard_rows_db(period, 10)
        title = "Leaderboard — " + (period.capitalize() if period!="alltime" else "All-time")
        e = _embed(title, color=0x3b82f6)
        if not rows:
            e.description = "No data."
        else:
            lines = []
            for i, r in enumerate(rows, 1):
                name = "Anonymous" if r["is_anon"] else r["display_name"]
                amt = "—" if r["is_anon"] else f"{GEM} {float(r['total_wagered']):.2f} DL"
                lines.append(f"**{i}.** {name} — {amt}")
            e.description = "\n".join(lines)
        await interaction.response.edit_message(embed=e, view=self)
    @discord.ui.button(label="Daily", style=discord.ButtonStyle.primary)
    async def daily(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._send(interaction, "daily")
    @discord.ui.button(label="Monthly", style=discord.ButtonStyle.secondary)
    async def monthly(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._send(interaction, "monthly")
    @discord.ui.button(label="All-time", style=discord.ButtonStyle.secondary)
    async def alltime(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._send(interaction, "alltime")

@bot.command(name="leaderboard")
async def leaderboard_cmd(ctx: commands.Context):
    rows = get_leaderboard_rows_db("daily", 10)
    e = _embed("Leaderboard — Daily", color=0x3b82f6)
    if not rows:
        e.description = "No data."
    else:
        lines = []
        for i, r in enumerate(rows, 1):
            name = "Anonymous" if r["is_anon"] else r["display_name"]
            amt = "—" if r["is_anon"] else f"{GEM} {float(r['total_wagered']):.2f} DL"
            lines.append(f"**{i}.** {name} — {amt}")
        e.description = "\n".join(lines)
    await ctx.send(embed=e, view=LBView())

@bot.command(name="addbal")
@is_owner_or_webhook()
async def addbal_cmd(ctx: commands.Context, target: str, amount: str):
    uid = _id_from_identifier(target)
    newbal = adjust_balance(str(OWNER_ID), uid, D(amount), "bot:addbal")
    e = _embed("Balance Added", color=0x22c55e)
    e.add_field(name="User", value=f"<@{uid}>", inline=True)
    e.add_field(name="New Balance", value=f"{GEM} {float(newbal):.2f} DL", inline=True)
    await ctx.send(embed=e)

@bot.command(name="removebal")
@is_owner_or_webhook()
async def removebal_cmd(ctx: commands.Context, target: str, amount: str):
    uid = _id_from_identifier(target)
    newbal = adjust_balance(str(OWNER_ID), uid, D("-"+str(amount).lstrip("+")), "bot:removebal")
    e = _embed("Balance Removed", color=0xef4444)
    e.add_field(name="User", value=f"<@{uid}>", inline=True)
    e.add_field(name="New Balance", value=f"{GEM} {float(newbal):.2f} DL", inline=True)
    await ctx.send(embed=e)

# ---------- Run local ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
