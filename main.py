# app/main.py — games removed (Crash & Mines imported from separate modules)

import os, json, asyncio, re, random, string, math, secrets, datetime, hashlib
from urllib.parse import urlencode
from typing import Optional, Tuple, Dict, List
from decimal import Decimal, ROUND_DOWN, getcontext
from contextlib import asynccontextmanager

import httpx
import psycopg
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from itsdangerous import URLSafeSerializer, BadSignature
import uvicorn
from pydantic import BaseModel

# ---------- Import games from separate files ----------
# Make sure crash.py and mines.py are in the same directory and expose these functions.
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

GEM = "💎"
MIN_BET = Decimal("1.00")
MAX_BET = Decimal("1000000.00")
BETTING_SECONDS = 10

HOUSE_EDGE_CRASH = Decimal(os.getenv("HOUSE_EDGE_CRASH", "0.06"))
HOUSE_EDGE_MINES = Decimal(os.getenv("HOUSE_EDGE_MINES", "0.03"))

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
    os.makedirs(static_dir, exist_ok=True)
    return static_dir

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    apply_migrations()
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=_get_static_dir()), name="static")

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
    cur.execute("ALTER TABLE balances ALTER COLUMN balance TYPE NUMERIC(18,2) USING balance::numeric")

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
    cur.execute("ALTER TABLE balance_log ALTER COLUMN amount TYPE NUMERIC(18,2) USING amount::numeric")

    # profiles (+ referral)
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
    cur.execute("ALTER TABLE profiles ADD COLUMN IF NOT EXISTS role TEXT NOT NULL DEFAULT 'member'")
    cur.execute("ALTER TABLE profiles ADD COLUMN IF NOT EXISTS is_anon BOOLEAN NOT NULL DEFAULT FALSE")
    cur.execute("ALTER TABLE profiles ADD COLUMN IF NOT EXISTS referred_by TEXT")

    # oauth tokens for guild join
    cur.execute("""
        CREATE TABLE IF NOT EXISTS oauth_tokens (
            user_id TEXT PRIMARY KEY,
            access_token TEXT NOT NULL,
            refresh_token TEXT,
            expires_at TIMESTAMPTZ
        )
    """)

    # referral registry
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
    cur.execute("ALTER TABLE promo_codes ALTER COLUMN amount TYPE NUMERIC(18,2) USING amount::numeric")
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
    cur.execute("ALTER TABLE crash_bets ALTER COLUMN bet TYPE NUMERIC(18,2) USING bet::numeric")
    cur.execute("ALTER TABLE crash_bets ALTER COLUMN win TYPE NUMERIC(18,2) USING win::numeric")
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
    cur.execute("ALTER TABLE crash_games ALTER COLUMN bet TYPE NUMERIC(18,2) USING bet::numeric")
    cur.execute("ALTER TABLE crash_games ALTER COLUMN win TYPE NUMERIC(18,2) USING win::numeric")

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
    cur.execute("ALTER TABLE mines_games ALTER COLUMN bet TYPE NUMERIC(18,2) USING bet::numeric")
    cur.execute("ALTER TABLE mines_games ALTER COLUMN win TYPE NUMERIC(18,2) USING win::numeric")

# ---- balances / profiles ----
@with_conn
def get_balance(cur, user_id: str) -> Decimal:
    cur.execute("SELECT balance FROM balances WHERE user_id = %s", (user_id,))
    row = cur.fetchone(); return q2(row[0]) if row else Decimal("0.00")

@with_conn
def adjust_balance(cur, actor_id: str, target_id: str, amount, reason: Optional[str]) -> Decimal:
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
        if r.status_code == 409: return {"ok": True}  # already a member
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

# ---------- Leaderboard ----------
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

@app.get("/r/{refname}")
async def referral_landing(refname: str, request: Request):
    refname = (refname or "").lower()
    with psycopg.connect(DATABASE_URL) as con, con.cursor() as cur:
        cur.execute("SELECT user_id FROM ref_names WHERE name_lower=%s", (refname,))
        r = cur.fetchone()
        if r:
            referrer = str(r[0])
            cur.execute("INSERT INTO ref_visits(referrer_id) VALUES (%s)", (referrer,))
            con.commit()
    html = f"""
    <script>
      document.cookie = "refname={refname}; path=/; max-age=1209600; samesite=lax";
      location.href = "/";
    </script>
    """
    return HTMLResponse(html)

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

# Admin create promo (needed by UI)
class PromoCreateIn(BaseModel):
    code: Optional[str] = None
    amount: str
    max_uses: int = 1
    expires_at: Optional[str] = None

@app.post("/api/admin/promo/create")
async def api_admin_promo_create(request: Request, body: PromoCreateIn):
    s = _require_session(request)
    role = get_role(s["id"])
    if role not in ("admin", "owner"): raise HTTPException(403, "No permission")
    res = create_promo(s["id"], body.code, body.amount, int(body.max_uses or 1), body.expires_at)
    return res

# ---------- Crash endpoints (delegating to crash.py) ----------
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

    # Progress state machine on every poll (keeps Crash moving)
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
    # Recompute current state to ensure still running
    cur = load_round()
    if not cur or cur["status"] != "running":
        raise HTTPException(400, "No running round")
    # Now settle
    return cashout_now(s["id"])

@app.get("/api/crash/history")
async def api_crash_history(request: Request):
    s = _require_session(request)
    return {"rows": your_history(s["id"], 10)}

# ---------- Mines endpoints (delegating to mines.py) ----------
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
        # Everyone can chat if logged in, unless timed out
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
:root{--bg:#0a0f1e;--bg2:#0c1428;--card:#111a31;--muted:#9eb3da;--text:#ecf2ff;--accent:#6aa6ff;--accent2:#22c1dc;--ok:#34d399;--warn:#f59e0b;--err:#ef4444;--border:#1f2b47;--chatW:340px;--input-bg:#0b1430;--input-br:#223457;--input-tx:#e6eeff;--input-ph:#9db4e4}
*{box-sizing:border-box}html,body{height:100%}body{margin:0;color:var(--text);background:radial-gradient(1400px 600px at 20% -10%, #11204d 0%, transparent 60%),linear-gradient(180deg,#0a0f1e,#0a0f1e 60%, #0b1124);font-family:Inter,system-ui,Segoe UI,Roboto,Arial,Helvetica,sans-serif;-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale}
a{color:inherit;text-decoration:none}.container{max-width:1120px;margin:0 auto;padding:16px}
input,select,textarea{width:100%;appearance:none;background:var(--input-bg);color:var(--input-tx);border:1px solid var(--input-br);border-radius:12px;padding:10px 12px;outline:none;transition:border-color .15s ease, box-shadow .15s ease}
input::placeholder{color:var(--input-ph)}input:focus{border-color:#4c78ff;box-shadow:0 0 0 3px rgba(76,120,255,.18)}
.field{display:flex;flex-direction:column;gap:6px}.row{display:grid;gap:10px}
.row.cols-2{grid-template-columns:1fr 1fr}.row.cols-3{grid-template-columns:1fr 1fr 1fr}.row.cols-4{grid-template-columns:1.6fr 1fr 1fr auto}.row.cols-5{grid-template-columns:2fr 1fr 1fr auto auto}
.card{background:linear-gradient(180deg,#0f1a33,#0b1326);border:1px solid var(--border);border-radius:16px;padding:16px}
.header{position:sticky;top:0;z-index:30;backdrop-filter:blur(8px);background:rgba(10,15,30,.72);border-bottom:1px solid var(--border)}
.header-inner{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:10px 12px}
.left{display:flex;align-items:center;gap:14px;flex:1;min-width:0}.brand{display:flex;align-items:center;gap:10px;font-weight:800;white-space:nowrap}
.brand .logo{width:28px;height:28px;border-radius:8px;background:linear-gradient(135deg,var(--accent),var(--accent2))}
.tabs{display:flex;gap:4px;align-items:center;padding:4px;border-radius:14px;background:linear-gradient(180deg,#0f1a33,#0b1326);border:1px solid var(--border)}
.tab{padding:8px 12px;border-radius:10px;cursor:pointer;font-weight:700;white-space:nowrap;color:#d8e6ff;opacity:.85;transition:all .15s ease;display:flex;align-items:center;gap:8px}
.tab:hover{opacity:1;transform:translateY(-1px)}.tab.active{background:linear-gradient(135deg,#3b82f6,#22c1dc);color:#051326;box-shadow:0 6px 16px rgba(59,130,246,.25);opacity:1}
.right{display:flex;gap:8px;align-items:center;margin-left:12px}
.chip{background:#0c1631;border:1px solid var(--border);color:#dce7ff;padding:6px 10px;border-radius:999px;font-size:12px;white-space:nowrap;cursor:pointer}
.avatar{width:34px;height:34px;border-radius:50%;object-fit:cover;border:1px solid var(--border);cursor:pointer}
.avatar-wrap{position:relative}
.menu{position:absolute;right:0;top:40px;background:#0c1631;border:1px solid var(--border);border-radius:12px;padding:6px;display:none;min-width:160px;z-index:50}
.menu.open{display:block}.menu .item{padding:8px 10px;border-radius:8px;cursor:pointer;font-size:14px}.menu .item:hover{background:#11234a}
.btn{display:inline-flex;align-items:center;gap:8px;padding:10px 14px;border-radius:12px;border:1px solid var(--border);background:linear-gradient(180deg,#0e1833,#0b1326);cursor:pointer;font-weight:600}
.btn.primary{background:linear-gradient(135deg,#3b82f6,#22c1dc);border-color:transparent}
.btn.ghost{background:#162a52;border:1px solid var(--border);color:#eaf2ff}
.btn.ok{background:linear-gradient(135deg,#22c55e,#16a34a);border-color:transparent}
.big{font-size:22px;font-weight:900}.label{font-size:12px;color:var(--muted);letter-spacing:.2px;text-transform:uppercase}.muted{color:var(--muted)}
.games-grid{display:grid;gap:14px;grid-template-columns:1fr}@media(min-width:700px){.games-grid{grid-template-columns:1fr 1fr}}@media(min-width:1020px){.games-grid{grid-template-columns:1fr 1fr 1fr}}
.game-card{position:relative;min-height:130px;display:flex;flex-direction:column;justify-content:flex-end;gap:4px;background:linear-gradient(180deg,#0f1a33,#0c152a);border:1px solid var(--border);border-radius:16px;padding:16px;cursor:pointer;transition:transform .08s ease, box-shadow .12s ease, border-color .12s ease, background .18s ease;overflow:hidden}
.game-card:hover{transform:translateY(-2px);box-shadow:0 8px 18px rgba(0,0,0,.25)}.game-card .title{font-size:20px;font-weight:800}
.ribbon{position:absolute;top:12px;right:-32px;transform:rotate(35deg);background:linear-gradient(135deg,#f59e0b,#fb923c);color:#1a1206;font-weight:900;padding:6px 50px;border:1px solid rgba(0,0,0,.2)}
.cr-graph-wrap{position:relative;height:240px;background:#0e1833;border:1px solid var(--border);border-radius:16px;overflow:hidden}
canvas{display:block;width:100%;height:100%}
.lb-controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
.seg{display:flex;border:1px solid var(--border);border-radius:12px;overflow:hidden}
.seg button{padding:8px 12px;background:#0c1631;color:#dce7ff;border:none;cursor:pointer}
.seg button.active{background:linear-gradient(135deg,#3b82f6,#22c1dc);color:#051326;font-weight:800}
table{width:100%;border-collapse:collapse}th,td{padding:10px;border-bottom:1px solid rgba(255,255,255,.06);text-align:left}
tr.me-row{background:linear-gradient(90deg, rgba(34,197,94,.12), transparent 60%)}tr.anon td.name{color:#9db4e4;font-style:italic}
.countdown{font-size:12px;color:var(--muted)}.hint{font-size:12px;color:var(--muted);margin-top:6px}
.grid-2{display:grid;grid-template-columns:1fr;gap:16px}@media(min-width:900px){.grid-2{grid-template-columns:1.1fr .9fr}}
.hero{display:flex;justify-content:space-between;align-items:center;gap:14px;flex-wrap:wrap}
.kpi{display:flex;gap:8px;flex-wrap:wrap}
.kpi .pill{background:#0c1631;border:1px solid var(--border);border-radius:999px;padding:6px 10px;font-size:12px}
.copy{display:flex;gap:8px}
.copy input{flex:1}
.sep{height:1px;background:rgba(255,255,255,.06);margin:10px 0}
.discord-cta{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.bad{color:#ffb4b4}.good{color:#b7ffcc}
.card.soft{background:linear-gradient(180deg,#0f1836,#0c152b)}
.chat-drawer{position:fixed;right:0;top:64px;bottom:0;width:var(--chatW);max-width:92vw;transform:translateX(100%);transition:transform .2s ease-out;background:linear-gradient(180deg,#0f1a33,#0b1326);border-left:1px solid var(--border);display:flex;flex-direction:column;z-index:40}
.chat-drawer.open{transform:translateX(0)}.chat-head{display:flex;align-items:center;justify-content:space-between;padding:10px 12px;border-bottom:1px solid var(--border)}
.chat-body{flex:1;overflow:auto;padding:10px 12px}.chat-input{display:flex;gap:8px;padding:10px 12px;border-top:1px solid var(--border)}.chat-input input{flex:1}
.msg{margin-bottom:12px;padding-bottom:8px;border-bottom:1px dashed rgba(255,255,255,.04);position:relative}
.msghead{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.msghead .time{margin-left:auto;color:var(--muted);font-size:12px}
.badge{font-size:10px;padding:3px 7px;border-radius:999px;border:1px solid var(--border);letter-spacing:.2px}
.badge.member{background:#0c1631;color:#cfe6ff}.badge.admin{background:linear-gradient(135deg,#f59e0b,#fb923c);color:#1a1206;border-color:rgba(0,0,0,.2);font-weight:900}.badge.owner{background:linear-gradient(135deg,#3b82f6,#22c1dc);color:#041018;border-color:transparent;font-weight:900}
.level{font-size:10px;padding:3px 7px;border-radius:999px;background:#0b1f3a;color:#cfe6ff;border:1px solid var(--border)}
.user-link{cursor:pointer;font-weight:800;padding:2px 6px;border-radius:8px;background:#0b1f3a;border:1px solid var(--border)}
.fab{position:fixed;right:18px;bottom:18px;width:56px;height:56px;border-radius:50%;background:linear-gradient(135deg,#3b82f6,#22c1dc);border:none;cursor:pointer;display:flex;align-items:center;justify-content:center;box-shadow:0 14px 30px rgba(59,130,246,.35), 0 4px 10px rgba(0,0,0,.35);z-index:45}
.fab svg{width:26px;height:26px;fill:#041018}
</style>
</head>
<body>
  <div class="header">
    <div class="header-inner container">
      <div class="left">
        <a class="brand" href="#" id="homeLink"><span class="logo"></span> GROWCB</a>
        <div class="tabs">
          <a class="tab active" id="tab-games">Games</a>
          <a class="tab" id="tab-ref">Referral</a>
          <a class="tab" id="tab-promo">Promo Codes</a>
          <a class="tab" id="tab-lb">Leaderboard</a>
          <!-- Settings tab intentionally hidden (open from avatar menu) -->
        </div>
      </div>
      <div class="right" id="authArea"></div>
    </div>
  </div>

  <div class="container" style="padding-top:16px">
    <!-- Games -->
    <div id="page-games">
      <div class="card">
        <div class="hero">
          <div class="big">Welcome to GROWCB</div>
          <div class="discord-cta">
            <button class="btn ghost" id="btnJoinDiscord">Join Discord</button>
            <a class="chip" id="btnInvite" href="__INVITE__" target="_blank" rel="noopener">Invite Link</a>
          </div>
        </div>
        <div class="games-grid" style="margin-top:12px">
          <div class="game-card" id="openCrash" style="background-image: radial-gradient(600px 280px at 10% -10%, rgba(59,130,246,.25), transparent 60%);">
            <div class="title">🚀 Crash</div><div class="muted">Shared rounds • 10s betting • Live cashout</div>
          </div>
          <div class="game-card" id="openMines" style="background-image: radial-gradient(600px 280px at 85% -20%, rgba(34,197,94,.25), transparent 60%);">
            <div class="title">💣 Mines</div><div class="muted">5×5 board • Choose mines • Cash out anytime</div>
          </div>

          <!-- New stubs -->
          <div class="game-card" id="openCoinflip" style="background-image: radial-gradient(600px 280px at 50% -20%, rgba(250,204,21,.22), transparent 60%);">
            <div class="title">🪙 Coinflip</div><div class="muted">Quick 50/50 — coming soon</div>
          </div>
          <div class="game-card" id="openBlackjack" style="background-image: radial-gradient(600px 280px at 30% -10%, rgba(16,185,129,.22), transparent 60%);">
            <div class="title">🃏 Blackjack</div><div class="muted">Beat the dealer — coming soon</div>
          </div>
          <div class="game-card" id="openPump" style="background-image: radial-gradient(600px 280px at 70% -10%, rgba(147,51,234,.22), transparent 60%);">
            <div class="title">📈 Pump</div><div class="muted">Ride the spike — coming soon</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Crash -->
    <div id="page-crash" style="display:none">
      <div class="card">
        <div class="hero">
          <div style="display:flex;align-items:baseline;gap:10px"><div class="big" id="crNow">0.00×</div><div class="muted" id="crHint">Loading…</div></div>
          <button class="chip" id="backToGames">← Games</button>
        </div>
        <div class="cr-graph-wrap" style="margin-top:10px"><canvas id="crCanvas"></canvas></div>
        <div style="margin-top:12px"><div class="label" style="margin-bottom:4px">Previous Busts</div><div id="lastBusts" class="muted">Loading last rounds…</div></div>
        <div class="games-grid" style="grid-template-columns:1fr 1fr;gap:12px;margin-top:8px">
          <div class="field"><div class="label">Bet (DL)</div><input id="crBet" type="number" min="1" step="0.01" placeholder="min 1.00"/></div>
          <div class="field"><div class="label">Auto Cashout (×) — optional</div><input id="crCash" type="number" min="1.01" step="0.01" placeholder="e.g. 2.00"/></div>
        </div>
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:8px">
          <button class="btn primary" id="crPlace">Place Bet</button>
          <button class="btn ok" id="crCashout" style="display:none">💸 Cash Out</button>
          <span id="crMsg" class="muted"></span>
        </div>
        <div class="card soft" style="margin-top:14px"><div class="label">Your recent rounds</div><div id="crLast" class="muted">—</div></div>
      </div>
    </div>

    <!-- Mines (show only Cash Out while active) -->
    <div id="page-mines" style="display:none">
      <div class="card">
        <div class="hero"><div class="big">💣 Mines</div><button class="chip" id="backToGames2">← Games</button></div>
        <div class="grid-2" style="margin-top:12px">
          <div>
            <div id="mSetup">
              <div class="field"><div class="label">Bet (DL)</div><input id="mBet" type="number" min="1" step="0.01" placeholder="min 1.00"/></div>
              <div class="field" style="margin-top:10px"><div class="label">Mines (1–24)</div><input id="mMines" type="number" min="1" max="24" step="1" value="3"/></div>
              <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:12px">
                <button class="btn primary" id="mStart">Start Game</button>
                <span id="mMsg" class="muted"></span>
              </div>
            </div>

            <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:8px">
              <button class="btn ok" id="mCash" style="display:none">💸 Cash Out</button>
              <span class="pill" id="mMult">Multiplier: 1.0000×</span><span class="pill" id="mPotential">Potential: —</span>
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

    <!-- Coinflip / Blackjack / Pump placeholders -->
    <div id="page-coinflip" style="display:none"><div class="card"><div class="hero"><div class="big">🪙 Coinflip</div><button class="chip" id="backToGames_cf">← Games</button></div><div class="muted" style="margin-top:8px">Coming soon.</div></div></div>
    <div id="page-blackjack" style="display:none"><div class="card"><div class="hero"><div class="big">🃏 Blackjack</div><button class="chip" id="backToGames_bj">← Games</button></div><div class="muted" style="margin-top:8px">Coming soon.</div></div></div>
    <div id="page-pump" style="display:none"><div class="card"><div class="hero"><div class="big">📈 Pump</div><button class="chip" id="backToGames_pu">← Games</button></div><div class="muted" style="margin-top:8px">Coming soon.</div></div></div>

    <!-- Referral -->
    <div id="page-ref" style="display:none">
      <div class="card">
        <div class="hero">
          <div class="big">🙌 Referral Program</div>
          <div class="discord-cta">
            <button class="btn ghost" id="btnJoinDiscord2">Join Discord</button>
            <a class="chip" id="btnInvite2" href="__INVITE__" target="_blank" rel="noopener">Invite Link</a>
          </div>
        </div>
        <div class="sep"></div>
        <div class="grid-2">
          <div class="card soft">
            <div class="label">Your Referral Handle</div>
            <div class="row cols-2" style="margin-top:6px">
              <div class="field"><input id="refName" placeholder="choose-handle (3-20 chars)"/></div>
              <button class="btn primary" id="refSave">Save</button>
            </div>
            <div class="hint" id="refMsg" style="margin-top:6px"></div>
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
            <div class="hint" style="margin-top:6px">Clicks count when someone opens your link. Joins count when they sign in.</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Promo Codes -->
    <div id="page-promo" style="display:none">
      <div class="card">
        <div class="hero">
          <div class="big">🎁 Promo Codes</div>
          <div class="discord-cta">
            <button class="btn ghost" id="btnJoinDiscord3">Join Discord</button>
            <a class="chip" id="btnInvite3" href="__INVITE__" target="_blank" rel="noopener">Invite Link</a>
          </div>
        </div>
        <div class="sep"></div>
        <div class="grid-2">
          <div class="card soft">
            <div class="label">Redeem</div>
            <div class="row cols-2" style="margin-top:6px">
              <div class="field"><input id="promoInput" placeholder="e.g. WELCOME10"/></div>
              <button class="btn primary" id="redeemBtn">Redeem</button>
            </div>
            <div id="promoMsg" class="hint" style="margin-top:6px"></div>
          </div>
          <div class="card soft">
            <div class="label">Your Redemptions</div>
            <div id="myCodes" class="muted" style="margin-top:8px">—</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Leaderboard -->
    <div id="page-lb" style="display:none">
      <div class="card">
        <div class="hero"><div class="big">🏆 Leaderboard — Top Wagered</div><div class="countdown" id="lbCountdown">—</div></div>
        <div class="lb-controls" style="margin-top:10px">
          <div class="seg" id="lbSeg"><button data-period="daily" class="active">Daily</button><button data-period="monthly">Monthly</button><button data-period="alltime">All-time</button></div>
          <span class="hint">Anonymous players show as “Anonymous”. Amounts hidden for anonymous users.</span>
        </div>
        <div id="lbWrap" class="muted">Loading…</div>
      </div>
    </div>

    <!-- Settings (open via avatar; logout at bottom) -->
    <div id="page-settings" style="display:none">
      <div class="card">
        <div class="label">Settings</div>
        <div style="margin-top:8px">
          <label style="display:flex;align-items:center;gap:10px">
            <input type="checkbox" id="anonToggle" style="width:auto"/>
            <span><strong>Anonymous Mode</strong> — hide your name & wager amounts from others. You still show as “Anonymous”.</span>
          </label>
          <div class="hint">Takes effect immediately.</div>
          <div id="setMsg" class="muted" style="margin-top:8px"></div>
          <div class="sep"></div>
          <a class="btn ghost" href="/logout" style="margin-top:6px">Logout</a>
        </div>
      </div>
    </div>

    <!-- Profile (open by clicking avatar) -->
    <div id="page-profile" style="display:none">
      <div class="card">
        <div class="label">Profile</div><div id="profileBox">Loading…</div>
        <div id="ownerPanel" class="card soft" style="display:none;margin-top:12px">
          <div class="label">Owner / Admin Panel</div>
          <div class="row cols-4" style="margin-top:6px">
            <div class="field"><div class="label">Discord ID or &lt;@mention&gt;</div><input id="tIdent" placeholder="ID or <@id>"/></div>
            <div class="field"><div class="label">Amount (+/- DL)</div><input id="tAmt" type="text" placeholder="10 or -5.25"/></div>
            <div class="field"><div class="label">Reason (optional)</div><input id="tReason" placeholder="promo/correction/prize"/></div>
            <div style="align-self:end"><button class="btn primary" id="tApply">Apply</button></div>
          </div>
          <div id="tMsg" class="hint" style="margin-top:6px"></div>
          <div class="sep"></div>
          <div class="row cols-3">
            <div class="field"><div class="label">Target</div><input id="rIdent" placeholder="ID or <@id>"/></div>
            <button class="btn" id="rAdmin">Make ADMIN</button>
            <button class="btn" id="rMember">Make MEMBER</button>
          </div>
          <div id="rMsg" class="hint" style="margin-top:6px"></div>
          <div class="sep"></div>
          <div class="row cols-5">
            <div class="field"><div class="label">Target</div><input id="xIdent" placeholder="ID or <@id>"/></div>
            <div class="field"><div class="label">Seconds</div><input id="xSecs" type="number" value="600"/></div>
            <div class="field"><div class="label">Reason</div><input id="xReason" placeholder="spam / rude / etc"/></div>
            <button class="btn" id="xSite">Site Only</button><button class="btn" id="xBoth">Site + Discord</button>
          </div>
          <div id="xMsg" class="hint" style="margin-top:6px"></div>
          <div class="sep"></div>
          <div class="row cols-3">
            <div class="field"><div class="label">Code (optional)</div><input id="cCode" placeholder="auto-generate if empty"/></div>
            <div class="field"><div class="label">Amount (DL)</div><input id="cAmount" type="text" placeholder="e.g. 10 or 1.24"/></div>
            <div class="field"><div class="label">Max Uses</div><input id="cMax" type="number" placeholder="e.g. 100"/></div>
          </div>
          <div style="margin-top:8px"><button class="btn primary" id="cMake">Create</button> <span id="cMsg" class="hint"></span></div>
        </div>
      </div>
    </div>
  </div>

  <!-- Floating chat -->
  <button class="fab" id="fabChat" title="Open chat"><svg viewBox="0 0 24 24"><path d="M4 4h16v12H7l-3 3V4z"/></svg></button>
  <div class="chat-drawer" id="chatDrawer">
    <div class="chat-head"><div>Global Chat <span id="chatNote" class="muted"></span></div><button class="chip" id="chatClose">Close</button></div>
    <div class="chat-body" id="chatBody"></div>
    <div class="chat-input"><input id="chatText" placeholder="Say something…"/><button class="btn primary" id="chatSend">Send</button></div>
  </div>

  <script>
  const qs = id => document.getElementById(id);
  const j = async (url, init) => {
    const r = await fetch(url, init);
    if(!r.ok){
      let t = await r.text().catch(()=> '');
      try{ const js = JSON.parse(t); throw new Error(js.detail || js.message || t || r.statusText); }
      catch{ throw new Error(t || r.statusText); }
    }
    const ct = r.headers.get('content-type')||'';
    return ct.includes('application/json') ? r.json() : r.text();
  };
  const GEM = "💎"; const fmtDL = (n)=> `${GEM} ${(Number(n)||0).toFixed(2)} DL`;

  // Simple router
  const pages = ['page-games','page-crash','page-mines','page-coinflip','page-blackjack','page-pump','page-ref','page-promo','page-lb','page-settings','page-profile'];
  function showOnly(id){
    for(const p of pages){ const el = qs(p); if(el) el.style.display = (p===id) ? '' : 'none'; }
    // Tabs highlight (settings/profile not in tabs)
    const map = {'page-games':'tab-games','page-ref':'tab-ref','page-promo':'tab-promo','page-lb':'tab-lb'};
    for(const t of ['tab-games','tab-ref','tab-promo','tab-lb']){
      const el = qs(t); if(el) el.classList.toggle('active', map[id]===t);
    }
  }

  // Header / Auth (avatar opens menu with Profile/Settings)
  async function renderHeader(){
    try{
      const me = await j('/api/me');
      const bal = await j('/api/balance');
      qs('authArea').innerHTML = `
        <button class="btn primary" id="btnJoinSmall">${me.in_guild ? 'In Discord' : 'Join Discord'}</button>
        <span class="chip">Balance: <strong>${fmtDL(bal.balance)}</strong></span>
        <div class="avatar-wrap">
          <img class="avatar" id="avatarBtn" src="${me.avatar_url||''}" title="${me.username||'user'}"/>
          <div id="userMenu" class="menu">
            <div class="item" id="menuProfile">Profile</div>
            <div class="item" id="menuSettings">Settings</div>
          </div>
        </div>
      `;
      qs('btnJoinSmall').onclick = joinDiscord;
      const menu = qs('userMenu');
      const av = qs('avatarBtn');
      av.onclick = (e)=>{ e.stopPropagation(); menu.classList.toggle('open'); };
      document.body.addEventListener('click', ()=> menu.classList.remove('open'));
      qs('menuProfile').onclick = ()=>{ menu.classList.remove('open'); showOnly('page-profile'); renderProfile(); };
      qs('menuSettings').onclick = ()=>{ menu.classList.remove('open'); showOnly('page-settings'); loadSettings(); };
    }catch(_){
      qs('authArea').innerHTML = `<a class="btn primary" href="/login">Login with Discord</a>`;
    }
  }

  // Join Discord (buttons in multiple places)
  async function joinDiscord(){
    try{
      await j('/api/discord/join', { method:'POST' });
      alert('Joined the Discord server!');
      renderHeader();
    }catch(e){ alert(e.message || 'Could not join. Try relogin.'); }
  }
  for(const id of ['btnJoinDiscord','btnJoinDiscord2','btnJoinDiscord3']){
    const el = qs(id); if(el) el.onclick = joinDiscord;
  }
  for(const id of ['btnInvite','btnInvite2','btnInvite3']){
    const a = qs(id); if(a && a.getAttribute('href') === '__INVITE__'){ a.style.display='none'; }
  }

  // Tabs
  qs('homeLink').onclick = (e)=>{ e.preventDefault(); showOnly('page-games'); };
  qs('tab-games').onclick = ()=> showOnly('page-games');
  qs('tab-ref').onclick = ()=> { showOnly('page-ref'); loadReferral(); };
  qs('tab-promo').onclick = ()=> { showOnly('page-promo'); renderPromo(); };
  qs('tab-lb').onclick = ()=> { showOnly('page-lb'); refreshLeaderboard(); };

  // ---------- Referral ----------
  async function loadReferral(){
    try{
      const st = await j('/api/referral/state');
      if(st && st.name){ qs('refName').value = st.name; qs('refLink').value = location.origin + '/r/' + st.name; }
      qs('refClicks').textContent = st.clicks||0; qs('refJoins').textContent = st.joined||0;
    }catch(_){}
  }
  qs('refSave').onclick = async()=>{
    const name = qs('refName').value.trim();
    qs('refMsg').textContent = '';
    try{
      await j('/api/referral/set', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ name })});
      qs('refMsg').textContent = 'Saved.'; qs('refLink').value = location.origin + '/r/' + name.toLowerCase();
    }catch(e){ qs('refMsg').textContent = e.message; }
  };
  qs('copyRef').onclick = ()=>{
    const inp = qs('refLink'); inp.select(); inp.setSelectionRange(0, 99999); document.execCommand('copy');
  };

  // ---------- Promo ----------
  async function renderPromo(){
    try{
      const my = await j('/api/promo/my');
      qs('myCodes').innerHTML = (my.rows && my.rows.length)
        ? '<table><thead><tr><th>Code</th><th>Redeemed</th></tr></thead><tbody>' +
          my.rows.map(r=>`<tr><td>${r.code}</td><td>${new Date(r.redeemed_at).toLocaleString()}</td></tr>`).join('') +
          '</tbody></table>' : '—';
    }catch(_){}
  }
  qs('redeemBtn').onclick = async ()=>{
    const code = qs('promoInput').value.trim();
    qs('promoMsg').textContent = '';
    try{
      const r = await j('/api/promo/redeem', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ code }) });
      qs('promoMsg').textContent = 'Redeemed! New balance: ' + fmtDL(r.new_balance);
      renderHeader(); renderPromo();
    }catch(e){ qs('promoMsg').textContent = e.message; }
  };

  // ---------- Profile ----------
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
          qs('tMsg').textContent='';
          try{
            const r = await j('/api/admin/adjust',{ method:'POST', headers:{'Content-Type':'application/json'},
              body: JSON.stringify({ identifier: qs('tIdent').value.trim(), amount: qs('tAmt').value.trim(), reason: qs('tReason').value.trim() })});
            qs('tMsg').textContent = 'OK. New balance: ' + fmtDL(r.new_balance); renderHeader();
          }catch(e){ qs('tMsg').textContent = e.message; }
        };
        qs('rAdmin').onclick = ()=> setRole('admin');
        qs('rMember').onclick = ()=> setRole('member');
        async function setRole(role){
          qs('rMsg').textContent='';
          try{
            await j('/api/admin/role',{ method:'POST', headers:{'Content-Type':'application/json'},
              body: JSON.stringify({ identifier: qs('rIdent').value.trim(), role })});
            qs('rMsg').textContent = 'Role updated.';
          }catch(e){ qs('rMsg').textContent = e.message; }
        }
        qs('xSite').onclick = ()=> timeout('site');
        qs('xBoth').onclick = ()=> timeout('both');
        async function timeout(which){
          qs('xMsg').textContent='';
          try{
            const payload = { identifier: qs('xIdent').value.trim(), seconds: parseInt(qs('xSecs').value||'600',10), reason: qs('xReason').value.trim() };
            const url = which==='both' ? '/api/admin/timeout_both' : '/api/admin/timeout_site';
            await j(url, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
            qs('xMsg').textContent='Timeout set.';
          }catch(e){ qs('xMsg').textContent = e.message; }
        }
        qs('cMake').onclick = async ()=>{
          qs('cMsg').textContent='';
          try{
            const r = await j('/api/admin/promo/create',{ method:'POST', headers:{'Content-Type':'application/json'},
              body: JSON.stringify({ code: (qs('cCode').value||'').trim() || null, amount: qs('cAmount').value.trim(), max_uses: parseInt(qs('cMax').value||'1',10)||1 })});
            qs('cMsg').textContent = 'Created: ' + r.code;
          }catch(e){ qs('cMsg').textContent = e.message; }
        };
      }
    }catch(_){ qs('profileBox').textContent = '—'; }
  }

  // ---------- Settings ----------
  async function loadSettings(){
    qs('setMsg').textContent='';
    try{ const r = await j('/api/settings/get'); qs('anonToggle').checked = !!(r && r.is_anon); }catch(_){}
  }
  qs('anonToggle')?.addEventListener('change', async (e)=>{
    qs('setMsg').textContent='';
    try{
      const r = await j('/api/settings/set_anon', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ is_anon: !!e.target.checked }) });
      qs('setMsg').textContent = r && r.ok ? 'Saved.' : 'Updated.';
    }catch(err){ qs('setMsg').textContent = err.message; }
  });

  // ---------- Leaderboard ----------
  const lbWrap = ()=> qs('lbWrap'); let lbPeriod = 'daily', lbMe = null;
  function setLbButtons(){
    const btns = Array.from(qs('lbSeg').querySelectorAll('button'));
    btns.forEach(b=> b.classList.toggle('active', b.dataset.period===lbPeriod));
  }
  function nextUtcMidnight(){
    const now = new Date();
    return new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()+1, 0,0,0,0));
  }
  function endOfUtcMonth(){
    const now = new Date();
    return new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth()+1, 1, 0,0,0, 0,0)) }
  function fmtCountdown(ms){
    if(ms <= 0) return '—';
    const s = Math.floor(ms/1000);
    const hh = Math.floor(s/3600);
    const mm = Math.floor((s%3600)/60);
    const ss = s%60;
    const p = n=> String(n).padStart(2,'0');
    return `${p(hh)}:${p(mm)}:${p(ss)} until reset`;
  }
  async function refreshLeaderboard(){
    try{
      setLbButtons();
      const r = await j(`/api/leaderboard?period=${encodeURIComponent(lbPeriod)}&limit=50`);
      const rows = r.rows||[];
      lbWrap().innerHTML = rows.length ? (
        '<table><thead><tr><th>#</th><th>Name</th><th>Total Wagered</th></tr></thead><tbody>' +
        rows.map((row,i)=>{
          const meCls = (lbMe && String(row.user_id)===String(lbMe)) ? ' class="me-row"' : '';
          const anonCls = row.is_anon ? ' class="name anon"' : ' class="name"';
          const name = row.is_anon ? 'Anonymous' : (row.display_name || 'User');
          const amt = row.is_anon ? '—' : fmtDL(row.total_wagered||0);
          return `<tr${meCls}><td>${i+1}</td><td${anonCls}>${name}</td><td>${amt}</td></tr>`;
        }).join('') + '</tbody></table>'
      ) : '—';
      // countdown
      const now = new Date();
      let end = null;
      if(lbPeriod==='daily') end = nextUtcMidnight();
      else if(lbPeriod==='monthly') end = endOfUtcMonth();
      qs('lbCountdown').textContent = end ? fmtCountdown(end - now) : '—';
    }catch(_){
      lbWrap().textContent = '—';
    }
  }
  // periodic countdown tick
  setInterval(()=>{
    const now = new Date();
    let end = null;
    if(lbPeriod==='daily') end = nextUtcMidnight();
    else if(lbPeriod==='monthly') end = endOfUtcMonth();
    qs('lbCountdown').textContent = end ? fmtCountdown(end - now) : '—';
  }, 1000);
  // buttons
  Array.from(qs('lbSeg').querySelectorAll('button')).forEach(b=>{
    b.onclick = ()=>{ lbPeriod = b.dataset.period; refreshLeaderboard(); };
  });

  // ---------- Crash ----------
  let crPoll = null, crAnim = null, crLastState = null;
  const crCanvas = ()=> qs('crCanvas');
  function drawCrash(mult){
    const cv = crCanvas(); if(!cv) return;
    const dpr = window.devicePixelRatio||1;
    const w = cv.clientWidth, h = cv.clientHeight;
    if(cv.width !== Math.floor(w*dpr)) cv.width = Math.floor(w*dpr);
    if(cv.height !== Math.floor(h*dpr)) cv.height = Math.floor(h*dpr);
    const ctx = cv.getContext('2d'); ctx.save(); ctx.scale(dpr,dpr);
    // clear bg
    ctx.clearRect(0,0,w,h);
    // axes-ish grid
    ctx.globalAlpha = 0.15;
    for(let i=0;i<=10;i++){
      const x = (w/10)*i; ctx.fillRect(x,0,1,h);
      const y = (h/10)*i; ctx.fillRect(0,y,w,1);
    }
    ctx.globalAlpha = 1.0;
    // progress path: map multiplier 1.00x.. to width
    const maxMult = Math.max(2, Math.min(20, Math.ceil(Math.max(2, mult))));
    const xFrac = Math.min(1, (mult-1)/(maxMult-1));
    const xEnd = 12 + xFrac*(w-24);
    // curve
    ctx.beginPath(); ctx.moveTo(12,h-18);
    const steps = 120, m0 = Math.max(1, mult);
    for(let i=1;i<=steps;i++){
      const m = 1 + (i/steps)*(m0-1);
      const xf = Math.min(1,(m-1)/(maxMult-1));
      const x = 12 + xf*(w-24);
      const yf = Math.min(0.92, Math.log(m)/Math.log(maxMult)); // compress
      const y = (h-18) - yf*(h-40);
      ctx.lineTo(x,y);
    }
    ctx.strokeStyle = 'rgba(255,255,255,0.9)';
    ctx.lineWidth = 2; ctx.stroke();
    // current dot
    ctx.beginPath(); ctx.arc(xEnd, (h-18)-Math.min(0.92, Math.log(m0)/Math.log(maxMult))*(h-40), 4, 0, Math.PI*2);
    ctx.fillStyle = 'rgba(255,255,255,0.95)'; ctx.fill();
    ctx.restore();
  }
  function setCrashHint(t){ qs('crHint').textContent = t||''; }
  function setCrashNow(x){ qs('crNow').textContent = `${Number(x||0).toFixed(2)}×`; }
  function setLastBusts(arr){
    const el = qs('lastBusts');
    if(!arr || !arr.length){ el.textContent = '—'; return; }
    el.innerHTML = arr.map(v=>{
      const bust = Number(v||0);
      const good = bust>=2.0 ? 'good' : (bust<=1.1 ? 'bad':'');
      return `<span class="${good}" style="display:inline-block;margin-right:8px;font-weight:800">${bust.toFixed(2)}×</span>`;
    }).join('');
  }
  function setCrashButtons(state){
    const you = state.your_bet;
    const place = qs('crPlace'), cash = qs('crCashout'), msg = qs('crMsg');
    cash.style.display = (state.phase==='running' && you && !you.cashed_out) ? '' : 'none';
    place.disabled = (state.phase!=='betting');
    msg.textContent = you
      ? (you.cashed_out ? `You cashed at ${Number(you.cashed_out).toFixed(2)}×` : `Your bet: ${fmtDL(you.bet)} @ ${Number(you.cashout||0).toFixed(2)}×`)
      : '';
  }
  async function pollCrash(){
    try{
      const st = await j('/api/crash/state');
      crLastState = st;
      setLastBusts(st.last_busts||[]);
      setCrashButtons(st);
      if(st.phase==='betting'){
        const ends = new Date(st.betting_ends_at||Date.now());
        const left = Math.max(0, Math.floor((ends - new Date())/1000));
        setCrashHint(`Betting… ${left}s`);
        setCrashNow(1.00);
        drawCrash(1.00);
      }else if(st.phase==='running'){
        const m = Number(st.current_multiplier||1);
        setCrashHint('Live!');
        setCrashNow(m);
        drawCrash(m);
      }else{
        setCrashHint('Ending…');
      }
      // recent history for you
      try{
        const h = await j('/api/crash/history');
        qs('crLast').innerHTML = (h.rows && h.rows.length)
          ? h.rows.map(r=>{
              const w = Number(r.win||0);
              const bust = Number(r.bust||0).toFixed(2);
              const mult = Number(r.cashout||0).toFixed(2);
              const cls = w>0 ? 'good' : 'bad';
              return `<div>Bet ${fmtDL(r.bet)} • Cashout ${mult}× • Bust ${bust}× • <strong class="${cls}">${fmtDL(w)}</strong></div>`;
            }).join('') : '—';
      }catch(_){}
    }catch(e){
      setCrashHint('Error loading crash.');
    }
  }
  function startCrash(){
    stopCrash();
    pollCrash();
    crPoll = setInterval(pollCrash, 1000);
  }
  function stopCrash(){
    if(crPoll){ clearInterval(crPoll); crPoll=null; }
    if(crAnim){ cancelAnimationFrame(crAnim); crAnim=null; }
  }
  qs('crPlace').onclick = async ()=>{
    const bet = qs('crBet').value.trim();
    const cash = qs('crCash').value.trim();
    qs('crMsg').textContent='';
    try{
      const payload = { bet, cashout: cash ? Number(cash) : undefined };
      const r = await j('/api/crash/place', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
      qs('crMsg').textContent = r && r.ok ? 'Bet placed!' : (r.message||'Placed.');
      pollCrash();
    }catch(e){ qs('crMsg').textContent = e.message; }
  };
  qs('crCashout').onclick = async ()=>{
    qs('crMsg').textContent='';
    try{
      const r = await j('/api/crash/cashout', { method:'POST' });
      qs('crMsg').textContent = r && r.ok ? 'Cashed out!' : (r.message||'Cashed!');
      renderHeader(); pollCrash();
    }catch(e){ qs('crMsg').textContent = e.message; }
  };

  // ---------- Mines ----------
  const M_SIZE = 25;
  function buildMinesGrid(){
    const g = qs('mGrid'); g.innerHTML='';
    for(let i=0;i<M_SIZE;i++){
      const b = document.createElement('button');
      b.className='btn'; b.style.width='64px'; b.style.height='64px'; b.textContent = String(i+1);
      b.dataset.index = String(i);
      b.onclick = ()=> pickMine(i);
      g.appendChild(b);
    }
  }
  function updateMinesUI(st){
    const msg = qs('mMsg'), mult = qs('mMult'), pot = qs('mPotential');
    const picks = qs('mPicks'), bombs = qs('mBombs'), hash = qs('mHash'), status = qs('mStatus');
    const cash = qs('mCash'), setup = qs('mSetup');
    if(!st || st.status==='idle' || st.status==='none'){
      msg.textContent=''; mult.textContent='Multiplier: 1.0000×'; pot.textContent='Potential: —';
      picks.textContent='Picks: 0'; bombs.textContent='Mines: 3'; hash.textContent='Commit: —'; status.textContent='Status: —';
      cash.style.display='none'; setup.style.display='';
      Array.from(qs('mGrid').children).forEach(b=>{ b.disabled=true; b.textContent=b.textContent.replace('💣','').replace('✅',''); });
      return;
    }
    // Active or finished
    setup.style.display = (st.status==='active') ? 'none' : '';
    cash.style.display = (st.status==='active') ? '' : 'none';
    mult.textContent = `Multiplier: ${(Number(st.multiplier||1)).toFixed(4)}×`;
    pot.textContent = `Potential: ${st.potential!=null ? fmtDL(st.potential) : '—'}`;
    picks.textContent = `Picks: ${st.picks||0}`;
    bombs.textContent = `Mines: ${st.mines||3}`;
    hash.textContent = `Commit: ${st.commit_hash || st.commit || '—'}`;
    status.textContent = `Status: ${st.status}`;
    // board reveals if provided
    if(st.reveals && Array.isArray(st.reveals)){
      st.reveals.forEach((v,i)=>{
        const b = qs('mGrid').children[i]; if(!b) return;
        b.disabled = st.status!=='active' || v!==null;
        if(v===true) b.textContent = '✅';
        else if(v===false) b.textContent = '💣';
      });
    }else{
      // enable buttons during active
      Array.from(qs('mGrid').children).forEach(b=>{ b.disabled = st.status!=='active'; });
    }
  }
  async function refreshMines(){
    try{
      const st = await j('/api/mines/state');
      updateMinesUI(st);
    }catch(e){
      // ignore
    }
  }
  qs('mStart').onclick = async ()=>{
    const bet = qs('mBet').value.trim();
    const mines = parseInt(qs('mMines').value||'3',10);
    qs('mMsg').textContent='';
    try{
      const r = await j('/api/mines/start', { method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ bet, mines }) });
      qs('mMsg').textContent = r && r.ok ? 'Game started!' : (r.message||'Started.');
      buildMinesGrid();
      refreshMines();
      renderMinesHistory();
      renderHeader();
    }catch(e){ qs('mMsg').textContent = e.message; }
  };
  async function pickMine(i){
    try{
      await j(`/api/mines/pick?index=${i}`, { method:'POST' });
      refreshMines();
      renderHeader();
    }catch(e){ qs('mMsg').textContent = e.message; }
  }
  qs('mCash').onclick = async ()=>{
    try{
      const r = await j('/api/mines/cashout', { method:'POST' });
      qs('mMsg').textContent = r && r.ok ? 'Cashed out!' : (r.message||'Cashed!');
      refreshMines();
      renderMinesHistory();
      renderHeader();
    }catch(e){ qs('mMsg').textContent = e.message; }
  };
  async function renderMinesHistory(){
    try{
      const h = await j('/api/mines/history');
      qs('mHist').innerHTML = (h.rows && h.rows.length)
        ? h.rows.map(r=>{
            const cls = Number(r.win||0)>0 ? 'good':'bad';
            return `<div>${new Date(r.created_at||r.ended_at||Date.now()).toLocaleString()} • Bet ${fmtDL(r.bet)} • Mines ${r.mines} • <strong class="${cls}">${fmtDL(r.win||0)}</strong></div>`;
          }).join('') : '—';
    }catch(_){ qs('mHist').textContent = '—'; }
  }

  // ---------- Chat ----------
  let chatOpen = false, chatSince = 0, chatTimer = null;
  function escapeHtml(s){ return (s||'').replace(/[&<>"']/g, m=> ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m])); }
  function roleBadge(r){ return `<span class="badge ${r||'member'}">${(r||'member').toUpperCase()}</span>`; }
  function addChatRows(rows){
    const body = qs('chatBody');
    for(const m of rows){
      chatSince = Math.max(chatSince, m.id||0);
      const who = `<span class="user-link" data-uid="${m.user_id||''}">${escapeHtml(m.username||'user')}</span>`;
      const lvl = `<span class="level">Lv ${m.level||1}</span>`;
      const rb = roleBadge(m.role||'member');
      const time = new Date(m.created_at||Date.now()).toLocaleTimeString();
      const priv = m.private_to ? `<span class="badge">DM</span>` : '';
      const row = document.createElement('div'); row.className='msg';
      row.innerHTML = `<div class="msghead">${who} ${lvl} ${rb} ${priv}<span class="time">${time}</span></div><div>${escapeHtml(m.text||'')}</div>`;
      body.appendChild(row);
    }
    body.scrollTop = body.scrollHeight;
  }
  async function fetchChat(){
    try{
      const r = await j(`/api/chat/fetch?since=${chatSince}&limit=50`);
      if(r && r.rows && r.rows.length) addChatRows(r.rows);
    }catch(_){}
  }
  function openChat(){
    if(chatOpen) return; chatOpen=true;
    qs('chatDrawer').classList.add('open');
    qs('chatBody').innerHTML='';
    chatSince = 0;
    fetchChat();
    chatTimer = setInterval(fetchChat, 1500);
    qs('chatText').focus();
  }
  function closeChat(){
    chatOpen=false;
    qs('chatDrawer').classList.remove('open');
    if(chatTimer){ clearInterval(chatTimer); chatTimer=null; }
  }
  qs('fabChat').onclick = openChat;
  qs('chatClose').onclick = closeChat;
  qs('chatSend').onclick = async ()=>{
    const txt = qs('chatText').value.trim();
    if(!txt) return;
    try{
      await j('/api/chat/send', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ text: txt })});
      qs('chatText').value=''; fetchChat();
    }catch(e){ alert(e.message||'Could not send'); }
  };
  qs('chatText').addEventListener('keydown', (e)=>{ if(e.key==='Enter'){ e.preventDefault(); qs('chatSend').click(); } });

  // ---------- Simple navigation wiring ----------
  function bindNav(){
    const map = [
      ['openCrash', ()=>{ showOnly('page-crash'); startCrash(); }],
      ['openMines', ()=>{ showOnly('page-mines'); buildMinesGrid(); refreshMines(); renderMinesHistory(); }],
      ['openCoinflip', ()=> showOnly('page-coinflip')],
      ['openBlackjack', ()=> showOnly('page-blackjack')],
      ['openPump', ()=> showOnly('page-pump')],
      ['backToGames', ()=>{ showOnly('page-games'); stopCrash(); }],
      ['backToGames2', ()=>{ showOnly('page-games'); }],
      ['backToGames_cf', ()=> showOnly('page-games')],
      ['backToGames_bj', ()=> showOnly('page-games')],
      ['backToGames_pu', ()=> showOnly('page-games')],
    ];
    for(const [id,fn] of map){ const el=qs(id); if(el) el.onclick = fn; }
  }

  // ---------- Referral attach from cookie ----------
  function getCookie(name){
    const m = document.cookie.match(new RegExp('(?:^|; )'+name.replace(/([$?*|{}\]\\^])/g,'\\$1')+'=([^;]*)'));
    return m ? decodeURIComponent(m[1]) : null;
  }
  async function attachReferralIfAny(){
    const rn = getCookie('refname');
    if(!rn) return;
    try{ await j(`/api/referral/attach?refname=${encodeURIComponent(rn)}`); }
    catch(_){}
    // expire quickly
    document.cookie = `refname=; Max-Age=0; path=/; samesite=lax`;
  }

  // ---------- Init ----------
  window.addEventListener('load', async ()=>{
    showOnly('page-games');
    bindNav();
    await renderHeader();
    attachReferralIfAny();
    // Try to detect my user id for lb highlight
    try{ const me = await j('/api/me'); lbMe = String(me.id||''); }catch(_){}
    refreshLeaderboard();
  });
  </script>
</body>
</html>
"""

# ---------- Root ----------
@app.get("/", response_class=HTMLResponse)
async def root():
    html = HTML_TEMPLATE.replace("__INVITE__", DISCORD_INVITE or "__INVITE__")
    html = html.replace("__OWNER_ID__", str(OWNER_ID))
    return HTMLResponse(html)

# ---------- Run ----------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=bool(os.getenv("RELOAD","1")=="1"))
