# app/main.py

import os, json, asyncio, re, random, string, datetime, base64
from urllib.parse import urlencode, urlparse
from typing import Optional, Tuple, Dict, List
from decimal import Decimal, ROUND_DOWN, getcontext
from contextlib import asynccontextmanager

import httpx
import psycopg
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from itsdangerous import URLSafeSerializer, BadSignature
from pydantic import BaseModel

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
GUILD_ID = int(os.getenv("GUILD_ID", "0"))
DISCORD_INVITE = os.getenv("DISCORD_INVITE", "")
REFERRAL_SHARE_BASE = os.getenv("REFERRAL_SHARE_BASE", "https://growcb.new/referral")

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

# ---------- App / Lifespan / Static ----------
def _get_static_dir():
    base = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(base, "static")
    try: os.makedirs(static_dir, exist_ok=True)
    except Exception: pass
    return static_dir

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    apply_migrations()
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=_get_static_dir()), name="static")

# Serve images next to main.py or /static
_TRANSPARENT_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
)
@app.get("/img/{filename}")
async def serve_img(filename: str):
    base = os.path.dirname(os.path.abspath(__file__))
    p1 = os.path.join(base, filename)
    p2 = os.path.join(base, "static", filename)
    for p in (p1, p2):
        if os.path.isfile(p): return FileResponse(p)
    return Response(content=_TRANSPARENT_PNG, media_type="image/png")

# ---------- Sessions (sticky cookie) ----------
SER = URLSafeSerializer(SECRET_KEY, salt="session-v1")

def _cookie_domain_from_request(request: Optional[Request]) -> Optional[str]:
    # Set cookie for eTLD+1 when possible; otherwise leave None
    try:
        if not request: return None
        host = request.headers.get("host") or urlparse(str(request.url)).hostname
        if not host: return None
        # strip port
        host = host.split(":")[0]
        # If it's localhost or an IP, skip domain= to avoid invalid cookies
        if re.match(r"^\d+\.\d+\.\d+\.\d+$", host) or host in ("localhost","127.0.0.1"):
            return None
        parts = host.split(".")
        if len(parts) >= 2:
            return "." + ".".join(parts[-2:])  # e.g. .growcb.net
    except Exception:
        return None
    return None

def _set_session(resp, data: dict, request: Optional[Request] = None):
    cookie_val = SER.dumps(data)
    domain = _cookie_domain_from_request(request)
    resp.set_cookie(
        "session",
        cookie_val,
        max_age=30*86400,
        httponly=True,
        samesite="lax",
        secure=True,       # works on HTTPS; ignore if local http (browser may still set)
        path="/",
        domain=domain
    )

def _clear_session(resp, request: Optional[Request] = None):
    domain = _cookie_domain_from_request(request)
    resp.delete_cookie("session", path="/", domain=domain)

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
    # balance_log
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

    # profiles (+ roles + referral)
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
    # oauth tokens
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
    # transfers for deposit/withdraw
    cur.execute("""
        CREATE TABLE IF NOT EXISTS transfers (
            id BIGSERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            ttype TEXT NOT NULL,
            amount NUMERIC(18,2),
            world TEXT,
            grow_id TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
    """)

@with_conn
def apply_migrations(cur):
    cur.execute("ALTER TABLE profiles ADD COLUMN IF NOT EXISTS is_anon BOOLEAN NOT NULL DEFAULT FALSE")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_crash_games_created_at ON crash_games (created_at)")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_mines_games_started_at ON mines_games (started_at)")
    cur.execute("ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS private_to TEXT")
    cur.execute("ALTER TABLE profiles ALTER COLUMN role SET DEFAULT 'member'")

# ---- balances / profiles
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
    cur.execute("SELECT display_name FROM profiles WHERE user_id=%s", (user_id,))
    r = cur.fetchone(); return r[0] if r else None

@with_conn
def set_profile_is_anon(cur, user_id: str, is_anon: bool):
    ensure_profile_row(user_id)
    cur.execute("UPDATE profiles SET is_anon=%s WHERE user_id=%s", (bool(is_anon), user_id))
    return {"ok": True, "is_anon": bool(is_anon)}

@with_conn
def profile_info(cur, user_id: str):
    ensure_profile_row(user_id)
    cur.execute("SELECT display_name, xp, role, is_anon FROM profiles WHERE user_id=%s", (user_id,))
    name, xp, role, is_anon = cur.fetchone()
    level = 1 + int(xp) // 100
    base = (level - 1) * 100; need = level * 100 - base
    progress = int(xp) - base; pct = 0 if need==0 else int(progress*100/need)
    bal = get_balance(user_id)
    return {
        "id": str(user_id), "name": name, "xp": int(xp), "level": level,
        "progress": progress, "next_needed": need, "progress_pct": pct,
        "balance": float(bal), "role": role, "is_anon": bool(is_anon)
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
    if role not in ("member","media","moderator","admin","owner"):
        raise ValueError("Invalid role")
    cur.execute("UPDATE profiles SET role=%s WHERE user_id=%s", (role, target_id))
    return {"ok": True, "role": role}

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
async def callback(request: Request, code: str):
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
    avatar_url = f"https://cdn.discordapp.com/avatars/{user_id}/{avatar_hash}.png?size=128" if avatar_hash else "https://cdn.discordapp.com/embed/avatars/0.png"

    ensure_profile_row(user_id)
    save_tokens(user_id, tok.get("access_token",""), tok.get("refresh_token"), tok.get("expires_in"))

    resp = RedirectResponse("/")
    _set_session(resp, {"id": user_id, "username": username, "avatar_url": avatar_url}, request)
    return resp

@app.get("/logout")
async def logout(request: Request):
    resp = RedirectResponse("/")
    _clear_session(resp, request)
    return resp

# ---------- Token store & guild join ----------
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

# ---------- Me / Balance / Public profile ----------
@app.get("/api/me")
async def api_me(request: Request):
    # If cookie missing, 401 is fine; UI falls back to Login button
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

@app.get("/api/profile/public")
async def api_profile_public(user_id: str):
    prof = public_profile(user_id)
    if not prof: raise HTTPException(404, "User not found")
    return prof

# ---------- Settings (Anon) ----------
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

NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{3,20}$")

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
class PromoIn(BaseModel):
    code: str

@app.get("/api/promo/my")
async def api_promo_my(request: Request):
    s = _require_session(request)
    with psycopg.connect(DATABASE_URL) as con, con.cursor() as cur:
        cur.execute("SELECT code, redeemed_at FROM promo_redemptions WHERE user_id=%s ORDER BY redeemed_at DESC LIMIT 50", (s["id"],))
        rows = [{"code": r[0], "redeemed_at": str(r[1])} for r in cur.fetchall()]
    return {"rows": rows}

@app.post("/api/promo/redeem")
async def api_promo_redeem(request: Request, body: PromoIn):
    s = _require_session(request)
    try:
        bal = redeem_promo(s["id"], body.code)
        return {"ok": True, "new_balance": float(bal)}
    except (PromoInvalid, PromoExpired, PromoExhausted, PromoAlreadyRedeemed) as e:
        raise HTTPException(400, str(e))

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

# ---------- Transfers (deposit/withdraw) ----------
class TransferIn(BaseModel):
    ttype: str           # 'deposit' or 'withdraw'
    amount: Optional[str] = None
    world: Optional[str] = None
    grow_id: Optional[str] = None

@with_conn
def create_transfer(cur, user_id: str, ttype: str, amount: Optional[str], world: Optional[str], grow_id: Optional[str]):
    ttype = (ttype or "").lower().strip()
    if ttype not in ("deposit","withdraw"):
        raise ValueError("Invalid type")
    amt = q2(D(amount or "0")) if amount else None
    cur.execute("""
        INSERT INTO transfers(user_id, ttype, amount, world, grow_id, status)
        VALUES (%s,%s,%s,%s,%s,'pending') RETURNING id
    """, (user_id, ttype, amt, world, (grow_id or "").strip() or None))
    return {"ok": True, "id": int(cur.fetchone()[0])}

@app.post("/api/transfer/create")
async def api_transfer_create(request: Request, body: TransferIn):
    s = _require_session(request)
    try:
        r = create_transfer(s["id"], body.ttype, body.amount, body.world, body.grow_id)
        return r
    except Exception as e:
        raise HTTPException(400, str(e))

# ---------- Chat ----------
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
    prof = profile_info(s["id"])
    if prof["level"] < 5:
        raise HTTPException(403, "You need to be level 5 to chat.")
    return chat_insert(s["id"], s["username"], body.text, None)

@app.get("/api/chat/fetch")
async def api_chat_fetch(request: Request, since: int = 0, limit: int = 30):
    uid = None
    try: uid = _require_session(request)["id"]
    except: pass
    rows = chat_fetch(since, limit, uid)
    return {"rows": rows}

@app.post("/api/chat/delete")
async def api_chat_del(request: Request, id: int):
    s = _require_session(request)
    role = get_role(s["id"])
    if role not in ("admin","owner"): raise HTTPException(403, "No permission")
    return chat_delete(id)

# ---------- Admin (minimal) ----------
class AdjustIn(BaseModel):
    identifier: str  # id or mention or handle
    amount: str
    reason: Optional[str] = None

def _id_from_identifier(identifier: str) -> str:
    m = re.search(r"\d{5,}", identifier or "")
    if m: return m.group(0)
    with psycopg.connect(DATABASE_URL) as con, con.cursor() as cur:
        cur.execute("SELECT user_id FROM profiles WHERE name_lower=%s", (identifier.lower(),))
        r = cur.fetchone()
        if r: return str(r[0])
    raise HTTPException(400, "Provide a numeric Discord ID, mention, or exact handle")

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

class AnnounceIn(BaseModel):
    text: str

@app.post("/api/admin/announce")
async def api_admin_announce(request: Request, body: AnnounceIn):
    s = _require_session(request)
    role = get_role(s["id"])
    if role not in ("admin","owner","moderator","media"): raise HTTPException(403, "No permission")
    return chat_insert(s["id"], s["username"], f"[Announcement] {body.text}", None)

# ---------- Discord Join ----------
@app.post("/api/discord/join")
async def api_discord_join(request: Request):
    s = _require_session(request)
    nick = get_profile_name(s["id"]) or s["username"]
    return await guild_add_member(s["id"], nickname=nick)

# ---------- Crash ----------
class CrashBetIn(BaseModel):
    bet: str
    cashout: Optional[float] = None

@app.get("/api/crash/state")
async def api_crash_state(request: Request):
    try:
        s = _require_session(request); uid = s["id"]
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

@app.get("/api/crash/history")
async def api_crash_history(request: Request):
    s = _require_session(request)
    return {"rows": your_history(s["id"], 10)}

# ---------- Mines ----------
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

# (HTML template, SPA routes, bot, and runner are in Part 2)
# ---------- HTML (UI/UX) ----------
HTML_TEMPLATE = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"/>
<title>GROWCB</title>

<!-- Basic SEO / Icons -->
<link rel="icon" href="/img/GrowCBnobackground.png" type="image/png"/>
<link rel="apple-touch-icon" href="/img/GrowCBnobackground.png"/>
<meta name="theme-color" content="#0a0f1e"/>
<meta property="og:title" content="GROWCB"/>
<meta property="og:description" content="Crash, Mines and more — play with friends."/>
<meta property="og:image" content="/img/GrowCBnobackground.png"/>

<style>
:root{
  --bg:#0a0f1e;--bg2:#0c1428;--card:#111a31;--muted:#9eb3da;--text:#ecf2ff;--accent:#6aa6ff;--accent2:#22c1dc;
  --ok:#34d399;--warn:#f59e0b;--err:#ef4444;--border:#1f2b47;--chatW:360px;--input-bg:#0b1430;--input-br:#223457;--input-tx:#e6eeff;--input-ph:#9db4e4
}
*{box-sizing:border-box}html,body{height:100%}
body{margin:0;color:var(--text);background:radial-gradient(1400px 600px at 20% -10%, #11204d 0%, transparent 60%),linear-gradient(180deg,#0a0f1e,#0a0f1e 60%, #0b1124);font-family:Inter,system-ui,Segoe UI,Roboto,Arial,Helvetica,sans-serif}
a{color:inherit;text-decoration:none}
.container{max-width:1120px;margin:0 auto;padding:16px}

/* Preloader */
#preload{position:fixed;inset:0;background:#0a0f1e;display:grid;place-items:center;z-index:9999;transition:opacity .25s ease}
#preload.hide{opacity:0;pointer-events:none}
.loader{width:64px;height:64px;border-radius:50%;border:6px solid rgba(255,255,255,.12);border-top-color:#3b82f6;animation:spin 1s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

/* Inputs & buttons */
input,select,textarea{width:100%;appearance:none;background:var(--input-bg);color:var(--input-tx);border:1px solid var(--input-br);border-radius:12px;padding:10px 12px;outline:none}
.field{display:flex;flex-direction:column;gap:6px}
.btn{display:inline-flex;align-items:center;gap:10px;padding:10px 14px;border-radius:12px;border:1px solid var(--border);cursor:pointer;font-weight:700;user-select:none}
.btn.primary{background:linear-gradient(135deg,#3b82f6,#22c1dc);border-color:transparent;color:#041018;box-shadow:0 12px 24px rgba(59,130,246,.25)}
.btn.ghost{background:linear-gradient(180deg,#0e1833,#0b1326);border:1px solid var(--border)}
.btn.alt{background:linear-gradient(135deg,#22c55e,#16a34a);border-color:transparent;color:#041018}
.btn.gray{background:#0e1833;border:1px solid var(--border);color:#eaf2ff}

/* Toasts bottom-right */
#toasts{position:fixed;right:16px;bottom:16px;display:flex;flex-direction:column;gap:10px;z-index:10000}
.toast{min-width:240px;max-width:360px;background:#0e1833;border:1px solid var(--border);border-left:5px solid #3b82f6;border-radius:12px;padding:10px 12px;box-shadow:0 10px 20px rgba(0,0,0,.25)}
.toast.err{border-left-color:#ef4444}
.toast.ok{border-left-color:#22c55e}
.toast .t{font-weight:800;margin-bottom:4px}
.toast .m{color:var(--muted);font-size:13px}

/* Cards / layout */
.card{background:linear-gradient(180deg,#0f1a33,#0b1326);border:1px solid var(--border);border-radius:16px;padding:16px}
.header{position:sticky;top:0;z-index:60;backdrop-filter:blur(8px);background:rgba(10,15,30,.72);border-bottom:1px solid var(--border)}
.header-inner{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:10px 12px}

.left{display:flex;align-items:center;gap:12px;flex:1;min-width:0}
.brand{display:flex;align-items:center;gap:12px;font-weight:900;white-space:nowrap}
.brand .logo{width:56px;height:56px;border-radius:14px;object-fit:contain;border:1px solid var(--border);background:linear-gradient(135deg,var(--accent),var(--accent2))}
.brand .name{font-size:22px;letter-spacing:.3px}

.tabs{display:flex;gap:4px;align-items:center;padding:4px;border-radius:14px;background:linear-gradient(180deg,#0f1a33,#0b1326);border:1px solid var(--border)}
.tab{padding:8px 12px;border-radius:10px;cursor:pointer;font-weight:700;white-space:nowrap;color:#d8e6ff;opacity:.85;transition:all .15s ease;display:flex;align-items:center;gap:8px}
.tab:hover{opacity:1;transform:translateY(-1px)}
.tab.active{background:linear-gradient(135deg,#3b82f6,#22c1dc);color:#051326;box-shadow:0 6px 16px rgba(59,130,246,.25);opacity:1}

.right{display:flex;gap:8px;align-items:center;margin-left:12px}
.avatar{width:34px;height:34px;border-radius:50%;object-fit:cover;border:1px solid var(--border);cursor:pointer}
.avatar-wrap{position:relative}
.menu{position:absolute;right:0;top:40px;background:#0c1631;border:1px solid var(--border);border-radius:12px;padding:6px;display:none;min-width:180px;z-index:70}
.menu.open{display:block}
.menu .item{padding:8px 10px;border-radius:8px;cursor:pointer;font-size:14px}
.menu .item:hover{background:#11234a}

.balance-chip{display:inline-flex;align-items:center;gap:6px;background:#0c1631;border:1px solid var(--border);border-radius:999px;padding:6px 10px}
.balance-chip .dl-num{display:inline-flex;align-items:baseline;font-weight:900}
.dl-int{font-size:18px}
.dl-dec{font-size:12px;opacity:.85;margin-left:1px}
.dl-icon{width:18px;height:18px;vertical-align:middle;object-fit:contain}

.games-grid{display:grid;gap:14px;grid-template-columns:1fr}
@media(min-width:700px){.games-grid{grid-template-columns:1fr 1fr}}
@media(min-width:1020px){.games-grid{grid-template-columns:1fr 1fr 1fr}}

/* Game cards: image-only, compact */
.game-card{border:1px solid var(--border);border-radius:16px;overflow:hidden;cursor:pointer;background:#0b1326;padding:0}
.game-card .banner{display:block;width:100%;height:140px;object-fit:cover}

.hero{display:flex;justify-content:space-between;align-items:center;gap:14px;flex-wrap:wrap}
.sep{height:1px;background:rgba(255,255,255,.06);margin:10px 0}

/* Crash graph */
.cr-graph-wrap{position:relative;height:240px;background:#0e1833;border:1px solid var(--border);border-radius:16px;overflow:hidden}
canvas{display:block;width:100%;height:100%}

/* Chat */
.chat-drawer{position:fixed;right:0;top:64px;bottom:0;width:var(--chatW);max-width:92vw;transform:translateX(100%);transition:transform .2s ease-out;background:linear-gradient(180deg,#0f1a33,#0b1326);border-left:1px solid var(--border);display:flex;flex-direction:column;z-index:55}
.chat-drawer.open{transform:translateX(0)}
.chat-head{display:flex;align-items:center;justify-content:space-between;padding:10px 12px;border-bottom:1px solid var(--border)}
.chat-body{flex:1;overflow:auto;padding:10px 12px}
.chat-input{display:flex;gap:8px;padding:10px 12px;border-top:1px solid var(--border)}.chat-input input{flex:1}
.msg{margin-bottom:12px;padding-bottom:8px;border-bottom:1px dashed rgba(255,255,255,.04);position:relative}
.msghead{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.msghead .time{margin-left:auto;color:#9eb3da;font-size:12px}

/* Role badges */
.badge{font-size:10px;padding:3px 7px;border-radius:999px;border:1px solid var(--border);letter-spacing:.2px}
.badge.member{background:#0c1631;color:#cfe6ff}
.badge.media{background:linear-gradient(135deg,#8b5cf6,#22d3ee);color:#06121a}
.badge.moderator{background:linear-gradient(135deg,#22c55e,#0ea5e9);color:#051118}
.badge.admin{background:linear-gradient(135deg,#f59e0b,#fb923c);color:#1a1206;border-color:rgba(0,0,0,.2);font-weight:900}
.badge.owner{background:linear-gradient(135deg,#3b82f6,#22c1dc);color:#041018;border-color:transparent;font-weight:900}

.level{font-size:10px;padding:3px 7px;border-radius:999px;background:#0b1f3a;color:#cfe6ff;border:1px solid var(--border)}
.user-link{cursor:pointer;font-weight:800;padding:2px 6px;border-radius:8px;background:#0b1f3a;border:1px solid var(--border)}

.fab{position:fixed;right:18px;bottom:18px;width:56px;height:56px;border-radius:50%;background:linear-gradient(135deg,#3b82f6,#22c1dc);border:none;cursor:pointer;display:flex;align-items:center;justify-content:center;box-shadow:0 14px 30px rgba(59,130,246,.35), 0 4px 10px rgba(0,0,0,.35);z-index:50}
.fab svg{width:26px;height:26px;fill:#041018}

/* Modals */
.modal{position:fixed;inset:0;background:rgba(0,0,0,.55);display:none;align-items:center;justify-content:center;z-index:80}
.modal.open{display:flex}
.modal .box{width:min(560px,92vw);background:linear-gradient(180deg,#0f1a33,#0b1326);border:1px solid var(--border);border-radius:16px;padding:16px}
.modal .head{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}
.modal .foot{display:flex;gap:8px;justify-content:flex-end;margin-top:12px}

.grid-2{display:grid;grid-template-columns:1fr;gap:16px}
@media(min-width:900px){.grid-2{grid-template-columns:1.1fr .9fr}}

.lb-controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
.seg{display:flex;border:1px solid var(--border);border-radius:12px;overflow:hidden}
.seg button{padding:8px 12px;background:#0c1631;color:#dce7ff;border:none;cursor:pointer}
.seg button.active{background:linear-gradient(135deg,#3b82f6,#22c1dc);color:#051326;font-weight:800}

/* About Us */
.about .links{display:flex;gap:10px;flex-wrap:wrap}

/* Hide profile page legacy */
#page-profile{display:none!important}
</style>
</head>
<body>

<!-- Preloader -->
<div id="preload"><div class="loader"></div></div>

<!-- Toasts -->
<div id="toasts"></div>

<div class="header">
  <div class="header-inner container">
    <div class="left">
      <a class="brand" href="/" id="homeLink">
        <img class="logo" src="/img/GrowCBnobackground.png" alt="GROWCB" />
        <span class="name">GROWCB</span>
      </a>
      <div class="tabs">
        <a class="tab active" id="tab-games" data-path="/">Games</a>
        <a class="tab" id="tab-ref" data-path="/referral">Referral</a>
        <a class="tab" id="tab-promo" data-path="/promocodes">Promo Codes</a>
        <a class="tab" id="tab-lb" data-path="/leaderboard">Leaderboard</a>
        <a class="tab" id="tab-about" data-path="/about">About Us</a>
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
      </div>
      <div class="games-grid" style="margin-top:12px">
        <div class="game-card" data-path="/crash" id="openCrash"><img class="banner" src="/img/crash.png" alt="Crash"/></div>
        <div class="game-card" data-path="/mines" id="openMines"><img class="banner" src="/img/mines.png" alt="Mines"/></div>
        <div class="game-card" data-path="/coinflip" id="openCoinflip"><img class="banner" src="/img/coinflip.png" alt="Coinflip"/></div>
        <div class="game-card" data-path="/blackjack" id="openBlackjack"><img class="banner" src="/img/blackjack.png" alt="Blackjack"/></div>
        <div class="game-card" data-path="/pump" id="openPump"><img class="banner" src="/img/pump.png" alt="Pump"/></div>
      </div>
    </div>
  </div>

  <!-- Crash -->
  <div id="page-crash" style="display:none">
    <div class="card">
      <div class="hero">
        <div style="display:flex;align-items:baseline;gap:10px"><div class="big" id="crNow">0.00×</div><div class="muted" id="crHint">Loading…</div></div>
        <button class="btn gray" id="backToGames">← Games</button>
      </div>
      <div class="cr-graph-wrap" style="margin-top:10px"><canvas id="crCanvas"></canvas></div>
      <div style="margin-top:12px"><div class="label" style="margin-bottom:4px">Previous Busts</div><div id="lastBusts" class="muted">Loading last rounds…</div></div>
      <div class="games-grid" style="grid-template-columns:1fr 1fr;gap:12px;margin-top:8px">
        <div class="field"><div class="label">Bet (DL)</div><input id="crBet" type="number" min="1" step="0.01" placeholder="min 1.00"/></div>
        <div class="field"><div class="label">Auto Cashout (×) — optional</div><input id="crCash" type="number" min="1.01" step="0.01" placeholder="e.g. 2.00"/></div>
      </div>
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:8px">
        <button class="btn primary" id="crPlace">Place Bet</button>
        <button class="btn alt" id="crCashout" style="display:none">💸 Cash Out</button>
        <span id="crMsg" class="muted"></span>
      </div>
      <div class="card" style="margin-top:14px"><div class="label">Your recent rounds</div><div id="crLast" class="muted">—</div></div>
    </div>
  </div>

  <!-- Mines -->
  <div id="page-mines" style="display:none">
    <div class="card">
      <div class="hero"><div class="big">💣 Mines</div><button class="btn gray" id="backToGames2">← Games</button></div>
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
            <button class="btn alt" id="mCash" style="display:none">💸 Cash Out</button>
            <span class="balance-chip"><span class="t">Multiplier:</span> <strong id="mMult">1.0000×</strong></span>
            <span class="balance-chip"><span class="t">Potential:</span> <strong id="mPotential">—</strong></span>
          </div>
          <div class="kpi" style="display:flex;gap:8px;flex-wrap:wrap;margin-top:8px">
            <span class="balance-chip" id="mHash">Commit: —</span>
            <span class="balance-chip" id="mStatus">Status: —</span>
            <span class="balance-chip" id="mPicks">Picks: 0</span>
            <span class="balance-chip" id="mBombs">Mines: 3</span>
          </div>
          <div class="card" style="margin-top:14px"><div class="label">Recent Mines Games</div><div id="mHist" class="muted">—</div></div>
        </div>
        <div>
          <div class="card" style="min-height:420px;display:grid;place-items:center">
            <div id="mGrid" style="display:grid;gap:10px;grid-template-columns:repeat(5,64px)"></div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Coming soon placeholders -->
  <div id="page-coinflip" style="display:none"><div class="card"><div class="hero"><div class="big">🪙 Coinflip</div><button class="btn gray" id="backToGames_cf">← Games</button></div><div class="muted" style="margin-top:8px">Coming soon.</div></div></div>
  <div id="page-blackjack" style="display:none"><div class="card"><div class="hero"><div class="big">🃏 Blackjack</div><button class="btn gray" id="backToGames_bj">← Games</button></div><div class="muted" style="margin-top:8px">Coming soon.</div></div></div>
  <div id="page-pump" style="display:none"><div class="card"><div class="hero"><div class="big">📈 Pump</div><button class="btn gray" id="backToGames_pu">← Games</button></div><div class="muted" style="margin-top:8px">Coming soon.</div></div></div>

  <!-- Referral -->
  <div id="page-ref" style="display:none">
    <div class="card">
      <div class="hero">
        <div class="big">🙌 Referral Program</div>
      </div>
      <div class="sep"></div>
      <div class="grid-2">
        <div class="card">
          <div class="label">Your Referral Handle</div>
          <div class="field" style="margin-top:6px"><input id="refName" placeholder="choose-handle (3-20 chars)"/></div>
          <div style="display:flex;gap:8px;margin-top:8px">
            <button class="btn primary" id="refSave">Save</button>
            <button class="btn ghost" id="copyRef">Copy Link</button>
          </div>
          <div class="muted" id="refMsg" style="margin-top:6px"></div>
          <div class="sep"></div>
          <div class="label">Share Link</div>
          <div class="balance-chip" style="margin-top:6px"><input id="refLink" readonly value=""/></div>
        </div>
        <div class="card">
          <div class="label">Stats</div>
          <div class="balance-chip" style="margin-top:8px">Clicks: <strong id="refClicks" style="margin-left:6px">0</strong></div>
          <div class="balance-chip" style="margin-top:8px">Joins: <strong id="refJoins" style="margin-left:6px">0</strong></div>
          <div class="muted" style="margin-top:6px">Clicks count when someone opens your link. Joins count when they sign in.</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Promo -->
  <div id="page-promo" style="display:none">
    <div class="card">
      <div class="hero"><div class="big">🎁 Promo Codes</div></div>
      <div class="sep"></div>
      <div class="grid-2">
        <div class="card">
          <div class="label">Redeem</div>
          <div style="display:flex;gap:8px;margin-top:6px">
            <input id="promoInput" placeholder="e.g. WELCOME10"/>
            <button class="btn primary" id="redeemBtn">Redeem</button>
          </div>
          <div id="promoMsg" class="muted" style="margin-top:6px"></div>
        </div>
        <div class="card">
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
        <span class="muted">Anonymous players show as “Anonymous”. Amounts hidden for anonymous users.</span>
      </div>
      <div id="lbWrap" class="muted">Loading…</div>
    </div>
  </div>

  <!-- About Us -->
  <div id="page-about" style="display:none">
    <div class="card about">
      <div class="hero"><div class="big">ℹ️ About GROWCB</div></div>
      <p style="color:#dbe8ff;line-height:1.5;margin-top:10px">
        GROWCB is a community project offering fun, provably-fair mini-games like Crash and Mines, with Discord integration.
      </p>
      <div class="links" style="margin-top:10px">
        <a class="btn primary" href="__INVITE__" target="_blank" rel="noopener">Discord</a>
        <a class="btn ghost" href="https://instagram.com/" target="_blank" rel="noopener">Instagram</a>
        <a class="btn ghost" href="https://growcb.net/" target="_blank" rel="noopener">Website</a>
      </div>
    </div>
  </div>

  <!-- Owner Panel -->
  <div id="page-owner" style="display:none">
    <div class="card">
      <div class="hero"><div class="big">Owner / Admin Panel</div></div>
      <div class="sep"></div>
      <div class="grid-2">
        <div class="card">
          <div class="label">Adjust Balance</div>
          <div class="field" style="margin-top:8px"><input id="opIdent" placeholder="UserID, @mention, or exact handle"/></div>
          <div style="display:flex;gap:8px;margin-top:8px">
            <input id="opAmt" placeholder="+10 or -5.25"/>
            <input id="opReason" placeholder="Reason (optional)"/>
            <button class="btn primary" id="opApply">Apply</button>
          </div>
          <div id="opMsg" class="muted" style="margin-top:6px"></div>
        </div>
        <div class="card">
          <div class="label">Roles / Announcements</div>
          <div style="display:flex;gap:8px;margin-top:8px">
            <input id="roleIdent" placeholder="Target (id/mention/handle)"/>
            <select id="roleSelect">
              <option value="member">member</option>
              <option value="media">media</option>
              <option value="moderator">moderator</option>
              <option value="admin">admin</option>
              <option value="owner">owner</option>
            </select>
            <button class="btn primary" id="roleApply">Set Role</button>
          </div>
          <div class="sep"></div>
          <div style="display:flex;gap:8px"><input id="announceTxt" placeholder="Global announcement…"/><button class="btn ghost" id="announceBtn">Announce</button></div>
          <div id="roleMsg" class="muted" style="margin-top:6px"></div>
        </div>
      </div>
    </div>
  </div>

</div>

<!-- Floating chat -->
<button class="fab" id="fabChat" title="Open chat"><svg viewBox="0 0 24 24"><path d="M4 4h16v12H7l-3 3V4z"/></svg></button>
<div class="chat-drawer" id="chatDrawer">
  <div class="chat-head"><div>Global Chat <span id="chatNote" class="muted"></span></div><button class="btn gray" id="chatClose">Close</button></div>
  <div class="chat-body" id="chatBody"></div>
  <div class="chat-input"><input id="chatText" placeholder="Say something… (Enter to send)"/><button class="btn primary" id="chatSend">Send</button></div>
</div>

<!-- Deposit / Withdraw Modal -->
<div class="modal" id="dwModal">
  <div class="box">
    <div class="head"><div class="big">💠 Deposit / Withdraw</div><button class="btn gray" id="dwClose">Close</button></div>
    <div class="field"><div class="label">Action</div>
      <div style="display:flex;gap:8px">
        <button class="btn primary" id="dwActionDeposit">Deposit</button>
        <button class="btn ghost" id="dwActionWithdraw">Withdraw</button>
      </div>
    </div>
    <div class="grid-2" style="margin-top:10px">
      <div>
        <div class="field"><div class="label">Discord Account</div>
          <div style="display:flex;gap:8px">
            <input id="dwDiscord" placeholder="username#1234 or ID"/>
            <button class="btn gray" id="dwUseMe">Use my account</button>
          </div>
        </div>
        <div class="field" style="margin-top:8px"><div class="label">GrowID</div><input id="dwGrow" placeholder="Your GrowID"/></div>
        <div class="field" style="margin-top:8px"><div class="label">World</div><input id="dwWorld" placeholder="World name"/></div>
      </div>
      <div>
        <div class="field"><div class="label">Amount (for withdraw only)</div><input id="dwAmount" placeholder="e.g. 10.00"/></div>
        <div class="muted" style="margin-top:8px">Heads up: if GrowID is incorrect, delivery may fail.</div>
      </div>
    </div>
    <div class="foot">
      <button class="btn primary" id="dwSubmit">Submit</button>
    </div>
  </div>
</div>

<!-- Profile Modal (from chat username click) -->
<div class="modal" id="profileModal">
  <div class="box">
    <div class="head"><div class="big" id="pmTitle">Player</div><button class="btn gray" id="pmClose">Close</button></div>
    <div id="pmBody">Loading…</div>
  </div>
</div>

<script>
const REF_BASE="__REF_BASE__";
const OWNER_ID="__OWNER_ID__";

const qs = id => document.getElementById(id);
const qsa = sel => Array.from(document.querySelectorAll(sel));

function toast(msg, type='info', title='Notice'){
  const box = document.createElement('div'); box.className='toast ' + (type==='error'?'err':(type==='success'?'ok':'')); 
  box.innerHTML = `<div class="t">${title}</div><div class="m">${msg}</div>`;
  qs('toasts').appendChild(box);
  setTimeout(()=>{ box.style.opacity='0'; setTimeout(()=> box.remove(), 250); }, 3500);
}

const j = async (url, init) => {
  const r = await fetch(url, init);
  if(!r.ok){
    let t = await r.text().catch(()=> '');
    try{ const js = JSON.parse(t); throw new Error(js.detail || js.message || t || r.statusText); }
    catch{ throw new Error(t || r.statusText); }
  }
  const ct = (r.headers.get('content-type')||'').toLowerCase();
  return ct.includes('application/json') ? r.json() : r.text();
};
const fmtDL=(n)=> (Number(n)||0).toFixed(2);
function dlHtml(n){
  const s = fmtDL(n);
  const [i,d] = s.split('.');
  return `<span class="dl-num"><span class="dl-int">${i}</span><span class="dl-dec">.${d}</span></span><img class="dl-icon" src="/img/diamondlock.png" alt="DL"/>`;
}

/* Router */
const pages = ['page-games','page-crash','page-mines','page-coinflip','page-blackjack','page-pump','page-ref','page-promo','page-lb','page-about','page-owner'];
const pathToPage = {
  '/': 'page-games',
  '/crash': 'page-crash',
  '/mines': 'page-mines',
  '/coinflip': 'page-coinflip',
  '/blackjack': 'page-blackjack',
  '/pump': 'page-pump',
  '/referral': 'page-ref',
  '/promocodes': 'page-promo',
  '/leaderboard': 'page-lb',
  '/about': 'page-about',
  '/owner': 'page-owner'
};
function setActiveTabByPath(path){
  const map = {'/':'tab-games','/referral':'tab-ref','/promocodes':'tab-promo','/leaderboard':'tab-lb','/about':'tab-about'};
  for(const id of ['tab-games','tab-ref','tab-promo','tab-lb','tab-about']){
    const el = qs(id); if(el) el.classList.toggle('active', map[path]===id);
  }
}
function showOnly(id){
  for(const p of pages){ const el = qs(p); if(el) el.style.display = (p===id) ? '' : 'none'; }
}
function goto(path, replace=false){
  const pg = pathToPage[path] || 'page-games';
  showOnly(pg);
  setActiveTabByPath(path);
  if(replace) history.replaceState({path}, '', path);
  else history.pushState({path}, '', path);
  // on entering some pages, load data
  if(pg==='page-ref') loadReferral();
  if(pg==='page-promo') renderPromo();
  if(pg==='page-lb') refreshLeaderboard();
  if(pg==='page-crash'){ openCrash(); }
  if(pg==='page-mines'){ refreshMines(); }
}
function handleRoute(){
  const path = location.pathname;
  goto(path, true);
}
window.addEventListener('popstate', handleRoute);

/* Header / Auth */
async function renderHeader(){
  let me=null, bal=null, prof=null;
  try{ me = await j('/api/me'); }catch(_){}
  if(me){
    try{ bal = await j('/api/balance'); }catch(_){}
    try{ prof = await j('/api/profile'); }catch(_){}
    const ownerBtn = (prof && (prof.role==='owner')) ? `<div class="item" id="menuOwner">Owner Panel</div>` : '';
    qs('authArea').innerHTML = `
      <button class="btn primary" id="btnDW">Deposit / Withdraw</button>
      <button class="btn ghost" id="btnJoinSmall">${me.in_guild ? 'In Discord' : 'Join Discord'}</button>
      <span class="balance-chip" id="balChip">${bal ? dlHtml(bal.balance) : dlHtml(0)}</span>
      <div class="avatar-wrap">
        <img class="avatar" id="avatarBtn" src="${me.avatar_url||''}" title="${me.username||'user'}"/>
        <div id="userMenu" class="menu">
          ${ownerBtn}
          <div class="item" id="menuSettings">Settings</div>
          <a class="item" href="/logout">Logout</a>
        </div>
      </div>
    `;
    qs('btnJoinSmall').onclick = joinDiscord;
    qs('btnDW').onclick = ()=> openDW();
    const menu = qs('userMenu'); const av = qs('avatarBtn');
    av.onclick = (e)=>{ e.stopPropagation(); menu.classList.toggle('open'); };
    document.body.addEventListener('click', ()=> menu.classList.remove('open'));
    qs('menuSettings').onclick = ()=> toast('Settings open not implemented in this view.', 'info', 'Soon');
    if(qs('menuOwner')) qs('menuOwner').onclick = ()=> goto('/owner');
  }else{
    qs('authArea').innerHTML = `
      <button class="btn primary" id="btnDW">Deposit / Withdraw</button>
      <a class="btn ghost" href="/login">Login with Discord</a>
    `;
    qs('btnDW').onclick = ()=> openDW();
  }
}
async function joinDiscord(){
  try{ await j('/api/discord/join', { method:'POST' }); toast('Joined the Discord server!', 'success', 'Success'); renderHeader(); }
  catch(e){ toast(e.message || 'Could not join. Try relogin.', 'error', 'Discord'); }
}

/* Deposit / Withdraw modal */
let dwType='deposit';
function openDW(){
  qs('dwModal').classList.add('open');
  setDWType('deposit');
}
function setDWType(t){
  dwType = t;
  qs('dwActionDeposit').classList.toggle('primary', t==='deposit');
  qs('dwActionWithdraw').classList.toggle('primary', t==='withdraw');
  qs('dwActionDeposit').classList.toggle('ghost', t!=='deposit');
  qs('dwActionWithdraw').classList.toggle('ghost', t!=='withdraw');
}
qs('dwActionDeposit').onclick = ()=> setDWType('deposit');
qs('dwActionWithdraw').onclick = ()=> setDWType('withdraw');
qs('dwClose').onclick = ()=> qs('dwModal').classList.remove('open');
qs('dwUseMe').onclick = async ()=>{
  try{ const me = await j('/api/me'); qs('dwDiscord').value = me.username || me.id || ''; }catch(_){ toast('Not logged in. Enter your Discord username or ID.', 'error', 'Discord'); }
};
qs('dwSubmit').onclick = async ()=>{
  const payload = {
    ttype: dwType,
    amount: (dwType==='withdraw') ? (qs('dwAmount').value||'') : null,
    world: qs('dwWorld').value||'',
    grow_id: qs('dwGrow').value||''
  };
  if(!payload.world){ toast('World is required.', 'error', 'Missing'); return; }
  if(!payload.grow_id){ toast('GrowID is required.', 'error', 'Missing'); return; }
  try{
    const r = await j('/api/transfer/create', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
    toast(`Request #${r.id} submitted.`, 'success', 'Submitted');
    qs('dwModal').classList.remove('open');
  }catch(e){ toast(e.message, 'error', 'Transfer'); }
};

/* Tabs / clicks / brand */
qsa('.tab').forEach(el=> el.onclick = (e)=>{ e.preventDefault(); goto(el.dataset.path||'/'); });
qs('homeLink').onclick = (e)=>{ e.preventDefault(); goto('/'); };

/* Referral */
async function loadReferral(){
  try{
    const st = await j('/api/referral/state');
    if(st && st.name){ qs('refName').value = st.name; qs('refLink').value = (REF_BASE + '/' + st.name); }
    else qs('refLink').value = '';
    qs('refClicks').textContent = st.clicks||0; qs('refJoins').textContent = st.joined||0;
  }catch(_){}
}
qs('refSave').onclick = async()=>{
  const name = qs('refName').value.trim();
  qs('refMsg').textContent = '';
  try{
    await j('/api/referral/set', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ name })});
    qs('refMsg').textContent = 'Saved.'; qs('refLink').value = REF_BASE + '/' + name.toLowerCase();
    toast('Referral saved', 'success', 'Referral');
  }catch(e){ qs('refMsg').textContent = e.message; toast(e.message, 'error', 'Referral'); }
};
qs('copyRef').onclick = ()=>{ const inp = qs('refLink'); inp.select(); inp.setSelectionRange(0, 99999); document.execCommand('copy'); toast('Link copied', 'success'); };

/* Promo */
async function renderPromo(){
  try{
    const my = await j('/api/promo/my');
    qs('myCodes').innerHTML = (my.rows && my.rows.length)
      ? '<table><thead><tr><th>Code</th><th>Redeemed</th></tr></thead><tbody>' +
        my.rows.map(r=>`<tr><td>${r.code}</td><td>${new Date(r.redeemed_at).toLocaleString()}</td></tr>`).join('') +
        '</tbody></table>' : '—';
  }catch(_){ qs('myCodes').textContent = '—'; }
}
qs('redeemBtn').onclick = async ()=>{
  const code = qs('promoInput').value.trim(); qs('promoMsg').textContent = '';
  try{ const r = await j('/api/promo/redeem', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ code }) });
    qs('promoMsg').textContent = 'Redeemed! New balance: ' + fmtDL(r.new_balance) + ' DL'; renderHeader(); renderPromo(); toast('Promo redeemed', 'success');
  }catch(e){ qs('promoMsg').textContent = e.message; toast(e.message, 'error', 'Promo'); }
};

/* Leaderboard */
let lbPeriod='daily';
function nextUtcMidnight(){ const n=new Date(); return new Date(Date.UTC(n.getUTCFullYear(),n.getUTCMonth(),n.getUTCDate()+1,0,0,0)); }
function endOfUtcMonth(){ const n=new Date(); return new Date(Date.UTC(n.getUTCFullYear(),n.getUTCMonth()+1,1,0,0,0)); }
async function refreshLeaderboard(){
  const wrap = qs('lbWrap'); wrap.textContent='Loading…';
  const res = await j('/api/leaderboard?period='+lbPeriod+'&limit=50');
  const rows = res.rows||[];
  wrap.innerHTML = rows.length ? `
    <table>
      <thead><tr><th>#</th><th>Name</th><th>Wagered</th></tr></thead>
      <tbody>${rows.map((r,i)=>`<tr><td>${i+1}</td><td class="name">${r.is_anon? 'Anonymous' : r.display_name}</td><td>${r.is_anon? '—' : (fmtDL(r.total_wagered)+' DL')}</td></tr>`).join('')}</tbody>
    </table>` : '—';

  const tgt = lbPeriod==='daily' ? nextUtcMidnight() : lbPeriod==='monthly' ? endOfUtcMonth() : null;
  if(tgt){
    const tick = ()=>{
      const now=new Date(), ms=tgt-now; if(ms<=0){ qs('lbCountdown').textContent='Resets soon…'; return; }
      const s=Math.floor(ms/1000), h=Math.floor(s/3600), m=Math.floor((s%3600)/60), sc=s%60;
      qs('lbCountdown').textContent=`Resets in ${h}h ${m}m ${sc}s`;
      requestAnimationFrame(()=>setTimeout(tick,500));
    }; tick();
  }else qs('lbCountdown').textContent='All-time';

  const seg=qs('lbSeg'); Array.from(seg.querySelectorAll('button')).forEach(b=>{
    b.classList.toggle('active', b.dataset.period===lbPeriod);
    b.onclick=()=>{ lbPeriod=b.dataset.period; refreshLeaderboard(); };
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
      qs('crNow').textContent = m.toFixed(2)+'×';
      qs('crHint').textContent = 'In flight…';
      drawCrash(m);
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
      qs('crHint').textContent = 'Round ended';
      drawCrash(Number(crBust)||1);
    }
  }catch(e){ /* silent */ }
  finally{ crPollTimer = setTimeout(pollCrash, 900); }
}
qs('crPlace').onclick = async ()=>{
  const bet = qs('crBet').value || '0';
  const cash = qs('crCash').value || null;
  try{ await j('/api/crash/place', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ bet, cashout: cash? Number(cash): null })});
    toast('Bet placed', 'success'); }catch(e){ toast(e.message, 'error', 'Crash'); }
};
qs('crCashout').onclick = async ()=>{
  try{ await j('/api/crash/cashout', { method:'POST' }); toast('Successfully cashed out!', 'success', 'Crash'); }
  catch(e){ toast(e.message, 'error', 'Crash'); }
};
function openCrash(){ showOnly('page-crash'); if(crPollTimer) clearTimeout(crPollTimer); pollCrash(); }

/* Mines UI */
function buildMinesGrid(){
  const grid = qs('mGrid'); grid.innerHTML='';
  for(let i=0;i<25;i++){
    const b = document.createElement('button');
    b.textContent = '?';
    Object.assign(b.style,{width:'64px',height:'64px',borderRadius:'12px',border:'1px solid var(--border)',background:'#0f1a33',color:'#cfe6ff'});
    b.dataset.index = i; b.onclick = ()=> pickCell(i);
    grid.appendChild(b);
  }
}
async function pickCell(i){ try{ await j('/api/mines/pick?index='+i, { method:'POST' }); await refreshMines(); }catch(e){ toast(e.message,'error','Mines'); } }
async function startMines(){
  const bet = qs('mBet').value || '0';
  const mines = parseInt(qs('mMines').value||'3',10);
  try{ await j('/api/mines/start', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ bet, mines })}); await refreshMines(); }
  catch(e){ toast(e.message, 'error', 'Mines'); }
}
async function cashoutMines(){ try{ await j('/api/mines/cashout', { method:'POST' }); await refreshMines(); toast('Cashed out!', 'success', 'Mines'); }catch(e){ toast(e.message,'error','Mines'); } }
async function refreshMines(){
  try{
    const st = await j('/api/mines/state'); const grid = qs('mGrid'); const status = st?.status || 'idle';
    qs('mStatus').textContent = 'Status: ' + status;
    qs('mPicks').textContent = 'Picks: ' + (st?.picks||0);
    qs('mBombs').textContent = 'Mines: ' + (st?.mines|| (qs('mMines').value||3));
    qs('mHash').textContent = 'Commit: ' + (st?.commit_hash || '—');
    qs('mMult').textContent = st?.multiplier ? (Number(st.multiplier)||1).toFixed(4)+'×' : '1.0000×';
    qs('mPotential').textContent = st?.potential_win ? (fmtDL(st.potential_win)+' DL') : '—';
    const playing = status==='active'; qs('mCash').style.display = playing ? '' : 'none'; qs('mSetup').style.display = playing ? 'none' : '';
    if(grid.children.length!==25) buildMinesGrid();
    if(st?.reveals && Array.isArray(st.reveals)){
      st.reveals.forEach((cell, idx)=>{ const b = grid.children[idx]; if(!b) return;
        if(cell==='u'){ b.textContent='?'; b.disabled=false; b.style.background='#0f1a33'; }
        else if(cell==='g'){ b.textContent='✅'; b.disabled=true; b.style.background='#163a2a'; }
        else if(cell==='b'){ b.textContent='💣'; b.disabled=true; b.style.background='#3a1620'; }
      });
    }
    const h = await j('/api/mines/history');
    qs('mHist').innerHTML = (h.rows && h.rows.length)
      ? '<table><thead><tr><th>Time</th><th>Bet</th><th>Mines</th><th>Win</th><th>Status</th></tr></thead><tbody>' +
        h.rows.map(r=>`<tr><td>${new Date(r.started_at).toLocaleString()}</td><td>${fmtDL(r.bet)} DL</td><td>${r.mines}</td><td>${fmtDL(r.win)} DL</td><td>${r.status}</td></tr>`).join('') +
        '</tbody></table>' : '—';
  }catch(_){}
}
qs('mStart').onclick = startMines;
qs('mCash').onclick = cashoutMines;

/* Chat UI */
let chatOpen=false, chatTimer=null, lastChatId=0;
function toggleChat(open){ chatOpen=open; qs('chatDrawer').classList.toggle('open', open); if(open){ pollChat(); } else { if(chatTimer) clearTimeout(chatTimer); } }
qs('fabChat').onclick = ()=> toggleChat(true);
qs('chatClose').onclick = ()=> toggleChat(false);
qs('chatText').addEventListener('keydown', (e)=>{ if(e.key==='Enter'){ e.preventDefault(); qs('chatSend').click(); } });
async function pollChat(){
  try{
    const r = await j('/api/chat/fetch?since='+lastChatId+'&limit=50'); const arr = r.rows||[];
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
          <div>${escapeHtml(m.text)}</div>`;
        body.appendChild(row);
      }
      qs('chatBody').scrollTop = qs('chatBody').scrollHeight;
    }
  }catch(_){}
  finally{ chatTimer=setTimeout(pollChat,1200); }
}
function escapeHtml(s){ return String(s||'').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
qs('chatSend').onclick = async ()=>{
  const t = qs('chatText').value.trim(); if(!t) return;
  try{ await j('/api/chat/send', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ text: t })}); qs('chatText').value=''; }
  catch(e){
    if(String(e.message||'').toLowerCase().includes('level 5')) toast('You need level 5 to chat.', 'error', 'Chat');
    else toast(e.message, 'error', 'Chat');
  }
};
// open profile modal on username click
qs('chatBody').addEventListener('click', async (e)=>{
  const el = e.target.closest('.user-link'); if(!el) return;
  const uid = el.getAttribute('data-uid');
  try{
    const p = await j('/api/profile/public?user_id='+encodeURIComponent(uid));
    qs('pmTitle').textContent = p.name + (p.is_anon ? ' (Anonymous)' : '');
    qs('pmBody').innerHTML = `
      <div class="games-grid" style="grid-template-columns:1fr 1fr 1fr">
        <div class="card"><div class="label">Level</div><div class="big">Lv ${p.level}</div><div class="muted">${p.xp} XP</div></div>
        <div class="card"><div class="label">Balance</div><div class="big">${fmtDL(p.balance)} DL</div></div>
        <div class="card"><div class="label">Role</div><div class="big" style="text-transform:uppercase">${p.role}</div></div>
      </div>
      <div class="sep"></div>
      <div class="grid-2">
        <div class="card"><div class="label">Crash Games</div><div class="big">${p.crash_games}</div></div>
        <div class="card"><div class="label">Mines Games</div><div class="big">${p.mines_games}</div></div>
      </div>`;
    qs('profileModal').classList.add('open');
  }catch(err){ toast('Could not load profile.', 'error'); }
});
qs('pmClose').onclick = ()=> qs('profileModal').classList.remove('open');

/* Games navigation clicks */
qsa('.game-card').forEach(el=> el.onclick = ()=> goto(el.dataset.path||'/'));
qs('backToGames').onclick = ()=> goto('/');
qs('backToGames2').onclick = ()=> goto('/');
qs('backToGames_cf').onclick = ()=> goto('/');
qs('backToGames_bj').onclick = ()=> goto('/');
qs('backToGames_pu').onclick = ()=> goto('/');

/* Owner Panel actions */
if(qs('opApply')){
  qs('opApply').onclick = async ()=>{
    try{
      const r = await j('/api/admin/adjust',{ method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ identifier: qs('opIdent').value.trim(), amount: qs('opAmt').value.trim(), reason: qs('opReason').value.trim() })});
      qs('opMsg').textContent = 'OK. New balance: ' + fmtDL(r.new_balance) + ' DL'; renderHeader();
      toast('Balance adjusted', 'success');
    }catch(e){ qs('opMsg').textContent = e.message; toast(e.message,'error','Owner'); }
  };
  qs('roleApply').onclick = async ()=>{
    try{
      await j('/api/admin/role', { method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ identifier: qs('roleIdent').value.trim(), role: qs('roleSelect').value })});
      qs('roleMsg').textContent = 'Role updated.'; toast('Role updated','success');
    }catch(e){ qs('roleMsg').textContent = e.message; toast(e.message,'error','Owner'); }
  };
  qs('announceBtn').onclick = async ()=>{
    try{
      await j('/api/admin/announce', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ text: qs('announceTxt').value.trim() })});
      qs('announceTxt').value=''; toast('Announcement sent','success');
    }catch(e){ toast(e.message,'error','Owner'); }
  };
}

/* Boot */
(async function boot(){
  // Preloader waits for images to complete
  const imgs = Array.from(document.images);
  if(imgs.length){
    await Promise.allSettled(imgs.map(img=> new Promise(res=> { if(img.complete) return res(); img.onload=img.onerror=()=>res(); } )));
  }
  qs('preload').classList.add('hide');

  buildMinesGrid();
  renderHeader();
  handleRoute();
})();
</script>
</body>
</html>
"""

# ---------- SPA roots (serve same index for these paths) ----------
from fastapi.responses import HTMLResponse

SPA_PATHS = {"", "crash", "mines", "coinflip", "blackjack", "pump", "referral", "promocodes", "leaderboard", "about", "owner"}

@app.get("/", response_class=HTMLResponse)
async def index():
    html = HTML_TEMPLATE.replace("__INVITE__", DISCORD_INVITE or "__INVITE__") \
                        .replace("__OWNER_ID__", str(OWNER_ID)) \
                        .replace("__REF_BASE__", os.getenv("REFERRAL_SHARE_BASE", "https://growcb.new/referral"))
    return HTMLResponse(html)

@app.get("/{path_name}", response_class=HTMLResponse)
async def spa_paths(path_name: str):
    if path_name in SPA_PATHS:
        return await index()
    # Allow unknown paths to still render index if they look like our SPA main pages
    if path_name.strip("/") in SPA_PATHS:
        return await index()
    raise HTTPException(404, "Not found")

# ---------- Discord Bot ----------
import discord
from discord.ext import commands

FOOTER_URL = os.getenv("FOOTER_URL", "https://growcb.net")

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=PREFIX, intents=intents, help_command=None)

def _id_from_any(s: str) -> Optional[str]:
    if not s: return None
    m = re.search(r"\d{5,}", s)
    if m: return m.group(0)
    # maybe exact handle
    with psycopg.connect(DATABASE_URL) as con, con.cursor() as cur:
        cur.execute("SELECT user_id FROM profiles WHERE name_lower=%s", (s.lower(),))
        r = cur.fetchone()
        if r: return str(r[0])
    return None

def embed_base(title: str, desc: str = ""):
    em = discord.Embed(title=title, description=desc, color=0x3b82f6)
    em.set_footer(text=FOOTER_URL)
    return em

@bot.event
async def on_ready():
    print(f"Bot logged in as {bot.user} (ready)")
    await bot.change_presence(activity=discord.Game(name="growcb.net"))

@bot.command(name="help")
async def _help(ctx: commands.Context):
    em = embed_base("Help — GROWCB Bot")
    em.add_field(name=".bal [@user|id]", value="Show your balance or another user's balance.", inline=False)
    em.add_field(name=".level [@user|id]", value="Show level, XP and role.", inline=False)
    em.add_field(name=".leaderboard", value="Show leaderboard with buttons to switch Daily / Monthly / All-time.", inline=False)
    em.add_field(name=".addbal <@user|id> <amount>", value="(Owner/Webhooks) Add DL balance.", inline=False)
    em.add_field(name=".removebal <@user|id> <amount>", value="(Owner/Webhooks) Remove DL balance.", inline=False)
    await ctx.reply(embed=em)

@bot.command(name="bal")
async def _bal(ctx: commands.Context, user: Optional[str] = None):
    target_id = _id_from_any(user) if user else str(ctx.author.id)
    if not target_id:
        return await ctx.reply(embed=embed_base("Balance", "User not found"))
    bal = float(get_balance(target_id))
    em = embed_base("Balance")
    em.description = f"**{bal:.2f} DL**"
    await ctx.reply(embed=em)

@bot.command(name="level")
async def _level(ctx: commands.Context, user: Optional[str] = None):
    target_id = _id_from_any(user) if user else str(ctx.author.id)
    if not target_id:
        return await ctx.reply(embed=embed_base("Level", "User not found"))
    p = profile_info(target_id)
    prog = f"Level **{p['level']}** — {p['xp']} XP • {p['progress_pct']}% to next"
    em = embed_base("Level / Profile")
    em.add_field(name="User", value=f"<@{target_id}>", inline=True)
    em.add_field(name="Level", value=str(p["level"]), inline=True)
    em.add_field(name="XP", value=str(p["xp"]), inline=True)
    em.add_field(name="Role", value=p["role"], inline=True)
    em.add_field(name="Balance", value=f"{p['balance']:.2f} DL", inline=True)
    em.description = prog
    await ctx.reply(embed=em)

def _lb_reset_text(period: str) -> str:
    now = now_utc()
    if period=="daily":
        nxt = now.replace(hour=0,minute=0,second=0,microsecond=0) + datetime.timedelta(days=1)
        return f"Resets in {(nxt-now).seconds//3600}h {((nxt-now).seconds%3600)//60}m"
    if period=="monthly":
        first_next = now.replace(day=1,hour=0,minute=0,second=0,microsecond=0)
        month = first_next.month + 1
        year = first_next.year + (1 if month>12 else 0)
        month = 1 if month>12 else month
        nxt = first_next.replace(year=year, month=month)
        delta = nxt - now
        return f"Resets in {delta.days}d {(delta.seconds//3600)}h"
    return "All-time"

class LBView(discord.ui.View):
    def __init__(self, author_id: int):
        super().__init__(timeout=60)
        self.author_id = author_id
        self.period = "daily"
        self.update_styles()

    def update_styles(self):
        for item in self.children:
            if isinstance(item, discord.ui.Button):
                item.style = discord.ButtonStyle.secondary
        # set selected to primary
        sel = next((b for b in self.children if isinstance(b, discord.ui.Button) and b.custom_id==self.period), None)
        if sel: sel.style = discord.ButtonStyle.primary

    async def refresh_embed(self, interaction: discord.Interaction):
        rows = get_leaderboard_rows_db(self.period, limit=10)
        lines = []
        for i,r in enumerate(rows, start=1):
            name = "Anonymous" if r["is_anon"] else r["display_name"]
            amt = "—" if r["is_anon"] else f"{r['total_wagered']:.2f} DL"
            lines.append(f"**{i}.** {name} — {amt}")
        em = embed_base(f"Leaderboard — {self.period.title()}", "\n".join(lines) or "—")
        em.set_footer(text=f"{_lb_reset_text(self.period)} • {FOOTER_URL}")
        self.update_styles()
        await interaction.response.edit_message(embed=em, view=self)

    @discord.ui.button(label="Daily", custom_id="daily")
    async def daily(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.period="daily"; await self.refresh_embed(interaction)

    @discord.ui.button(label="Monthly", custom_id="monthly")
    async def monthly(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.period="monthly"; await self.refresh_embed(interaction)

    @discord.ui.button(label="All-time", custom_id="alltime")
    async def alltime(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.period="alltime"; await self.refresh_embed(interaction)

@bot.command(name="leaderboard")
async def _leaderboard(ctx: commands.Context):
    view = LBView(author_id=ctx.author.id)
    rows = get_leaderboard_rows_db("daily", limit=10)
    lines = []
    for i,r in enumerate(rows, start=1):
        name = "Anonymous" if r["is_anon"] else r["display_name"]
        amt = "—" if r["is_anon"] else f"{r['total_wagered']:.2f} DL"
        lines.append(f"**{i}.** {name} — {amt}")
    em = embed_base("Leaderboard — Daily", "\n".join(lines) or "—")
    em.set_footer(text=f"{_lb_reset_text('daily')} • {FOOTER_URL}")
    await ctx.reply(embed=em, view=view)

def _owner_or_webhook(ctx: commands.Context) -> bool:
    try:
        if int(ctx.author.id) == int(OWNER_ID): return True
    except: pass
    # message.webhook_id is not None for webhooks
    return bool(getattr(ctx.message, "webhook_id", None))

@bot.command(name="addbal")
async def _addbal(ctx: commands.Context, user: str, amount: str):
    if not _owner_or_webhook(ctx):
        return await ctx.reply("Only owner or webhooks can use this.")
    target = _id_from_any(user)
    if not target: return await ctx.reply("User not found.")
    newbal = adjust_balance(str(OWNER_ID), target, D(amount), "bot:addbal")
    em = embed_base("Balance Added", f"New balance: **{newbal:.2f} DL**")
    await ctx.reply(embed=em)

@bot.command(name="removebal")
async def _removebal(ctx: commands.Context, user: str, amount: str):
    if not _owner_or_webhook(ctx):
        return await ctx.reply("Only owner or webhooks can use this.")
    target = _id_from_any(user)
    if not target: return await ctx.reply("User not found.")
    newbal = adjust_balance(str(OWNER_ID), target, D("-"+str(amount).lstrip("+")), "bot:removebal")
    em = embed_base("Balance Removed", f"New balance: **{newbal:.2f} DL**")
    await ctx.reply(embed=em)

async def start_bot():
    if not DISCORD_BOT_TOKEN:
        print("DISCORD_BOT_TOKEN not set; bot disabled.")
        return
    try:
        await bot.start(DISCORD_BOT_TOKEN)
    except Exception as e:
        print("Bot start failed:", e)

# ---------- Run both API and Bot ----------
# Keep a small /api/profile endpoint for internal UI needs (role check)
@app.get("/api/profile")
async def api_profile(request: Request):
    s = _require_session(request)
    return profile_info(s["id"])

# Leaderboard API (used by UI)
@app.get("/api/leaderboard")
async def api_leaderboard(period: str = Query("daily"), limit: int = Query(50, ge=1, le=200)):
    rows = get_leaderboard_rows_db(period, limit)
    return {"rows": rows}

# Lifespan: start bot concurrently
@app.on_event("startup")
async def _on_startup():
    if DISCORD_BOT_TOKEN:
        asyncio.create_task(start_bot())

# ---------- Local runner ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)

