# app/main.py
import os, json, asyncio, re, random, string, datetime, base64, secrets, hashlib, hmac, smtplib
from urllib.parse import urlencode, urlparse
from typing import Optional, Tuple, Dict, List
from decimal import Decimal, ROUND_DOWN, getcontext
from contextlib import asynccontextmanager
from email.message import EmailMessage
import ssl

import httpx
import psycopg
from fastapi import FastAPI, Request, HTTPException, Query, Body
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, Response, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from itsdangerous import URLSafeSerializer, BadSignature
from pydantic import BaseModel

# ---------- Games (external modules) ----------
from crash import (
    ensure_betting_round, place_bet, load_round, begin_running,
    finish_round, create_next_betting, last_busts, your_bet,
    your_history, cashout_now, current_multiplier
)
from mines import (
    mines_start, mines_pick, mines_cashout, mines_state, mines_history
)

# ---------- Promo (optional external module) ----------
# If you don’t have promo.py, these fallbacks keep the app importable.
try:
    from promo import redeem_promo, create_promo, PromoInvalid, PromoExpired, PromoExhausted, PromoAlreadyRedeemed
except Exception:
    class PromoInvalid(Exception): pass
    class PromoExpired(Exception): pass
    class PromoExhausted(Exception): pass
    class PromoAlreadyRedeemed(Exception): pass
    def redeem_promo(user_id: str, code: str):
        raise PromoInvalid("Promo feature not available")
    def create_promo(actor_id: str, code: Optional[str], amount: str, max_uses: int, expires_at: Optional[str]):
        raise PromoInvalid("Promo feature not available")

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
REFERRAL_SHARE_BASE = os.getenv("REFERRAL_SHARE_BASE", "https://growcb.net/referral")
CANONICAL_HOST = os.getenv("CANONICAL_HOST", "growcb.net")  # force show this host in browser
OWNER_ONLY_MODE = os.getenv("OWNER_ONLY_MODE", "1") != "0"   # lock site to owner only (default ON)

# Google OAuth
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT = os.getenv("GOOGLE_REDIRECT", "")

# Email SMTP for password reset
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587") or "587")
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
FROM_EMAIL = os.getenv("FROM_EMAIL", "")
RESET_LINK_BASE = os.getenv("RESET_LINK_BASE", f"https://{CANONICAL_HOST}/reset")

# Discord notification channels
DEPOSIT_CHANNEL_ID = int(os.getenv("DISCORD_DEPOSIT_CHANNEL_ID", "1407130350940979210") or "0")
WITHDRAW_CHANNEL_ID = int(os.getenv("DISCORD_WITHDRAW_CHANNEL_ID", "1407130468821766226") or "0")

# New toggles for reliability
START_DISCORD_BOT = os.getenv("START_DISCORD_BOT", "1") != "0"  # set to 0 to prevent bot from starting
RELOAD = os.getenv("RELOAD", "0") == "1"                        # if you want uvicorn.reload=True via env

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

# ---------- Password hashing (email auth) ----------
def _hash_password(password: str) -> str:
    if not password or len(password) < 6:
        raise ValueError("Password must be at least 6 characters")
    salt = base64.urlsafe_b64encode(os.urandom(16)).decode()
    iters = 120_000
    dk = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), iters)
    return f"pbkdf2${iters}${salt}${base64.b64encode(dk).decode()}"

def _verify_password(stored: str, password: str) -> bool:
    try:
        algo, iters_s, salt, h64 = stored.split("$", 3)
        iters = int(iters_s)
        dk = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), iters)
        return hmac.compare_digest(base64.b64decode(h64), dk)
    except Exception:
        return False

def _send_email(to_email: str, subject: str, body: str) -> bool:
    if not (SMTP_HOST and SMTP_USERNAME and SMTP_PASSWORD and FROM_EMAIL):
        # Fallback: log only
        print(f"[EMAIL-FAKE] To: {to_email}\nSubj: {subject}\n\n{body}")
        return False
    msg = EmailMessage()
    msg["From"] = FROM_EMAIL
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)
    ctx = ssl.create_default_context()
    try:
        if SMTP_PORT == 465:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ctx) as s:
                s.login(SMTP_USERNAME, SMTP_PASSWORD)
                s.send_message(msg)
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
                s.starttls(context=ctx)
                s.login(SMTP_USERNAME, SMTP_PASSWORD)
                s.send_message(msg)
        return True
    except Exception as e:
        print("SMTP error:", e)
        return False

# ---------- App / Lifespan / Static ----------
def _get_static_dir():
    base = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(base, "static")
    try: os.makedirs(static_dir, exist_ok=True)
    except Exception: pass
    return static_dir

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
    # Accounts for multi-auth
    cur.execute("""
        CREATE TABLE IF NOT EXISTS accounts (
            user_id TEXT PRIMARY KEY,
            provider TEXT NOT NULL,                -- 'discord' | 'google' | 'email'
            external_id TEXT,                      -- discord id, google sub, or same as email
            email TEXT UNIQUE,
            password_hash TEXT,
            reset_token TEXT,
            reset_expires TIMESTAMPTZ
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
    # announcements (global, non-chat)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS announcements (
            id BIGSERIAL PRIMARY KEY,
            text TEXT NOT NULL,
            starts_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            ends_at TIMESTAMPTZ,
            created_by TEXT,
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
    cur.execute("CREATE INDEX IF NOT EXISTS ix_announcements_times ON announcements((starts_at IS NULL), starts_at, (ends_at IS NULL), ends_at)")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_accounts_provider ON accounts(provider)")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_transfers_created_at ON transfers(created_at)")

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    apply_migrations()
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=_get_static_dir()), name="static")

# ---------- Health ----------
@app.get("/healthz")
async def healthz():
    return {"ok": True, "time": iso(now_utc())}

# ---------- Canonical host redirect (force growcb.net) ----------
@app.middleware("http")
async def _canonical_host_redirect(request: Request, call_next):
    try:
        host = (request.headers.get("host") or urlparse(str(request.url)).hostname or "").split(":")[0].lower()
        if host and CANONICAL_HOST and host not in ("localhost", "127.0.0.1") and host != CANONICAL_HOST.lower():
            url = request.url
            dest = str(url).replace(f"//{host}", f"//{CANONICAL_HOST}")
            return RedirectResponse(dest, status_code=308)
    except Exception:
        pass
    return await call_next(request)

# ---------- Owner-only lock (site closed for everyone except OWNER) ----------
OWNER_GATE_ALLOW = ("/login", "/callback", "/login/google", "/callback/google", "/reset", "/auth/", "/static/", "/img/", "/healthz", "/api/bot/status")

@app.middleware("http")
async def _owner_only_gate(request: Request, call_next):
    if not OWNER_ONLY_MODE:
        return await call_next(request)

    path = request.url.path or "/"
    # allow health, login, callback, assets, bot status
    if any(path == p or path.startswith(p) for p in OWNER_GATE_ALLOW):
        return await call_next(request)

    # Check session
    raw = request.cookies.get("session")
    uid = None
    if raw:
        try:
            uid = URLSafeSerializer(SECRET_KEY, salt="session-v1").loads(raw).get("id")
        except Exception:
            uid = None

    if str(uid) == str(OWNER_ID):
        return await call_next(request)

    # Gate page
    gate_html = """
    <!doctype html><html><head>
      <meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
      <title>GROWCB — Private</title>
      <style>
        :root{--bg:#070b16;--card:#0e1833;--border:#1b2746;--text:#eaf2ff;--muted:#9eb3da}
        *{box-sizing:border-box}body,html{height:100%}body{margin:0;background:radial-gradient(900px 420px at 15% -10%,#12265d55,transparent 60%),linear-gradient(180deg,#070b16,#0a1020);color:var(--text);font-family:Inter,system-ui,Segoe UI,Roboto,Arial,Helvetica,sans-serif}
        .wrap{min-height:100%;display:grid;place-items:center;padding:20px}
        .card{width:min(560px,92vw);background:linear-gradient(180deg,#0f1a33,#0b1326);border:1px solid var(--border);border-radius:16px;padding:20px;box-shadow:0 20px 50px rgba(0,0,0,.35)}
        .title{font-weight:900;font-size:26px;letter-spacing:.3px}
        .muted{color:var(--muted)}
        .btn{display:inline-flex;gap:10px;align-items:center;padding:10px 14px;border-radius:12px;border:1px solid var(--border);cursor:pointer;background:linear-gradient(135deg,#3b82f6,#22c1dc);color:#06121a;font-weight:900}
      </style>
    </head><body>
      <div class="wrap">
        <div class="card">
          <div class="title">GROWCB — Private Beta</div>
          <p class="muted">Access is closed for now. Only the owner can view the site.</p>
          <a class="btn" href="/games">Back to site</a>
        </div>
      </div>
    </body></html>
    """
    return HTMLResponse(gate_html, status_code=403)

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
    try:
        if not request: return None
        host = request.headers.get("host") or urlparse(str(request.url)).hostname
        if not host: return None
        host = host.split(":")[0]
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
        secure=True,
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

# ---------- DB helpers: balances / profiles ----------
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
def ensure_account_row(cur, user_id: str, provider: str, external_id: Optional[str], email: Optional[str], password_hash: Optional[str] = None):
    cur.execute("""
        INSERT INTO accounts(user_id, provider, external_id, email, password_hash)
        VALUES (%s,%s,%s,%s,%s)
        ON CONFLICT (user_id) DO NOTHING
    """, (user_id, provider, external_id, email, password_hash))

@with_conn
def get_profile_name(cur, user_id: str):
    cur.execute("SELECT display_name FROM profiles WHERE user_id=%s", (user_id,))
    r = cur.fetchone(); return r[0] if r else None

@with_conn
def set_profile_is_anon(cur, user_id: str, is_anon: bool):
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

# ---------- Leaderboard helpers ----------
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

# ---------- HTML (UI/UX) ----------
# Improvements:
#  - Robust preloader removal (hides on load & then removes from DOM).
#  - Crash canvas context is initialized in openCrash().
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
<link rel="canonical" href="https://growcb.net"/>
<meta name="theme-color" content="#0a0f1e"/>
<meta property="og:title" content="GROWCB"/>
<meta property="og:description" content="Crash, Mines and more — play with friends."/>
<meta property="og:image" content="/img/GrowCBnobackground.png"/>

<style>
:root{
  --bg:#0a0f1e;--bg2:#0c1428;--card:#111a31;--muted:#9eb3da;--text:#ecf2ff;--accent:#6aa6ff;--accent2:#22c1dc;
  --ok:#34d399;--warn:#f59e0b;--err:#ef4444;--border:#1f2b47;--chatW:360px;--input-bg:#0b1430;--input-br:#223457;--input-tx:#e6eeff;--input-ph:#9db4e4;
  --bannerH: 180px;
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
.btn{display:inline-flex;align-items:center;gap:10px;padding:10px 14px;border-radius:12px;border:1px solid var(--border);cursor:pointer;font-weight:800;user-select:none}
.btn.primary{background:linear-gradient(135deg,#3b82f6,#22c1dc);border-color:transparent;color:#041018;box-shadow:0 12px 24px rgba(59,130,246,.25)}
.btn.ghost{background:linear-gradient(180deg,#0e1833,#0b1326);border:1px solid var(--border);color:#eaf2ff}
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
@media (max-width:720px){ .brand .name{display:none} }

.tabs{display:flex;gap:4px;align-items:center;padding:4px;border-radius:14px;background:linear-gradient(180deg,#0f1a33,#0b1326);border:1px solid var(--border);overflow:auto}
.tab{padding:8px 12px;border-radius:10px;cursor:pointer;font-weight:700;white-space:nowrap;color:#d8e6ff;opacity:.85;transition:all .15s ease;display:flex;align-items:center;gap:8px}
.tab:hover{opacity:1;transform:translateY(-1px)}
.tab.active{background:linear-gradient(135deg,#3b82f6,#22c1dc);color:#051326;box-shadow:0 6px 16px rgba(59,130,246,.25);opacity:1}
@media (max-width:720px){ .tab{padding:6px 8px;font-size:13px} }

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

/* Games grid + image full-size banners */
.games-grid{display:grid;gap:14px;grid-template-columns:1fr}
@media(min-width:700px){.games-grid{grid-template-columns:1fr 1fr}}
@media(min-width:1020px){.games-grid{grid-template-columns:1fr 1fr 1fr}}

.game-card{border:1px solid var(--border);border-radius:16px;overflow:hidden;cursor:pointer;background:#0b1326;padding:0;position:relative;height:var(--bannerH)}
.game-card .banner{position:absolute;inset:0;display:block;width:100%;height:100%;object-fit:cover}

/* Hero / misc */
.hero{display:flex;justify-content:space-between;align-items:center;gap:14px;flex-wrap:wrap}
.sep{height:1px;background:rgba(255,255,255,.06);margin:10px 0}

/* Crash graph */
.cr-graph-wrap{position:relative;height:240px;background:#0e1833;border:1px solid var(--border);border-radius:16px;overflow:hidden}
canvas{display:block;width:100%;height:100%}

/* Chat (cleaner) */
.chat-drawer{position:fixed;right:0;top:64px;bottom:0;width:var(--chatW);max-width:92vw;transform:translateX(100%);transition:transform .2s ease-out;background:linear-gradient(180deg,#0f1a33,#0b1326);border-left:1px solid var(--border);display:flex;flex-direction:column;z-index:55}
.chat-drawer.open{transform:translateX(0)}
.chat-head{display:flex;align-items:center;justify-content:space-between;padding:10px 12px;border-bottom:1px solid var(--border)}
.chat-body{flex:1;overflow:auto;padding:12px}
.chat-input{display:flex;gap:8px;padding:10px 12px;border-top:1px solid var(--border)}.chat-input input{flex:1}
.msg{margin-bottom:12px;padding:10px;border:1px solid var(--border);border-left:4px solid #3b82f6;border-radius:12px;background:#0b1630}
.msg.role-owner{border-left-color:#22c1dc}
.msg.role-admin{border-left-color:#f59e0b}
.msg.role-moderator{border-left-color:#22c55e}
.msg .msghead{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:6px}
.msg .time{margin-left:auto;color:#9eb3da;font-size:12px}

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

/* Account modal tabs */
.tabbar{display:flex;gap:8px;border:1px solid var(--border);padding:4px;border-radius:12px;background:#0c1631}
.tabbar .t{padding:8px 12px;border-radius:8px;cursor:pointer}
.tabbar .t.active{background:linear-gradient(135deg,#3b82f6,#22c1dc);color:#06121a;font-weight:800}

.grid-2{display:grid;grid-template-columns:1fr;gap:16px}
@media(min-width:900px){.grid-2{grid-template-columns:1.1fr .9fr}}

.lb-controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
.seg{display:flex;border:1px solid var(--border);border-radius:12px;overflow:hidden}
.seg button{padding:8px 12px;background:#0c1631;color:#dce7ff;border:none;cursor:pointer}
.seg button.active{background:linear-gradient(135deg,#3b82f6,#22c1dc);color:#051326;font-weight:800}

/* About Us */
.about .links{display:flex;gap:10px;flex-wrap:wrap}

/* Hide legacy */
#page-profile{display:none!important}
</style>
</head>
<body>

<!-- Preloader -->
<div id="preload"><div class="loader"></div><div class="title" style="margin-top:12px;font-weight:900;letter-spacing:.5px;color:#eaf2ff;opacity:.9">GROWCB</div></div>

<!-- Toasts -->
<div id="toasts"></div>

<div class="header">
  <div class="header-inner container">
    <div class="left">
      <a class="brand" href="/games" id="homeLink">
        <img class="logo" src="/img/GrowCBnobackground.png" alt="GROWCB" />
        <span class="name">GROWCB</span>
      </a>
      <div class="tabs">
        <a class="tab active" id="tab-games" data-path="/games">Games</a>
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
            <div id="
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
  <div class="chat-head"><div>Global Chat <span id="chatNote" class="muted">(Lv 5+)</span></div><button class="btn gray" id="chatClose">Close</button></div>
  <div class="chat-body" id="chatBody"></div>
  <div class="chat-input"><input id="chatText" placeholder="Say something… (Lv 5+, Enter to send)"/><button class="btn primary" id="chatSend">Send</button></div>
</div>

<!-- Profile Modal -->
<div class="modal" id="profileModal">
  <div class="box">
    <div class="head"><div class="big" id="pmTitle">Player</div><button class="btn gray" id="pmClose">Close</button></div>
    <div id="pmBody">Loading…</div>
  </div>
</div>

<!-- Account Modal -->
<div class="modal" id="acctModal">
  <div class="box">
    <div class="head">
      <div class="tabbar">
        <div class="t active" id="acctTabProfile">Profile</div>
        <div class="t" id="acctTabSettings">Settings</div>
      </div>
      <button class="btn gray" id="acctClose">Close</button>
    </div>
    <div id="acctBody">
      <div id="acctProfileView">
        <div class="games-grid" style="grid-template-columns:1fr 1fr 1fr">
          <div class="card"><div class="label">Level</div><div class="big" id="acctLv">—</div><div class="muted"><span id="acctXP">—</span> XP</div></div>
          <div class="card"><div class="label">Balance</div><div class="big"><span id="acctBal">0.00</span> DL</div></div>
          <div class="card"><div class="label">Role</div><div class="big" id="acctRole">member</div></div>
        </div>
        <div class="sep"></div>
        <button class="btn primary" id="acctOwnerBtn" style="display:none">Open Owner Panel</button>
      </div>
      <div id="acctSettingsView" style="display:none">
        <div class="card">
          <div class="label">Privacy</div>
          <div style="display:flex;gap:10px;align-items:center;margin-top:8px">
            <label><input type="checkbox" id="acctAnonChk" /> Anonymous on leaderboards</label>
          </div>
        </div>
        <div class="sep"></div>
        <a class="btn ghost" href="/logout">Logout</a>
      </div>
    </div>
  </div>
</div>

<!-- Login Modal (choose provider + email form) -->
<div class="modal" id="loginModal">
  <div class="box">
    <div class="head"><div class="big">Log in</div><button class="btn gray" id="loginClose">Close</button></div>
    <div class="card">
      <div class="grid-2">
        <div class="card">
          <div class="label">Social</div>
          <div style="display:flex;gap:8px;margin-top:8px;flex-wrap:wrap">
            <a class="btn primary" href="/login">Login with Discord</a>
            <a class="btn ghost" href="/login/google">Login with Google</a>
          </div>
          <div class="muted" style="margin-top:6px">You’ll come back here after you authorize.</div>
        </div>
        <div class="card">
          <div class="label">Email</div>
          <div class="field" style="margin-top:6px"><input id="emEmail" placeholder="you@example.com"/></div>
          <div class="field" style="margin-top:6px"><input id="emPass" type="password" placeholder="Password (min 6)"/></div>
          <div class="field" style="margin-top:6px"><input id="emUser" placeholder="Display name (for first-time)"/></div>
          <div style="display:flex;gap:8px;margin-top:8px;flex-wrap:wrap">
            <button class="btn primary" id="emLoginBtn">Login</button>
            <button class="btn alt" id="emRegisterBtn">Register</button>
            <button class="btn gray" id="emResetBtn" title="Send password reset email">Reset password</button>
          </div>
          <div id="emMsg" class="muted" style="margin-top:6px"></div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- Deposit / Withdraw Modal -->
<div class="modal" id="dwModal">
  <div class="box">
    <div class="head">
      <div class="tabbar">
        <div class="t active" id="dwActionDeposit">Deposit</div>
        <div class="t" id="dwActionWithdraw">Withdraw</div>
      </div>
      <button class="btn gray" id="dwClose">Close</button>
    </div>
    <div class="card">
      <div class="grid-2">
        <div class="card">
          <div class="label">World</div>
          <input id="dwWorld" placeholder="Your GT world"/>
          <div class="label" style="margin-top:8px">GrowID</div>
          <input id="dwGrow" placeholder="Your GrowID"/>
          <div class="label" style="margin-top:8px">Discord (optional)</div>
          <div style="display:flex;gap:8px">
            <input id="dwDiscord" placeholder="Your Discord handle"/>
            <button class="btn ghost" id="dwUseMe">Use my Discord</button>
          </div>
        </div>
        <div class="card">
          <div class="label">Withdraw Amount (DL)</div>
          <input id="dwAmount" type="number" step="0.01" min="1" placeholder="e.g. 10.00"/>
          <div class="muted" style="margin-top:6px">Amount is required for withdraws only.</div>
        </div>
      </div>
      <div class="foot">
        <button class="btn primary" id="dwSubmit">Submit Request</button>
      </div>
    </div>
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

/* --- Robust preloader removal --- */
function hidePreload(){
  const el = qs('preload');
  if(!el) return;
  el.classList.add('hide');
  setTimeout(()=>{ try{ el.remove(); }catch(_){} }, 450);
}
window.addEventListener('load', ()=> setTimeout(hidePreload, 100));
setTimeout(hidePreload, 1500); // fallback

/* Router */
const pages = ['page-games','page-crash','page-mines','page-coinflip','page-blackjack','page-pump','page-ref','page-promo','page-lb','page-about','page-owner'];
const pathToPage = {
  '/games': 'page-games',
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
  const map = {'/games':'tab-games','/':'tab-games','/referral':'tab-ref','/promocodes':'tab-promo','/leaderboard':'tab-lb','/about':'tab-about'};
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
  if(pg==='page-ref') loadReferral();
  if(pg==='page-promo') renderPromo();
  if(pg==='page-lb') refreshLeaderboard();
  if(pg==='page-crash'){ openCrash(); }
  if(pg==='page-mines'){ refreshMines(); }
}
function handleRoute(){
  const path = pathToPage[location.pathname] ? location.pathname : '/games';
  goto(path, true);
}
window.addEventListener('popstate', handleRoute);

/* Header / Auth + Account modal */
async function renderHeader(){
  let me=null, bal=null, prof=null;
  try{ me = await j('/api/me'); }catch(_){}
  const area = qs('authArea');
  if(me){
    try{ bal = await j('/api/balance'); }catch(_){}
    try{ prof = await j('/api/profile'); }catch(_){}
    area.innerHTML = `
      <button class="btn primary" id="btnDW">Deposit / Withdraw</button>
      <button class="btn ghost" id="btnJoinSmall">${me.in_guild ? 'In Discord' : 'Join Discord'}</button>
      <span class="balance-chip" id="balChip">${bal ? dlHtml(bal.balance) : dlHtml(0)}</span>
      <img class="avatar" id="avatarBtn" src="${me.avatar_url||'/img/GrowCBnobackground.png'}" title="${me.username||'user'}"/>
    `;
    qs('btnJoinSmall').onclick = joinDiscord;
    qs('btnDW').onclick = ()=> openDW();
    qs('avatarBtn').onclick = ()=> openAccount(prof, bal);
  }else{
    area.innerHTML = `
      <button class="btn primary" id="btnDW">Deposit / Withdraw</button>
      <button class="btn ghost" id="btnLogin">Log in</button>
    `;
    qs('btnDW').onclick = ()=> openDW();
    qs('btnLogin').onclick = ()=> openLogin();
  }
}
function openLogin(){ qs('loginModal').classList.add('open'); }
qs('loginClose').onclick = ()=> qs('loginModal').classList.remove('open');

/* Email auth actions */
qs('emLoginBtn').onclick = async ()=>{
  const email = (qs('emEmail').value||'').trim(), pwd = qs('emPass').value||'';
  try{
    await j('/auth/login', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ email, password: pwd })});
    qs('emMsg').textContent = 'Logged in!';
    qs('loginModal').classList.remove('open');
    renderHeader();
  }catch(e){ qs('emMsg').textContent = e.message; }
};
qs('emRegisterBtn').onclick = async ()=>{
  const email = (qs('emEmail').value||'').trim(), pwd = qs('emPass').value||'', username = (qs('emUser').value||'').trim();
  try{
    await j('/auth/register', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ email, password: pwd, display_name: username })});
    qs('emMsg').textContent = 'Registered & logged in!';
    qs('loginModal').classList.remove('open');
    renderHeader();
  }catch(e){ qs('emMsg').textContent = e.message; }
};
qs('emResetBtn').onclick = async ()=>{
  const email = (qs('emEmail').value||'').trim();
  if(!email){ qs('emMsg').textContent = 'Enter your email first'; return; }
  try{
    await j('/auth/request_reset', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ email })});
    qs('emMsg').textContent = 'If the email exists, a reset link was sent.';
  }catch(e){ qs('emMsg').textContent = e.message; }
};

function acctTab(sel){
  const prof = qs('acctProfileView'); const set = qs('acctSettingsView');
  const tp = qs('acctTabProfile'); const ts = qs('acctTabSettings');
  tp.classList.toggle('active', sel==='p'); ts.classList.toggle('active', sel==='s');
  prof.style.display = (sel==='p') ? '' : 'none';
  set.style.display = (sel==='s') ? '' : 'none';
}
qs('acctTabProfile').onclick = ()=> acctTab('p');
qs('acctTabSettings').onclick = ()=> acctTab('s');
qs('acctClose').onclick = ()=> qs('acctModal').classList.remove('open');

async function openAccount(prof, bal){
  try{
    if(!prof) prof = await j('/api/profile');
    if(!bal) bal = await j('/api/balance');
    qs('acctLv').textContent = 'Lv ' + (prof.level||1);
    qs('acctXP').textContent = prof.xp||0;
    qs('acctBal').textContent = fmtDL(bal.balance||0);
    qs('acctRole').textContent = prof.role || 'member';
    const isOwner = (String(prof.id) === String(OWNER_ID)) || (prof.role==='owner');
    qs('acctOwnerBtn').style.display = isOwner ? '' : 'none';
    qs('acctOwnerBtn').onclick = ()=> { qs('acctModal').classList.remove('open'); goto('/owner'); };

    const st = await j('/api/settings/get').catch(()=>({is_anon:false}));
    qs('acctAnonChk').checked = !!(st && st.is_anon);

    qs('acctAnonChk').onchange = async ()=>{
      try{
        await j('/api/settings/set_anon',{ method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ is_anon: qs('acctAnonChk').checked })});
        toast('Settings saved', 'success');
      }catch(e){ toast(e.message,'error','Settings'); }
    };

    acctTab('p');
    qs('acctModal').classList.add('open');
  }catch(e){ toast('Please login first','error'); }
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
  const dep = document.getElementById('dwActionDeposit');
  const wdr = document.getElementById('dwActionWithdraw');
  if(dep && wdr){
    dep.classList.toggle('primary', t==='deposit');
    wdr.classList.toggle('primary', t==='withdraw');
    dep.classList.toggle('ghost', t!=='deposit');
    wdr.classList.toggle('ghost', t!=='withdraw');
  }
}
document.getElementById('dwActionDeposit').onclick = ()=> setDWType('deposit');
document.getElementById('dwActionWithdraw').onclick = ()=> setDWType('withdraw');
document.getElementById('dwClose').onclick = ()=> qs('dwModal').classList.remove('open');
document.getElementById('dwUseMe').onclick = async ()=>{
  try{ const me = await j('/api/me'); qs('dwDiscord').value = me.username || me.id || ''; }catch(_){ toast('Not logged in. Enter your Discord username or ID.', 'error', 'Discord'); }
};
document.getElementById('dwSubmit').onclick = async ()=>{
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
    toast(`Request #${r.id} submitted`, 'success', 'Submitted');
    qs('dwModal').classList.remove('open');
  }catch(e){ toast(e.message,'error','Transfer'); }
};

/* Crash logic */
let crPoll=null, crCanvas=null, crCtx=null, crBust=1.0;
function drawCrash(mult){
  if(!crCtx || !crCanvas) return;
  const w = crCanvas.width = crCanvas.clientWidth;
  const h = crCanvas.height = crCanvas.clientHeight;
  crCtx.clearRect(0,0,w,h);
  crCtx.beginPath();
  crCtx.moveTo(0,h-2);
  const maxT = Math.max(1.0, crBust);
  for(let x=0;x<w;x+=4){
    const t = x/w*maxT;
    const m = Math.min(mult, 1.0 + 2*Math.pow(t,1.2));
    const y = h - (h-8) * Math.min(m/Math.max(mult,1.01),1);
    crCtx.lineTo(x,y);
  }
  crCtx.strokeStyle = '#6aa6ff';
  crCtx.lineWidth = 2;
  crCtx.stroke();
}
async function openCrash(){
  qs('crMsg').textContent='';
  qs('crCashout').style.display='none';
  crCanvas = qs('crCanvas');
  crCtx = crCanvas ? crCanvas.getContext('2d') : null;
  await pollCrash(true);
  if(crPoll) clearInterval(crPoll);
  crPoll = setInterval(()=> pollCrash(false), 900);
  window.addEventListener('resize', ()=> drawCrash(1.0));
}
async function pollCrash(first=false){
  try{
    const st = await j('/api/crash/state');
    const nowEl = qs('crNow'); const hintEl = qs('crHint');
    crBust = Number(st.bust||1.0);
    if(st.phase==='betting'){
      nowEl.textContent = '1.00×';
      hintEl.textContent = 'Place your bets…';
      drawCrash(1.0);
    }else if(st.phase==='running'){
      const m = Number(st.current_multiplier||1.0).toFixed(2);
      nowEl.textContent = `${m}×`;
      hintEl.textContent = 'Running… cash out anytime!';
      drawCrash(Number(st.current_multiplier||1.0));
    }else if(st.phase==='ended'){
      nowEl.textContent = `${Number(st.bust||1).toFixed(2)}×`;
      hintEl.textContent = 'Round ended. Next round starting…';
      drawCrash(Number(st.bust||1.0));
    }
    qs('lastBusts').textContent = (st.last_busts||[]).map(x=> Number(x).toFixed(2)+'×').join(' • ') || '—';
    if(st.your_bet && st.phase==='running' && !st.your_bet.cashed_out){
      qs('crCashout').style.display='';
    }else{
      qs('crCashout').style.display='none';
    }
  }catch(e){
    if(first) toast(e.message,'error','Crash');
  }
}
document.getElementById('crPlace').onclick = async ()=>{
  try{
    const bet = String(qs('crBet').value||'');
    const cash = qs('crCash').value ? Number(qs('crCash').value) : null;
    const r = await j('/api/crash/place', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ bet, cashout: cash }) });
    toast('Bet placed','success','Crash');
    qs('crMsg').textContent = `Bet: ${fmtDL(r.bet)} DL  •  Auto: ${Number(r.cashout).toFixed(2)}×`;
    renderHeader();
  }catch(e){ toast(e.message,'error','Crash'); }
};
document.getElementById('crCashout').onclick = async ()=>{
  try{ const r = await j('/api/crash/cashout', { method:'POST' }); toast(`Cashed out at ${Number(r.cashed_out).toFixed(2)}× (+${fmtDL(r.win)} DL)`, 'success', 'Crash'); renderHeader(); }
  catch(e){ toast(e.message,'error','Crash'); }
};
async function loadCrashHistory(){
  try{
    const r = await j('/api/crash/history'); 
    qs('crLast').innerHTML = (r.rows||[]).map(g=>`<div>${new Date(g.created_at).toLocaleString()} — bet ${fmtDL(g.bet)} • bust ${Number(g.bust).toFixed(2)}× • win ${fmtDL(g.win)}</div>`).join('') || '—';
  }catch(_){ qs('crLast').textContent='—'; }
}

/* Mines logic */
let mState=null;
function renderMinesGrid(st){
  const grid = qs('mGrid'); grid.innerHTML='';
  const picks = st.picks||0;
  for(let i=0;i<25;i++){
    const b = document.createElement('button');
    b.className='btn gray';
    b.style.width='64px'; b.style.height='64px';
    b.textContent = (st.revealed && st.revealed[i]===1) ? '💎' : (st.revealed && st.revealed[i]===-1) ? '💣' : (picks & (1n<<BigInt(i))) ? '•' : '';
    b.disabled = !!st.ended || (st.revealed && (st.revealed[i]!==0));
    b.onclick = async ()=>{ try{ const r = await j('/api/mines/pick?index='+i,{method:'POST'}); mState=r; updateMinesUI(); }catch(e){ toast(e.message,'error','Mines'); } };
    grid.appendChild(b);
  }
}
function updateMinesUI(){
  const st = mState; if(!st){ return; }
  qs('mSetup').style.display = st.status==='active' ? 'none' : 'none';
  qs('mCash').style.display = (st.status==='active' ? '' : 'none');
  qs('mMult').textContent = (st.multiplier||1).toFixed(4)+'×';
  qs('mPotential').textContent = st.potential ? fmtDL(st.potential)+' DL' : '—';
  qs('mHash').textContent = 'Commit: ' + (st.commit_hash||'—');
  qs('mStatus').textContent = 'Status: ' + (st.status||'—');
  qs('mPicks').textContent = 'Picks: ' + (st.pick_count||0);
  qs('mBombs').textContent = 'Mines: ' + (st.mines||0);
  renderMinesGrid(st);
}
async function refreshMines(){
  try{
    const st = await j('/api/mines/state'); 
    if(st && st.status){
      mState = st; updateMinesUI();
    }else{
      qs('mSetup').style.display='';
      qs('mCash').style.display='none';
      qs('mGrid').innerHTML='';
      qs('mMult').textContent='1.0000×';
      qs('mPotential').textContent='—';
      qs('mHash').textContent='Commit: —';
      qs('mStatus').textContent='Status: —';
      qs('mPicks').textContent='Picks: 0';
      qs('mBombs').textContent='Mines: ' + (qs('mMines').value||3);
    }
    try{
      const h = await j('/api/mines/history');
      qs('mHist').innerHTML = (h.rows||[]).map(r=>`<div>${new Date(r.started_at).toLocaleString()} — bet ${fmtDL(r.bet)} • mines ${r.mines} • win ${fmtDL(r.win)}</div>`).join('') || '—';
    }catch(_){}
  }catch(e){ toast(e.message,'error','Mines'); }
}
document.getElementById('mStart').onclick = async ()=>{
  try{
    const bet = String(qs('mBet').value||'');
    const mines = parseInt(qs('mMines').value||'3',10);
    const st = await j('/api/mines/start', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ bet, mines }) });
    mState = st; updateMinesUI(); toast('Game started!','success','Mines');
    renderHeader();
  }catch(e){ toast(e.message,'error','Mines'); }
};
document.getElementById('mCash').onclick = async ()=>{
  try{ const r = await j('/api/mines/cashout',{method:'POST'}); toast(`Cashed out +${fmtDL(r.win)} DL`, 'success', 'Mines'); mState=r; updateMinesUI(); renderHeader(); }
  catch(e){ toast(e.message,'error','Mines'); }
};

/* Referral */
function refLinkFrom(name){
  const host = location.origin.replace(/\/$/,'');
  return `${REF_BASE.replace(/\/$/,'')}/${encodeURIComponent(name||'')}`;
}
async function loadReferral(){
  try{
    const r = await j('/api/referral/state');
    const name = r.name || '';
    qs('refName').value = name;
    qs('refClicks').textContent = r.clicks||0;
    qs('refJoins').textContent = r.joined||0;
    qs('refLink').value = name ? refLinkFrom(name) : '';
    qs('copyRef').onclick = ()=>{
      if(!qs('refLink').value){ toast('Set a handle first','error','Referral'); return; }
      navigator.clipboard.writeText(qs('refLink').value).then(()=> toast('Link copied!','success','Referral'));
    };
    qs('refSave').onclick = async ()=>{
      const name = (qs('refName').value||'').trim();
      if(!name){ toast('Enter a handle','error','Referral'); return; }
      try{
        const r = await j('/api/referral/set',{ method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ name }) });
        qs('refLink').value = refLinkFrom(r.name||name);
        toast('Saved','success','Referral');
      }catch(e){
        qs('refMsg').textContent = e.message || 'Could not save';
        toast(e.message,'error','Referral');
      }
    };
  }catch(e){
    toast(e.message || 'Referral unavailable','error','Referral');
  }
}

/* Promo Codes */
async function renderPromo(){
  try{
    const list = await j('/api/promo/my').catch(()=>({rows:[]}));
    const rows = (list.rows||[]).map(r=>`<div>${r.code} — ${new Date(r.redeemed_at).toLocaleString()}</div>`).join('') || '—';
    qs('myCodes').innerHTML = rows;
  }catch(_){}
  const btn = qs('redeemBtn');
  btn.onclick = async ()=>{
    const code = (qs('promoInput').value||'').trim();
    if(!code){ toast('Enter a code','error','Promo'); return; }
    try{
      const res = await j('/api/promo/redeem', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ code })});
      toast('Code redeemed!','success','Promo');
      renderHeader();
      renderPromo();
    }catch(e){
      qs('promoMsg').textContent = e.message || 'Invalid code';
      toast(e.message,'error','Promo');
    }
  };
}

/* Leaderboard */
let lbPeriod='daily';
function lbResetText(period){
  const now = new Date();
  if(period==='daily'){
    const nxt = new Date(now); nxt.setUTCHours(24,0,0,0);
    const ms = (+nxt) - (+now);
    const h = Math.floor(ms/3600000), m = Math.floor((ms%3600000)/60000);
    return `Resets in ${h}h ${m}m`;
  }
  if(period==='monthly'){
    const nxt = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth()+1, 1, 0,0,0,0));
    const ms = (+nxt) - (+now);
    const d = Math.floor(ms/86400000), h = Math.floor((ms%86400000)/3600000);
    return `Resets in ${d}d ${h}h`;
  }
  return 'All-time';
}
async function refreshLeaderboard(){
  try{
    const r = await j('/api/leaderboard?period='+encodeURIComponent(lbPeriod));
    const rows = (r.rows||[]);
    qs('lbWrap').innerHTML = rows.map((r,i)=>{
      const name = r.is_anon ? 'Anonymous' : r.display_name;
      const amt = r.is_anon ? '—' : fmtDL(r.total_wagered) + ' DL';
      return `<div style="display:flex;gap:10px;align-items:center;padding:8px 0;border-bottom:1px solid var(--border)">
        <div style="width:28px;text-align:right;font-weight:800">${i+1}.</div>
        <div style="flex:1">${name}</div>
        <div style="font-weight:900">${amt}</div>
      </div>`;
    }).join('') || '—';
    qs('lbCountdown').textContent = lbResetText(lbPeriod);
  }catch(e){ qs('lbWrap').textContent = e.message || 'Error'; }
  qsa('#lbSeg button').forEach(b=>{
    b.onclick = ()=>{ qsa('#lbSeg button').forEach(x=>x.classList.remove('active')); b.classList.add('active'); lbPeriod = b.dataset.period; refreshLeaderboard(); };
  });
}

/* Chat */
let chatOpen=false, chatPoll=null, chatLast=0;
function toggleChat(open){
  chatOpen=open;
  qs('chatDrawer').classList.toggle('open', open);
  if(open){
    pollChat(true);
    if(chatPoll) clearInterval(chatPoll);
    chatPoll = setInterval(()=> pollChat(false), 2500);
  }else{
    if(chatPoll) clearInterval(chatPoll);
  }
}
async function pollChat(initial){
  try{
    const r = await j('/api/chat/fetch?since='+chatLast+'&limit=50');
    const rows = r.rows||[];
    const body = qs('chatBody');
    for(const m of rows){
      chatLast = Math.max(chatLast, m.id||0);
      const div = document.createElement('div');
      const roleClass = 'role-'+(m.role||'member');
      const badge = `<span class="badge ${m.role||'member'}">${m.role||'member'}</span>`;
      const lvl = `<span class="level">Lv ${m.level||1}</span>`;
      const user = `<span class="user-link" data-user="${m.user_id}">${(m.username||'user')}</span>`;
      div.className = 'msg '+roleClass;
      div.innerHTML = `<div class="msghead">${user} ${lvl} ${badge}<span class="time">${new Date(m.created_at).toLocaleTimeString()}</span></div>
        <div class="txt"></div>`;
      div.querySelector('.txt').innerText = m.text||'';
      body.appendChild(div);
    }
    if(rows.length>0) body.scrollTop = body.scrollHeight;
    qsa('.user-link').forEach(u=>{
      if(u.dataset.bind) return;
      u.dataset.bind='1';
      u.onclick = ()=> openProfile(u.dataset.user);
    });
  }catch(_){}
}
async function openProfile(user_id){
  qs('pmBody').textContent='Loading…';
  qs('profileModal').classList.add('open');
  try{
    const p = await j('/api/profile/public?user_id='+encodeURIComponent(user_id));
    const body = `
      <div class="games-grid" style="grid-template-columns:1fr 1fr 1fr">
        <div class="card"><div class="label">Level</div><div class="big">Lv ${p.level||1}</div><div class="muted">${p.xp||0} XP</div></div>
        <div class="card"><div class="label">Balance</div><div class="big">${fmtDL(p.balance||0)} DL</div></div>
        <div class="card"><div class="label">Role</div><div class="big">${p.role||'member'}</div></div>
      </div>
      <div class="sep"></div>
      <div class="muted">Crash games: ${p.crash_games||0} • Mines games: ${p.mines_games||0}</div>
    `;
    qs('pmTitle').textContent = p.name || 'Player';
    qs('pmBody').innerHTML = body;
  }catch(e){
    qs('pmBody').textContent = e.message || 'Error';
  }
}
qs('pmClose').onclick = ()=> qs('profileModal').classList.remove('open');

qs('fabChat').onclick = ()=> toggleChat(true);
qs('chatClose').onclick = ()=> toggleChat(false);
qs('chatText').addEventListener('keydown', e=>{ if(e.key==='Enter') qs('chatSend').click(); });
qs('chatSend').onclick = async ()=>{
  const t = (qs('chatText').value||'').trim();
  if(!t) return;
  try{
    await j('/api/chat/send',{ method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ text: t })});
    qs('chatText').value='';
    pollChat(false);
  }catch(e){ toast(e.message || 'Could not send','error','Chat'); }
};

/* Owner panel controls */
async function bindOwnerPanel(){
  const op = qs('page-owner');
  if(!op) return;
  const a = id=> op.querySelector('#'+id);
  a('opApply').onclick = async ()=>{
    const identifier = a('opIdent').value||''; const amount = a('opAmt').value||''; const reason = a('opReason').value||'';
    if(!identifier || !amount){ toast('Need target and amount','error','Adjust'); return; }
    try{
      const r = await j('/api/admin/adjust',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({identifier,amount,reason})});
      a('opMsg').textContent = 'New balance: '+fmtDL(r.new_balance)+' DL';
      toast('Adjusted','success','Admin');
    }catch(e){ a('opMsg').textContent=e.message; toast(e.message,'error','Admin'); }
  };
  a('roleApply').onclick = async ()=>{
    const identifier = a('roleIdent').value||''; const role = a('roleSelect').value||'member';
    try{
      const r = await j('/api/admin/role',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({identifier,role})});
      a('roleMsg').textContent = 'Role set to '+(r.role||role);
      toast('Role updated','success','Admin');
    }catch(e){ a('roleMsg').textContent=e.message; toast(e.message,'error','Admin'); }
  };
  a('announceBtn').onclick = async ()=>{
    const text = a('announceTxt').value||'';
    if(!text){ toast('Write something to announce','error','Announce'); return; }
    try{
      await j('/api/admin/announce',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text, minutes:360})});
      a('announceTxt').value='';
      toast('Announcement posted','success','Announce');
    }catch(e){ toast(e.message,'error','Announce'); }
  };
}

/* Navigation wiring */
function wireNav(){
  qsa('.tabs .tab').forEach(t=>{
    t.onclick = ()=> goto(t.getAttribute('data-path')||'/games');
  });
  const bindCard = (id, path)=>{ const el = qs(id); if(el) el.onclick = ()=> goto(path); };
  bindCard('openCrash','/crash');
  bindCard('openMines','/mines');
  bindCard('openCoinflip','/coinflip');
  bindCard('openBlackjack','/blackjack');
  bindCard('openPump','/pump');

  const bindBack = (id)=>{ const el = qs(id); if(el) el.onclick = ()=> goto('/games'); };
  bindBack('backToGames');
  bindBack('backToGames2');
  bindBack('backToGames_cf');
  bindBack('backToGames_bj');
  bindBack('backToGames_pu');

  const home = qs('homeLink');
  if(home) home.onclick = (e)=>{ e.preventDefault(); goto('/games'); };
}

/* Initial boot */
(async function boot(){
  try{
    wireNav();
    renderHeader();
    handleRoute();
    bindOwnerPanel();
    try{
      const refCookie = (document.cookie.split('; ').find(x=> x.startsWith('refname='))||'').split('=').slice(1).join('=');
      if(refCookie){
        await j('/api/referral/attach?refname='+encodeURIComponent(decodeURIComponent(refCookie)));
        document.cookie = 'refname=; Max-Age=0; path=/; samesite=lax';
      }
    }catch(_){}
  }finally{
    hidePreload();
  }
})();
</script>
</body>
</html>
"""

# ---------- HTML augmentation (announcements + polish) ----------
EXTRA_SNIPPET = r"""
<script>
(function(){
  const SEEN_KEY = 'ann_seen_ids_v1';
  const seen = new Set((localStorage.getItem(SEEN_KEY)||'').split(',').filter(Boolean).map(x=>String(x)));
  async function poll(){
    try{
      const r = await fetch('/api/announcements/active'); if(!r.ok) throw new Error('net');
      const js = await r.json();
      const rows = (js && js.rows)||[];
      for(const a of rows){
        const id = String(a.id);
        if(!seen.has(id)){
          const box = document.createElement('div');
          box.className='toast';
          box.style.borderLeftColor='#6aa6ff';
          box.innerHTML = `<div class="t">Announcement</div><div class="m">${(a.text||'').replace(/[<>&]/g, m=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[m]))}</div>`;
          const host = document.getElementById('toasts'); host.appendChild(box);
          setTimeout(()=>{ box.style.opacity='0'; setTimeout(()=> box.remove(), 250); }, 6000);
          seen.add(id);
        }
      }
      localStorage.setItem(SEEN_KEY, Array.from(seen).join(','));
    }catch(_){}
    finally{ setTimeout(poll, 25000); }
  }
  setTimeout(poll, 2000);
})();
</script>
"""
HTML_TEMPLATE = HTML_TEMPLATE.replace("</body>", EXTRA_SNIPPET + "\n</body>")
HTML_TEMPLATE = HTML_TEMPLATE.replace(
    "GROWCB is a community project offering fun, provably-fair mini-games like Crash and Mines, with Discord integration.",
    "GROWCB is a sleek, community-driven arcade with provably-fair mini-games (Crash, Mines and more), cosmetic polish, and tight Discord integration. We’re building a smooth, low-friction experience: quick rounds, clear payouts, no-nonsense UI."
)

# ---------- SPA roots (serve same index for these paths) ----------
SPA_PATHS = {"", "games", "crash", "mines", "coinflip", "blackjack", "pump", "referral", "promocodes", "leaderboard", "about", "owner"}

@app.get("/", response_class=HTMLResponse)
async def index_root():
    html = HTML_TEMPLATE.replace("__INVITE__", DISCORD_INVITE or "__INVITE__") \
                        .replace("__OWNER_ID__", str(OWNER_ID)) \
                        .replace("__REF_BASE__", os.getenv("REFERRAL_SHARE_BASE", "https://growcb.net/referral"))
    return HTMLResponse(html)

@app.get("/{path_name}", response_class=HTMLResponse)
async def spa_paths(path_name: str):
    if path_name in SPA_PATHS or path_name.strip("/") in SPA_PATHS:
        return await index_root()
    raise HTTPException(404, "Not found")

# ---------- Leaderboard API ----------
@app.get("/api/leaderboard")
async def api_leaderboard(period: str = Query("daily"), limit: int = Query(50, ge=1, le=200)):
    rows = get_leaderboard_rows_db(period, limit)
    return {"rows": rows}

# ---------- OAuth / Auth ----------
@app.get("/login")
async def login():
    if not (CLIENT_ID and OAUTH_REDIRECT):
        return HTMLResponse("Discord OAuth not configured")
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
        return HTMLResponse("Discord OAuth not configured")
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
        u = await cx.get(f"{DISCORD_API}/users/@me", headers={"Authorization": f"Bearer {access}"})
        if u.status_code != 200:
            return HTMLResponse(f"User fetch failed: {u.text}", status_code=400)
        me = u.json()

    user_id = str(me["id"])
    username = f'{me.get("username","user")}#{me.get("discriminator","0")}'.replace("#0","")
    avatar_hash = me.get("avatar")
    avatar_url = f"https://cdn.discordapp.com/avatars/{user_id}/{avatar_hash}.png?size=128" if avatar_hash else "https://cdn.discordapp.com/embed/avatars/0.png"

    ensure_profile_row(user_id)
    ensure_account_row(user_id, 'discord', user_id, None, None)
    save_tokens(user_id, tok.get("access_token",""), tok.get("refresh_token"), tok.get("expires_in"))

    resp = RedirectResponse("/games")
    _set_session(resp, {"id": user_id, "username": username, "avatar_url": avatar_url}, request)
    return resp

# ---- Google OAuth ----
@app.get("/login/google")
async def login_google():
    if not (GOOGLE_CLIENT_ID and GOOGLE_REDIRECT):
        return HTMLResponse("Google OAuth not configured")
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "consent"
    }
    return RedirectResponse(f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}")

@app.get("/callback/google")
async def callback_google(request: Request, code: str):
    if not (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET and GOOGLE_REDIRECT):
        return HTMLResponse("Google OAuth not configured")
    async with httpx.AsyncClient(timeout=15) as cx:
        data = {
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": GOOGLE_REDIRECT
        }
        r = await cx.post("https://oauth2.googleapis.com/token", data=data)
        if r.status_code != 200:
            return HTMLResponse(f"Google token failed: {r.text}", status_code=400)
        tok = r.json()
        at = tok.get("access_token")
        u = await cx.get("https://www.googleapis.com/oauth2/v3/userinfo", headers={"Authorization": f"Bearer {at}"})
        if u.status_code != 200:
            return HTMLResponse(f"Google userinfo failed: {u.text}", status_code=400)
        me = u.json()
    sub = me.get("sub")
    email = me.get("email")
    user_id = f"g_{sub}"
    username = me.get("name") or (email or "user")
    avatar_url = me.get("picture") or "/img/GrowCBnobackground.png"

    ensure_profile_row(user_id)
    ensure_account_row(user_id, 'google', sub, email, None)

    resp = RedirectResponse("/games")
    _set_session(resp, {"id": user_id, "username": username, "avatar_url": avatar_url}, request)
    return resp

# ---- Email/password auth ----
class EmailRegisterIn(BaseModel):
    email: str
    password: str
    display_name: Optional[str] = None

class EmailLoginIn(BaseModel):
    email: str
    password: str

class ResetReqIn(BaseModel):
    email: str

class ResetApplyIn(BaseModel):
    email: str
    token: str
    new_password: str

def _normalize_email(e: str) -> str:
    return (e or "").strip().lower()

def _user_id_from_email(email: str) -> str:
    digest = hashlib.sha256(_normalize_email(email).encode()).hexdigest()[:16]
    return f"e_{digest}"

@with_conn
def _get_account_by_email(cur, email: str):
    cur.execute("SELECT user_id, password_hash FROM accounts WHERE email=%s", (email,))
    r = cur.fetchone()
    return (r[0], r[1]) if r else (None, None)

@app.post("/auth/register")
async def auth_register(request: Request, body: EmailRegisterIn):
    email = _normalize_email(body.email)
    if not email or "@" not in email:
        raise HTTPException(400, "Invalid email")
    if not body.password or len(body.password) < 6:
        raise HTTPException(400, "Password too short")
    user_id, _ = _get_account_by_email(email)
    if user_id:
        raise HTTPException(400, "Email already registered")
    user_id = _user_id_from_email(email)
    display = (body.display_name or f"user_{user_id[-4:]}").strip()
    ensure_profile_row(user_id)
    ensure_account_row(user_id, 'email', email, email, _hash_password(body.password))
    with psycopg.connect(DATABASE_URL) as con, con.cursor() as cur:
        cur.execute("UPDATE profiles SET display_name=%s, name_lower=%s WHERE user_id=%s", (display, display.lower(), user_id))
        con.commit()
    resp = RedirectResponse("/games")
    _set_session(resp, {"id": user_id, "username": display, "avatar_url": "/img/GrowCBnobackground.png"}, request)
    return resp

@app.post("/auth/login")
async def auth_login(request: Request, body: EmailLoginIn):
    email = _normalize_email(body.email)
    user_id, pwh = _get_account_by_email(email)
    if not user_id or not pwh or not _verify_password(pwh, body.password):
        raise HTTPException(401, "Invalid email or password")
    name = get_profile_name(user_id) or f"user_{user_id[-4:]}"
    resp = RedirectResponse("/games")
    _set_session(resp, {"id": user_id, "username": name, "avatar_url": "/img/GrowCBnobackground.png"}, request)
    return resp

@app.post("/auth/request_reset")
async def auth_request_reset(body: ResetReqIn):
    email = _normalize_email(body.email)
    user_id, _ = _get_account_by_email(email)
    if not user_id:
        # Do not reveal; pretend success
        return {"ok": True}
    token = secrets.token_urlsafe(32)
    expires = now_utc() + datetime.timedelta(minutes=30)
    with psycopg.connect(DATABASE_URL) as con, con.cursor() as cur:
        cur.execute("UPDATE accounts SET reset_token=%s, reset_expires=%s WHERE user_id=%s", (token, expires, user_id))
        con.commit()
    link = f"{RESET_LINK_BASE}?email={urlencode({'e':email})[2:]}&token={token}"
    _send_email(email, "Reset your GROWCB password", f"Hi,\n\nClick to reset your password:\n{link}\n\nThis link expires in 30 minutes.\n")
    return {"ok": True}

@app.get("/reset", response_class=HTMLResponse)
async def reset_form(email: str = Query(""), token: str = Query("")):
    html = f"""
    <!doctype html><meta charset="utf-8"/><title>Reset Password</title>
    <style>body{{background:#0a0f1e;color:#eaf2ff;font-family:Inter,system-ui;display:grid;place-items:center;height:100vh;margin:0}}
    .card{{background:#0f1a33;border:1px solid #1f2b47;border-radius:16px;padding:16px;width:min(480px,92vw)}}</style>
    <div class="card">
      <h2>Reset Password</h2>
      <form method="post" action="/auth/reset">
        <input type="hidden" name="email" value="{email}"/>
        <input type="hidden" name="token" value="{token}"/>
        <div>New password</div>
        <input name="new_password" type="password" minlength="6" style="width:100%;padding:8px;margin:8px 0"/>
        <button type="submit" style="padding:10px 14px;border-radius:10px">Set new password</button>
      </form>
    </div>
    """
    return HTMLResponse(html)

@app.post("/auth/reset")
async def auth_reset(email: str = Body(...), token: str = Body(...), new_password: str = Body(...)):
    if not new_password or len(new_password) < 6:
        raise HTTPException(400, "Password too short")
    email = _normalize_email(email)
    with psycopg.connect(DATABASE_URL) as con, con.cursor() as cur:
        cur.execute("SELECT user_id, reset_token, reset_expires FROM accounts WHERE email=%s", (email,))
        r = cur.fetchone()
        if not r or not r[1] or r[1] != token or not r[2] or r[2] < now_utc():
            raise HTTPException(400, "Invalid or expired reset token")
        cur.execute("UPDATE accounts SET password_hash=%s, reset_token=NULL, reset_expires=NULL WHERE user_id=%s", (_hash_password(new_password), r[0]))
        con.commit()
    return RedirectResponse("/games", status_code=303)

@app.get("/logout")
async def logout(request: Request):
    resp = RedirectResponse("/games")
    _clear_session(resp, request)
    return resp

# ---------- Token store & guild join helpers ----------
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

# ---------- Me / Balance / Profiles ----------
@app.get("/api/me")
async def api_me(request: Request):
    s = None
    try:
        s = _require_session(request)
    except Exception:
        pass
    if not s:
        return None
    in_guild = False
    if DISCORD_BOT_TOKEN and GUILD_ID and str(s["id"]).isdigit():
        try:
            async with httpx.AsyncClient(timeout=8) as cx:
                r = await cx.get(f"{DISCORD_API}/guilds/{GUILD_ID}/members/{s['id']}",
                                 headers={"Authorization": f"Bot {DISCORD_BOT_TOKEN}"})
                in_guild = (r.status_code == 200)
        except:
            in_guild = False
    return {"id": s["id"], "username": s.get("username","user"), "avatar_url": s.get("avatar_url"), "in_guild": in_guild}

@app.get("/api/balance")
async def api_balance(request: Request):
    s = _require_session(request)
    return {"balance": float(get_balance(s["id"]))}

@app.get("/api/profile/public")
async def api_profile_public(user_id: str):
    prof = public_profile(user_id)
    if not prof: raise HTTPException(404, "User not found")
    return prof

@app.get("/api/profile")
async def api_profile(request: Request):
    s = _require_session(request)
    return profile_info(s["id"])

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
      location.href = "/games";
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
        cur.execute("""
            SELECT pr.code, pr.redeemed_at
            FROM promo_redemptions pr
            WHERE pr.user_id=%s
            ORDER BY pr.redeemed_at DESC
            LIMIT 50
        """, (s["id"],))
        rows = [{"code": r[0], "redeemed_at": r[1]} for r in cur.fetchall()]
    return {"rows": rows}

@app.post("/api/promo/redeem")
async def api_promo_redeem(request: Request, body: PromoIn):
    s = _require_session(request)
    code = (body.code or "").strip()
    if not code:
        raise HTTPException(400, "Enter a code")
    try:
        amt = redeem_promo(s["id"], code)
        if amt:
            new_bal = adjust_balance(s["id"], s["id"], amt, f"promo:{code}")
            return {"ok": True, "win": float(q2(D(amt))), "balance": float(new_bal)}
        return {"ok": True}
    except PromoAlreadyRedeemed:
        raise HTTPException(400, "You already redeemed this code")
    except PromoExpired:
        raise HTTPException(400, "Code expired")
    except PromoExhausted:
        raise HTTPException(400, "Code has reached maximum uses")
    except PromoInvalid as e:
        raise HTTPException(400, str(e) or "Invalid code")

# ---------- Chat ----------
class ChatSendIn(BaseModel):
    text: str

def _role_rank(role: str) -> int:
    order = {"member":0,"media":1,"moderator":2,"admin":3,"owner":4}
    return order.get((role or "member"), 0)

@with_conn
def _is_timed_out(cur, user_id: str) -> Optional[datetime.datetime]:
    cur.execute("SELECT until FROM chat_timeouts WHERE user_id=%s", (user_id,))
    r = cur.fetchone()
    if not r:
        return None
    until = r[0]
    if until and until > now_utc():
        return until
    cur.execute("DELETE FROM chat_timeouts WHERE user_id=%s", (user_id,))
    return None

@app.get("/api/chat/fetch")
async def api_chat_fetch(since: int = 0, limit: int = 50):
    limit = max(1, min(200, limit))
    with psycopg.connect(DATABASE_URL) as con, con.cursor() as cur:
        cur.execute("""
            SELECT m.id, m.user_id, m.username, m.text, m.created_at,
                   COALESCE(p.role,'member') AS role,
                   (1 + COALESCE(p.xp,0)/100) AS level
            FROM chat_messages m
            LEFT JOIN profiles p ON p.user_id = m.user_id
            WHERE m.id > %s
            ORDER BY m.id ASC
            LIMIT %s
        """, (since, limit))
        rows = cur.fetchall()
    out = []
    for r in rows:
        out.append({
            "id": int(r[0]), "user_id": str(r[1]), "username": r[2],
            "text": r[3], "created_at": r[4].isoformat(),
            "role": r[5], "level": int(r[6]),
        })
    return {"rows": out}

@app.post("/api/chat/send")
async def api_chat_send(request: Request, body: ChatSendIn):
    s = _require_session(request)
    txt = (body.text or "").strip()
    if not txt:
        raise HTTPException(400, "Say something first")
    if len(txt) > 400:
        raise HTTPException(400, "Message too long (max 400)")
    info = profile_info(s["id"])
    if (info.get("level") or 1) < 5:
        raise HTTPException(403, "Level 5+ required to chat")
    with psycopg.connect(DATABASE_URL) as con, con.cursor() as cur:
        if _is_timed_out(cur, s["id"]):
            raise HTTPException(403, "You're temporarily muted")
        cur.execute("""
            INSERT INTO chat_messages(user_id, username, text) VALUES (%s,%s,%s)
            RETURNING id, created_at
        """, (s["id"], info.get("name") or s["username"], txt))
        mid, created = cur.fetchone()
        con.commit()
    return {"ok": True, "id": int(mid), "created_at": created.isoformat()}

# ---------- Discord join ----------
@app.post("/api/discord/join")
async def api_discord_join(request: Request):
    s = _require_session(request)
    try:
        nick = get_profile_name(s["id"]) or None
    except Exception:
        nick = None
    res = await guild_add_member(s["id"], nick)
    return res or {"ok": True}

# ---------- Discord channel notify ----------
async def discord_send(channel_id: int, content: str):
    if not (DISCORD_BOT_TOKEN and channel_id):
        return
    try:
        async with httpx.AsyncClient(timeout=10) as cx:
            await cx.post(f"{DISCORD_API}/channels/{channel_id}/messages",
                          headers={"Authorization": f"Bot {DISCORD_BOT_TOKEN}"},
                          json={"content": content})
    except Exception as e:
        print("Discord send error:", e)

# ---------- Transfers ----------
class TransferIn(BaseModel):
    ttype: str
    amount: Optional[str] = None
    world: str
    grow_id: str

@app.post("/api/transfer/create")
async def api_transfer_create(request: Request, body: TransferIn):
    s = _require_session(request)
    ttype = (body.ttype or "").lower()
    world = (body.world or "").strip()
    grow_id = (body.grow_id or "").strip()
    if ttype not in ("deposit","withdraw"):
        raise HTTPException(400, "Invalid type")
    if not world:
        raise HTTPException(400, "World required")
    if not grow_id:
        raise HTTPException(400, "GrowID required")
    amt = None
    if ttype == "withdraw":
        try:
            amt = q2(D(body.amount or "0"))
        except Exception:
            raise HTTPException(400, "Invalid amount")
        if amt <= 0:
            raise HTTPException(400, "Amount must be positive")
        bal = get_balance(s["id"])
        if bal < amt:
            raise HTTPException(400, "Insufficient balance")
        adjust_balance(s["id"], s["id"], -amt, "withdraw request lock")
    with psycopg.connect(DATABASE_URL) as con, con.cursor() as cur:
        cur.execute("""
            INSERT INTO transfers(user_id, ttype, amount, world, grow_id, status)
            VALUES (%s,%s,%s,%s,%s,'pending') RETURNING id
        """, (s["id"], ttype, amt, world, grow_id))
        new_id = cur.fetchone()[0]
        con.commit()
    # notify Discord
    try:
        if ttype == "deposit" and DEPOSIT_CHANNEL_ID:
            await discord_send(DEPOSIT_CHANNEL_ID, f"💰 **Deposit** request #{new_id} by `{s['id']}` — World: **{world}**, GrowID: **{grow_id}**")
        if ttype == "withdraw" and WITHDRAW_CHANNEL_ID:
            await discord_send(WITHDRAW_CHANNEL_ID, f"🏧 **Withdraw** request #{new_id} by `{s['id']}` — Amount: **{fmtDL(amt)} DL**, World: **{world}**, GrowID: **{grow_id}**")
    except Exception as e:
        print("notify err:", e)
    return {"ok": True, "id": int(new_id)}

# ---------- Crash endpoints ----------
class CrashPlaceIn(BaseModel):
    bet: str
    cashout: Optional[float] = None

@app.get("/api/crash/state")
async def api_crash_state(request: Request):
    try:
        state = load_round()
    except Exception as e:
        raise HTTPException(500, f"Crash unavailable: {e}")
    phase = state.get("status")
    resp = {"phase": phase}
    if phase == "running":
        try: resp["current_multiplier"] = float(current_multiplier())
        except Exception: resp["current_multiplier"] = 1.0
    if phase in ("ended","running"):
        try: resp["bust"] = float(state.get("bust") or 1.0)
        except Exception: resp["bust"] = 1.0
    try:
        resp["last_busts"] = [float(x) for x in last_busts(8)]
    except Exception:
        resp["last_busts"] = []
    try:
        s = None
        try: s = _require_session(request)
        except Exception: s = None
        if s:
            yb = your_bet(s["id"])
            if yb: resp["your_bet"] = yb
    except Exception:
        pass
    return resp

@app.post("/api/crash/place")
async def api_crash_place(request: Request, body: CrashPlaceIn):
    s = _require_session(request)
    try:
        bet = q2(D(body.bet))
    except Exception:
        raise HTTPException(400, "Invalid bet")
    if bet < MIN_BET or bet > MAX_BET:
        raise HTTPException(400, f"Bet must be between {MIN_BET} and {MAX_BET}")
    try:
        ensure_betting_round()
        r = place_bet(s["id"], bet, (float(body.cashout) if body.cashout else None))
        return {"ok": True, **r}
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/api/crash/cashout")
async def api_crash_cashout(request: Request):
    s = _require_session(request)
    try:
        r = cashout_now(s["id"])
        return r
    except Exception as e:
        raise HTTPException(400, str(e))

@app.get("/api/crash/history")
async def api_crash_history(request: Request):
    s = _require_session(request)
    try:
        rows = your_history(s["id"], limit=20)
        return {"rows": rows}
    except Exception:
        return {"rows": []}

# ---------- Mines endpoints ----------
class MinesStartIn(BaseModel):
    bet: str
    mines: int

@app.get("/api/mines/state")
async def api_mines_state(request: Request):
    s = _require_session(request)
    try:
        st = mines_state(s["id"])
        return st or {}
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/api/mines/start")
async def api_mines_start(request: Request, body: MinesStartIn):
    s = _require_session(request)
    try:
        bet = q2(D(body.bet))
    except Exception:
        raise HTTPException(400, "Invalid bet")
    if bet < MIN_BET or bet > MAX_BET:
        raise HTTPException(400, f"Bet must be between {MIN_BET} and {MAX_BET}")
    mines = int(body.mines or 3)
    if not (1 <= mines <= 24):
        raise HTTPException(400, "Mines must be between 1 and 24")
    try:
        st = mines_start(s["id"], bet, mines)
        return st
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/api/mines/pick")
async def api_mines_pick(request: Request, index: int = Query(..., ge=0, le=24)):
    s = _require_session(request)
    try:
        st = mines_pick(s["id"], index)
        return st
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/api/mines/cashout")
async def api_mines_cashout(request: Request):
    s = _require_session(request)
    try:
        st = mines_cashout(s["id"])
        return st
    except Exception as e:
        raise HTTPException(400, str(e))

@app.get("/api/mines/history")
async def api_mines_history(request: Request):
    s = _require_session(request)
    try:
        rows = mines_history(s["id"], limit=20)
        return {"rows": rows}
    except Exception:
        return {"rows": []}

# ---------- Admin helpers / endpoints ----------

def fmtDL(x) -> str:
    """Format an amount as 2dp string safely even for Decimals."""
    try:
        return format(q2(D(x)), ".2f")
    except Exception:
        return "0.00"

def _require_role(request: Request, min_role: str) -> Tuple[dict, dict]:
    """Ensure the calling user has at least min_role; returns (session, profile_info)."""
    s = _require_session(request)
    prof = profile_info(s["id"])
    if _role_rank(prof.get("role") or "member") < _role_rank(min_role):
        raise HTTPException(403, "Insufficient permissions")
    return s, prof

@with_conn
def _resolve_identifier(cur, identifier: str) -> Optional[str]:
    """
    Accepts:
      - raw user id (e_*, g_* or discord numeric id)
      - discord mention like <@123456789> or <@!123456789>
      - exact referral/profile handle (name_lower)
    Returns user_id or None.
    """
    if not identifier:
        return None
    ident = identifier.strip()

    # Discord mention
    m = re.match(r"^<@!?(?P<id>\d+)>$", ident)
    if m:
        return m.group("id")

    # If looks like a user id we already store
    if re.fullmatch(r"(e|g)_[0-9a-f]{16}", ident) or re.fullmatch(r"\d{5,20}", ident):
        return ident

    # Otherwise, treat as handle
    lower = ident.lower()
    cur.execute("SELECT user_id FROM profiles WHERE name_lower=%s", (lower,))
    r = cur.fetchone()
    return str(r[0]) if r else None

class AdminAdjustIn(BaseModel):
    identifier: str
    amount: str
    reason: Optional[str] = None

@app.post("/api/admin/adjust")
async def api_admin_adjust(request: Request, body: AdminAdjustIn):
    # Admins and owners only
    s, prof = _require_role(request, "admin")
    with psycopg.connect(DATABASE_URL) as con, con.cursor() as cur:
        target = _resolve_identifier(cur, body.identifier)
    if not target:
        raise HTTPException(404, "Target user not found")
    try:
        amt = q2(D(body.amount))
    except Exception:
        raise HTTPException(400, "Invalid amount")

    new_bal = adjust_balance(s["id"], target, amt, body.reason or "admin-adjust")
    return {"ok": True, "target_id": target, "new_balance": float(new_bal)}

class AdminRoleIn(BaseModel):
    identifier: str
    role: str

@app.post("/api/admin/role")
async def api_admin_role(request: Request, body: AdminRoleIn):
    # Owner only
    s, prof = _require_role(request, "owner")
    with psycopg.connect(DATABASE_URL) as con, con.cursor() as cur:
        target = _resolve_identifier(cur, body.identifier)
    if not target:
        raise HTTPException(404, "Target user not found")
    try:
        res = set_role(target, body.role)
        return res
    except ValueError as e:
        raise HTTPException(400, str(e))

class AdminAnnounceIn(BaseModel):
    text: str
    minutes: Optional[int] = 120

@with_conn
def _create_announcement(cur, creator_id: str, text: str, minutes: int):
    starts = now_utc()
    ends = starts + datetime.timedelta(minutes=max(1, int(minutes or 0)))
    cur.execute(
        "INSERT INTO announcements(text, starts_at, ends_at, created_by) VALUES (%s,%s,%s,%s) RETURNING id",
        (text, starts, ends, creator_id),
    )
    new_id = cur.fetchone()[0]
    return {"ok": True, "id": int(new_id)}

@app.post("/api/admin/announce")
async def api_admin_announce(request: Request, body: AdminAnnounceIn):
    # Admins and owners can post announcements
    s, _ = _require_role(request, "admin")
    if not (body.text or "").strip():
        raise HTTPException(400, "Text required")
    return _create_announcement(s["id"], body.text.strip(), int(body.minutes or 120))

# ---------- Announcements (public) ----------

@with_conn
def _active_announcements(cur):
    now = now_utc()
    cur.execute(
        """
        SELECT id, text, starts_at, ends_at
        FROM announcements
        WHERE (starts_at IS NULL OR starts_at <= %s)
          AND (ends_at IS NULL OR ends_at >= %s)
        ORDER BY COALESCE(starts_at, CURRENT_TIMESTAMP) DESC, id DESC
        LIMIT 10
        """,
        (now, now),
    )
    return [
        {"id": int(r[0]), "text": r[1], "starts_at": iso(r[2]), "ends_at": iso(r[3])}
        for r in cur.fetchall()
    ]

@app.get("/api/announcements/active")
async def api_ann_active():
    return {"rows": _active_announcements()}

# ---------- Bot status (for health / gating) ----------

@app.get("/api/bot/status")
async def api_bot_status():
    return {
        "ok": True,
        "configured": bool(DISCORD_BOT_TOKEN and GUILD_ID),
        "guild_id": int(GUILD_ID or 0),
        "start_flag": bool(START_DISCORD_BOT),
    }

# ---------- Uvicorn entrypoint ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=RELOAD,
        log_level="info",
    )
