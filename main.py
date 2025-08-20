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
OWNER_GATE_ALLOW = ("/login", "/callback", "/static/", "/img/", "/healthz", "/api/bot/status")

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
          <a class="btn" href="/login">Login with Discord</a>
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
<div id="preload"><div class="loader"></div></div>

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

<!-- Account Modal (avatar: Profile & Settings) -->
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
      <img class="avatar" id="avatarBtn" src="${me.avatar_url||''}" title="${me.username||'user'}"/>
    `;
    qs('btnJoinSmall').onclick = joinDiscord;
    qs('btnDW').onclick = ()=> openDW();
    qs('avatarBtn').onclick = ()=> openAccount(prof, bal);
  }else{
    area.innerHTML = `
      <button class="btn primary" id="btnDW">Deposit / Withdraw</button>
      <a class="btn ghost" href="/login">Login with Discord</a>
    `;
    qs('btnDW').onclick = ()=> openDW();
  }
}

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
    toast(`Request #${r.id} submitted
