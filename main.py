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

# NEW: Discord Guild Join config
GUILD_ID = os.getenv("GUILD_ID") or os.getenv("DISCORD_GUILD_ID")  # your server ID
BOT_TOKEN = os.getenv("BOT_TOKEN") or os.getenv("DISCORD_BOT_TOKEN")  # your bot token
OAUTH_JOIN_REDIRECT = os.getenv("OAUTH_JOIN_REDIRECT", "")  # e.g. https://your.app/oauth/join
DISCORD_INVITE = os.getenv("DISCORD_INVITE", "")  # optional fallback invite link

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

    # profiles
    cur.execute("""
        CREATE TABLE IF NOT EXISTS profiles (
            user_id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            name_lower TEXT NOT NULL UNIQUE,
            xp INTEGER NOT NULL DEFAULT 0,
            role TEXT NOT NULL DEFAULT 'member',
            is_anon BOOLEAN NOT NULL DEFAULT FALSE,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("ALTER TABLE profiles ADD COLUMN IF NOT EXISTS role TEXT NOT NULL DEFAULT 'member'")
    cur.execute("ALTER TABLE profiles ADD COLUMN IF NOT EXISTS is_anon BOOLEAN NOT NULL DEFAULT FALSE")

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

    # NEW: referrals
    cur.execute("""
        CREATE TABLE IF NOT EXISTS referrals (
            id BIGSERIAL PRIMARY KEY,
            referrer_id TEXT NOT NULL,
            referrer_name TEXT NOT NULL,
            invited_user_id TEXT UNIQUE NOT NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
    """)

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

# ---- NEW: referral helpers ----
@with_conn
def find_user_id_by_name_lower(cur, name_lower: str) -> Optional[str]:
    cur.execute("SELECT user_id FROM profiles WHERE name_lower=%s", (name_lower,))
    r = cur.fetchone()
    return str(r[0]) if r else None

@with_conn
def add_referral(cur, referrer_id: str, referrer_name: str, invited_user_id: str) -> bool:
    if str(referrer_id) == str(invited_user_id):
        return False
    cur.execute("""
        INSERT INTO referrals (referrer_id, referrer_name, invited_user_id)
        VALUES (%s,%s,%s)
        ON CONFLICT (invited_user_id) DO NOTHING
    """, (referrer_id, referrer_name, invited_user_id))
    return True

@with_conn
def referrals_count_for(cur, referrer_id: str) -> int:
    cur.execute("SELECT COUNT(*) FROM referrals WHERE referrer_id=%s", (referrer_id,))
    return int(cur.fetchone()[0])

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

# ---- Crash math ----
def _u(): return (secrets.randbelow(1_000_000_000)+1)/1_000_000_001.0
def gen_bust(edge: Decimal) -> float:
    u = _u()
    B = max(1.0, float((Decimal("1.0") - edge) / Decimal(str(u))))
    return math.floor(B*100)/100.0
def run_duration_for(bust: float) -> float:
    return min(22.0, 8.0 + math.log(bust + 1.0) * 6.0)
def current_multiplier(started_at: datetime.datetime, expected_end_at: datetime.datetime, bust: float, at: Optional[datetime.datetime] = None) -> float:
    at = at or now_utc()
    if at <= started_at: return 1.0
    if at >= expected_end_at: return bust
    frac = (at - started_at).total_seconds() / max(0.001, (expected_end_at - started_at).total_seconds())
    m = math.exp(math.log(bust) * frac)
    return math.floor(m*100)/100.0

@with_conn
def ensure_betting_round(cur) -> Tuple[int, dict]:
    cur.execute("SELECT id,status,betting_opens_at,betting_ends_at,started_at,expected_end_at,bust FROM crash_rounds ORDER BY id DESC LIMIT 1")
    r = cur.fetchone()
    now = now_utc()
    if not r or r[1] == 'ended':
        opens = now
        ends = now + datetime.timedelta(seconds=BETTING_SECONDS)
        cur.execute("""INSERT INTO crash_rounds(status,betting_opens_at,betting_ends_at)
                       VALUES('betting',%s,%s) RETURNING id,status,betting_opens_at,betting_ends_at,started_at,expected_end_at,bust""",
                    (opens, ends))
        r = cur.fetchone()
    return r[0], {
        "status": r[1],
        "betting_opens_at": r[2],
        "betting_ends_at": r[3],
        "started_at": r[4],
        "expected_end_at": r[5],
        "bust": float(r[6]) if r[6] is not None else None
    }

@with_conn
def place_bet(cur, user_id: str, bet: Decimal, cashout: float):
    cur.execute("""SELECT id, betting_ends_at FROM crash_rounds
                   WHERE status='betting'
                   ORDER BY id DESC LIMIT 1""")
    row = cur.fetchone()
    if not row: raise ValueError("Betting is closed")
    round_id, ends_at = int(row[0]), row[1]
    cur.execute("SELECT NOW() < %s", (ends_at,))
    if not cur.fetchone()[0]: raise ValueError("Betting just closed")

    cur.execute("INSERT INTO balances(user_id,balance) VALUES (%s,0) ON CONFLICT(user_id) DO NOTHING", (user_id,))
    cur.execute("SELECT balance FROM balances WHERE user_id=%s FOR UPDATE", (user_id,))
    bal = D(cur.fetchone()[0])
    if bet < MIN_BET: raise ValueError(f"Min bet is {MIN_BET:.2f} DL")
    if bet > MAX_BET: raise ValueError(f"Max bet is {MAX_BET:.2f} DL")
    if bal < bet: raise ValueError("Insufficient balance")
    cur.execute("UPDATE balances SET balance=balance-%s WHERE user_id=%s", (q2(bet), user_id))

    try:
        cur.execute("""INSERT INTO crash_bets(round_id,user_id,bet,cashout)
                       VALUES(%s,%s,%s,%s)""",
                    (round_id, user_id, q2(bet), float(cashout)))
    except Exception:
        cur.execute("UPDATE balances SET balance=balance+%s WHERE user_id=%s", (q2(bet), user_id))
        raise ValueError("You already placed a bet this round")
    return {"round_id": round_id}

@with_conn
def load_round(cur):
    cur.execute("""SELECT id,status,betting_opens_at,betting_ends_at,started_at,expected_end_at,bust
                   FROM crash_rounds ORDER BY id DESC LIMIT 1""")
    r = cur.fetchone()
    if not r: return None
    return {
        "id": int(r[0]),
        "status": r[1],
        "betting_opens_at": r[2], "betting_ends_at": r[3],
        "started_at": r[4], "expected_end_at": r[5],
        "bust": float(r[6]) if r[6] is not None else None
    }

@with_conn
def begin_running(cur, round_id: int):
    cur.execute("SELECT status FROM crash_rounds WHERE id=%s FOR UPDATE", (round_id,))
    st = cur.fetchone()
    if not st or st[0] != 'betting': return None

    bust = gen_bust(HOUSE_EDGE_CRASH)
    dur = run_duration_for(bust)
    now = now_utc()
    exp_end = now + datetime.timedelta(seconds=dur)
    cur.execute("""UPDATE crash_rounds
                   SET status='running', bust=%s, started_at=%s, expected_end_at=%s
                   WHERE id=%s""",
                (float(bust), now, exp_end, round_id))
    return {"bust": bust, "expected_end_at": exp_end}

@with_conn
def cashout_now(cur, user_id: str) -> Dict:
    cur.execute("""SELECT id, started_at, expected_end_at, bust FROM crash_rounds
                   WHERE status='running' ORDER BY id DESC LIMIT 1""")
    r = cur.fetchone()
    if not r: raise ValueError("No running round")
    rid, started_at, exp_end, bust = int(r[0]), r[1], r[2], float(r[3])
    now = now_utc()
    if now >= exp_end: raise ValueError("Round already crashed")
    cur.execute("SELECT bet, cashout, cashed_out, resolved FROM crash_bets WHERE round_id=%s AND user_id=%s FOR UPDATE",
                (rid, user_id))
    b = cur.fetchone()
    if not b: raise ValueError("You have no active bet")
    bet, cash_goal, cashed_out, resolved = D(b[0]), float(b[1]), b[2], bool(b[3])
    if resolved or cashed_out is not None:
        raise ValueError("Already cashed out")

    m = current_multiplier(started_at, exp_end, bust, now)
    if m >= bust: raise ValueError("Too late — crashed")
    win = q2(bet * D(m))
    cur.execute("UPDATE balances SET balance=balance+%s WHERE user_id=%s", (win, user_id))
    cur.execute("""UPDATE crash_bets
                   SET cashed_out=%s, cashed_out_at=%s, win=%s, resolved=TRUE
                   WHERE round_id=%s AND user_id=%s""",
                (float(m), now, win, rid, user_id))
    return {"round_id": rid, "multiplier": m, "win": float(win)}

@with_conn
def resolve_round_end(cur, round_id: int, bust: float):
    cur.execute("""SELECT user_id, bet, cashout, cashed_out, resolved, win
                   FROM crash_bets WHERE round_id=%s""", (round_id,))
    bets = cur.fetchall()
    for uid, bet, goal, cashed, resolved, win in bets:
        uid = str(uid); bet=D(bet); goal=float(goal); win=D(win); resolved=bool(resolved)
        if resolved and cashed is not None:
            xp_gain = max(1, min(int(bet), 50))
            cur.execute("""INSERT INTO crash_games(user_id,bet,cashout,bust,win,xp_gain)
                           VALUES(%s,%s,%s,%s,%s,%s)""",
                        (uid, q2(bet), float(cashed), float(bust), q2(win), xp_gain))
            ensure_profile_row(uid)
            cur.execute("UPDATE profiles SET xp=xp+%s WHERE user_id=%s", (xp_gain, uid))
            continue

        if not resolved:
            if goal <= bust:
                win = q2(bet * D(goal))
                cur.execute("UPDATE balances SET balance=balance+%s WHERE user_id=%s", (win, uid))
                cur.execute("""UPDATE crash_bets SET win=%s, resolved=TRUE WHERE round_id=%s AND user_id=%s""",
                            (win, round_id, uid))
                cashed_val = goal
            else:
                cur.execute("""UPDATE crash_bets SET resolved=TRUE WHERE round_id=%s AND user_id=%s""",
                            (round_id, uid))
                win = Decimal("0.00")
                cashed_val = goal

            xp_gain = max(1, min(int(bet), 50))
            cur.execute("""INSERT INTO crash_games(user_id,bet,cashout,bust,win,xp_gain)
                           VALUES(%s,%s,%s,%s,%s,%s)""",
                        (uid, q2(bet), float(cashed_val), float(bust), q2(win), xp_gain))
            ensure_profile_row(uid)
            cur.execute("UPDATE profiles SET xp=xp+%s WHERE user_id=%s", (xp_gain, uid))

@with_conn
def finish_round(cur, round_id: int):
    cur.execute("SELECT bust FROM crash_rounds WHERE id=%s", (round_id,))
    bust = float(cur.fetchone()[0])
    now = now_utc()
    resolve_round_end(cur, round_id, bust)
    cur.execute("""UPDATE crash_rounds
                   SET status='ended', ended_at=%s
                   WHERE id=%s AND status='running'""", (now, round_id))

@with_conn
def create_next_betting(cur):
    now = now_utc()
    opens = now
    ends = now + datetime.timedelta(seconds=BETTING_SECONDS)
    cur.execute("""INSERT INTO crash_rounds(status,betting_opens_at,betting_ends_at)
                   VALUES('betting',%s,%s) RETURNING id""", (opens, ends))
    return int(cur.fetchone()[0])

@with_conn
def last_busts(cur, limit: int = 15):
    cur.execute("SELECT bust FROM crash_rounds WHERE status='ended' ORDER BY id DESC LIMIT %s", (limit,))
    return [float(r[0]) for r in cur.fetchall()]

@with_conn
def your_bet(cur, round_id: int, user_id: str):
    cur.execute("""SELECT bet, cashout, cashed_out, resolved, win
                   FROM crash_bets WHERE round_id=%s AND user_id=%s""", (round_id, user_id))
    r = cur.fetchone()
    if not r: return None
    return {"bet": float(q2(D(r[0]))), "cashout": float(r[1]),
            "cashed_out": (float(r[2]) if r[2] is not None else None),
            "resolved": bool(r[3]), "win": float(q2(D(r[4])))}

@with_conn
def your_history(cur, user_id: str, limit: int = 10):
    cur.execute("""SELECT bet, cashout, bust, win, xp_gain, created_at
                   FROM crash_games WHERE user_id=%s
                   ORDER BY id DESC LIMIT %s""", (user_id, limit))
    return [{"bet":float(q2(D(r[0]))),"cashout":float(r[1]),"bust":float(r[2]),"win":float(q2(D(r[3]))),"xp_gain":int(r[4]),"created_at":str(r[5])} for r in cur.fetchall()]

# ---------- Chat helpers ----------
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
        cur.execute("SELECT xp FROM profiles WHERE user_id=%s", (user_id,))
        xp = int(cur.fetchone()[0])
        lvl = 1 + xp // 100
        if lvl < 5: raise PermissionError("You must be level 5 to chat")
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

# ---------- MINES helpers ----------
def mines_random_board(mines: int) -> str:
    idxs = list(range(25))
    random.shuffle(idxs)
    mines_set = set(idxs[:mines])
    return ''.join('1' if i in mines_set else '0' for i in range(25))
def sha256(s: str) -> str: return hashlib.sha256(s.encode('utf-8')).hexdigest()
def picks_count_from_bitmask(mask: int) -> int: return mask.bit_count()
def mines_multiplier(mines: int, picks_count: int) -> float:
    if picks_count <= 0: return 1.0
    total = Decimal("1.0")
    for i in range(picks_count):
        total *= D(25 - i) / D(max(1, 25 - mines - i))
        total *= (Decimal("1.0") - HOUSE_EDGE_MINES)
    return float(total)

@with_conn
def mines_active_for(cur, user_id: str):
    cur.execute("""SELECT id, bet, mines, board, picks, seed, commit_hash, status
                   FROM mines_games
                   WHERE user_id=%s AND status='active'
                   ORDER BY id DESC LIMIT 1""", (user_id,))
    r = cur.fetchone()
    if not r: return None
    return {"id": int(r[0]), "bet": float(q2(D(r[1]))), "mines": int(r[2]),
            "board": str(r[3]), "picks": int(r[4]), "seed": str(r[5]), "hash": str(r[6]), "status": str(r[7])}

@with_conn
def mines_start(cur, user_id: str, bet: Decimal, mines: int):
    if bet < MIN_BET or bet > MAX_BET: raise ValueError(f"Bet must be between {MIN_BET:.2f} and {MAX_BET:.2f}")
    if mines < 1 or mines > 24: raise ValueError("Mines must be between 1 and 24")
    cur.execute("SELECT 1 FROM mines_games WHERE user_id=%s AND status='active'", (user_id,))
    if cur.fetchone(): raise ValueError("You already have an active Mines game")

    cur.execute("INSERT INTO balances(user_id,balance) VALUES (%s,0) ON CONFLICT(user_id) DO NOTHING", (user_id,))
    cur.execute("SELECT balance FROM balances WHERE user_id=%s FOR UPDATE", (user_id,))
    bal = D(cur.fetchone()[0])
    if bal < bet: raise ValueError("Insufficient balance")
    cur.execute("UPDATE balances SET balance=balance-%s WHERE user_id=%s", (q2(bet), user_id))

    board = mines_random_board(mines)
    seed = secrets.token_hex(16)
    commit_hash = sha256(f"{seed}:{board}")
    cur.execute("""INSERT INTO mines_games(user_id, bet, mines, board, seed, commit_hash)
                   VALUES (%s,%s,%s,%s,%s,%s) RETURNING id""",
                (user_id, q2(bet), mines, board, seed, commit_hash))
    gid = int(cur.fetchone()[0])
    return {"id": gid, "hash": commit_hash}

@with_conn
def mines_pick(cur, user_id: str, index: int):
    if index < 0 or index >= 25: raise ValueError("Invalid tile")
    cur.execute("""SELECT id, bet, mines, board, picks, status
                   FROM mines_games WHERE user_id=%s AND status='active'
                   ORDER BY id DESC LIMIT 1 FOR UPDATE""", (user_id,))
    r = cur.fetchone()
    if not r: raise ValueError("No active Mines game")
    gid, bet, mines, board, picks, status = int(r[0]), D(r[1]), int(r[2]), str(r[3]), int(r[4]), str(r[5])
    bit = 1 << index
    if picks & bit: raise ValueError("Tile already revealed")

    is_mine = (board[index] == '1')
    new_picks = picks | bit
    if is_mine:
        cur.execute("""UPDATE mines_games
                       SET picks=%s, status='lost', ended_at=NOW()
                       WHERE id=%s""", (new_picks, gid))
        return {"status": "lost", "board": board, "index": index}
    else:
        cur.execute("""UPDATE mines_games SET picks=%s WHERE id=%s""", (new_picks, gid))
        pcount = picks_count_from_bitmask(new_picks)
        mult = mines_multiplier(mines, pcount)
        win = q2(bet * D(mult))
        return {"status": "active", "picks": new_picks, "multiplier": float(mult), "potential_win": float(win)}

@with_conn
def mines_cashout(cur, user_id: str):
    cur.execute("""SELECT id, bet, mines, board, picks, status
                   FROM mines_games WHERE user_id=%s AND status='active'
                   ORDER BY id DESC LIMIT 1 FOR UPDATE""", (user_id,))
    r = cur.fetchone()
    if not r: raise ValueError("No active Mines game")
    gid, bet, mines, board, picks, status = int(r[0]), D(r[1]), int(r[2]), str(r[3]), int(r[4]), str(r[5])
    pcount = picks_count_from_bitmask(picks)
    if pcount < 1: raise ValueError("Reveal at least one tile before cashing out")
    mult = mines_multiplier(mines, pcount)
    win = q2(bet * D(mult))
    cur.execute("UPDATE balances SET balance=balance+%s WHERE user_id=%s", (win, user_id))
    cur.execute("""UPDATE mines_games
                   SET status='cashed', win=%s, ended_at=NOW()
                   WHERE id=%s""", (win, gid))
    return {"win": float(win), "board": board}

@with_conn
def mines_state(cur, user_id: str):
    cur.execute("""SELECT id, bet, mines, picks, commit_hash, status
                   FROM mines_games WHERE user_id=%s AND status<>'lost'
                   ORDER BY id DESC LIMIT 1""", (user_id,))
    r = cur.fetchone()
    if not r: return None
    return {"id": int(r[0]), "bet": float(q2(D(r[1]))), "mines": int(r[2]),
            "picks": int(r[3]), "hash": str(r[4]), "status": str(r[5])}

@with_conn
def mines_history(cur, user_id: str, limit: int = 15):
    cur.execute("""SELECT id, bet, mines, win, status, started_at, ended_at, commit_hash, seed, board
                   FROM mines_games WHERE user_id=%s AND status<>'active'
                   ORDER BY id DESC LIMIT %s""", (user_id, limit))
    rows = cur.fetchall()
    out = []
    for r in rows:
        out.append({
            "id": int(r[0]), "bet": float(q2(D(r[1]))), "mines": int(r[2]), "win": float(q2(D(r[3]))),
            "status": str(r[4]), "started_at": str(r[5]), "ended_at": str(r[6]),
            "hash": str(r[7]), "seed": str(r[8]), "board": str(r[9])
        })
    return out

# ---------- Leaderboard (final version) ----------
def _start_of_utc_day(dt: datetime.datetime) -> datetime.datetime:
    return dt.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
def _start_of_utc_month(dt: datetime.datetime) -> datetime.datetime:
    return dt.astimezone(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)

@with_conn
def leaderboard_rows(cur, period: str, limit: int = 50):
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
    # If an older 'anonymous' column exists, copy into is_anon once
    cur.execute("""
    DO $$
    BEGIN
      IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='profiles' AND column_name='anonymous'
      ) THEN
        UPDATE profiles SET is_anon = COALESCE(is_anon, FALSE) OR COALESCE(anonymous, FALSE);
      END IF;
    END$$;
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS ix_crash_games_created_at ON crash_games (created_at)")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_mines_games_started_at ON mines_games (started_at)")

# ---------- HTML ----------
HTML_TEMPLATE = """
<!doctype html><html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"/>
<title>💎 DL Bank</title>
<style>
:root{--bg:#0a0f1e;--bg2:#0c1428;--card:#111a31;--muted:#9eb3da;--text:#ecf2ff;--accent:#6aa6ff;--accent2:#22c1dc;--ok:#34d399;--warn:#f59e0b;--err:#ef4444;--border:#1f2b47;--chatW:300px;--input-bg:#0b1430;--input-br:#223457;--input-tx:#e6eeff;--input-ph:#9db4e4;--hl:#1b2f5c;--me:#123e25}
*{box-sizing:border-box}html,body{height:100%}body{margin:0;color:var(--text);background:radial-gradient(1400px 600px at 20% -10%, #11204d 0%, transparent 60%),linear-gradient(180deg,#0a0f1e,#0a0f1e 60%, #0b1124);font-family:Inter,system-ui,Segoe UI,Roboto,Arial,Helvetica,sans-serif;-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale}
a{color:inherit;text-decoration:none}.container{max-width:1100px;margin:0 auto;padding:16px}
input,select,textarea{width:100%;appearance:none;background:var(--input-bg);color:var(--input-tx);border:1px solid var(--input-br);border-radius:12px;padding:10px 12px;outline:none;transition:border-color .15s ease, box-shadow .15s ease}
input::placeholder{color:var(--input-ph)}input:focus{border-color:#4c78ff;box-shadow:0 0 0 3px rgba(76,120,255,.18)}
.field{display:flex;flex-direction:column;gap:6px}.row{display:grid;gap:10px}
.row.cols-2{grid-template-columns:1fr 1fr}.row.cols-3{grid-template-columns:1fr 1fr 1fr}.row.cols-4{grid-template-columns:1.6fr 1fr 1fr auto}.row.cols-5{grid-template-columns:2fr 1fr 1fr auto auto}
.card{background:linear-gradient(180deg,#0f1a33,#0b1326);border:1px solid var(--border);border-radius:16px;padding:16px}
.header{position:sticky;top:0;z-index:30;backdrop-filter:blur(8px);background:rgba(10,15,30,.7);border-bottom:1px solid var(--border)}
.header-inner{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:10px 12px}
.left{display:flex;align-items:center;gap:14px;flex:1;min-width:0}.brand{display:flex;align-items:center;gap:10px;font-weight:800;letter-spacing:.2px;white-space:nowrap}
.brand .logo{width:28px;height:28px;border-radius:8px;background:linear-gradient(135deg,var(--accent),var(--accent2))}
.tabs{display:flex;gap:4px;align-items:center;padding:4px;border-radius:14px;background:linear-gradient(180deg,#0f1a33,#0b1326);border:1px solid var(--border)}
.tab{padding:8px 12px;border-radius:10px;cursor:pointer;font-weight:700;white-space:nowrap;color:#d8e6ff;opacity:.85;transition:all .15s ease;display:flex;align-items:center;gap:8px}
.tab:hover{opacity:1;transform:translateY(-1px)}.tab.active{background:linear-gradient(135deg,#3b82f6,#22c1dc);color:#051326;box-shadow:0 6px 16p


x rgba(59,130,246,.25);opacity:1}
.right{display:flex;gap:8px;align-items:center;margin-left:12px}.chip{background:#0c1631;border:1px solid var(--border);color:#dce7ff;padding:6px 10px;border-radius:999px;font-size:12px;white-space:nowrap;cursor:pointer}
.avatar{width:34px;height:34px;border-radius:50%;object-fit:cover;border:1px solid var(--border);cursor:pointer}
.btn{display:inline-flex;align-items:center;gap:8px;padding:10px 14px;border-radius:12px;border:1px solid var(--border);background:linear-gradient(180deg,#0e1833,#0b1326);cursor:pointer;font-weight:600}
.btn.primary{background:linear-gradient(135deg,#3b82f6,#22c1dc);color:#051326}
.btn.ok{background:linear-gradient(135deg,#34d399,#16a34a);color:#03130a}
.btn.warn{background:linear-gradient(135deg,#f59e0b,#fbbf24);color:#130c03}
.btn.err{background:linear-gradient(135deg,#ef4444,#f87171);color:#130303}
.stat{display:flex;align-items:center;gap:8px;padding:8px 12px;border:1px solid var(--border);border-radius:12px;background:#0b1326}
.grid{display:grid;gap:10px}
.grid.cols-2{grid-template-columns:1fr 1fr}
.list{display:flex;flex-direction:column;gap:8px}
.table{width:100%;border-collapse:collapse}
.table th,.table td{padding:8px 10px;border-bottom:1px solid var(--border);text-align:left}
.badge{padding:2px 8px;border-radius:999px;border:1px solid var(--border);background:#0c1631;font-size:12px}
.code{font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;background:#0a142c;border:1px solid var(--border);padding:3px 6px;border-radius:8px}
.kbd{font-family:inherit;border:1px solid var(--border);border-bottom-width:2px;background:#0b1430;border-radius:6px;padding:2px 6px}
.small{font-size:12px;color:#b7c7ea}
.center{display:flex;align-items:center;justify-content:center}
.hidden{display:none}
.mines-board{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-top:10px}
.tile{aspect-ratio:1;border:1px solid var(--border);border-radius:12px;background:#0b1430;display:flex;align-items:center;justify-content:center;font-weight:800;cursor:pointer}
.tile.revealed{background:#0e1c3d}
.tile.mine{background:#3b0f18;border-color:#80293b}
.chat-box{max-height:320px;overflow:auto;border:1px solid var(--border);border-radius:12px;padding:8px;background:#0b1326}
.chat-line{display:flex;gap:8px;padding:6px 8px;border-radius:8px}
.chat-line.me{background:rgba(52,211,153,.08)}
.chat-meta{font-size:12px;color:#9eb3da}
.notice{padding:10px 12px;border:1px dashed var(--border);border-radius:12px;background:rgba(255,255,255,.03)}
.copy{cursor:pointer}
</style>
</head>
<body>
<header class="header">
  <div class="header-inner container">
    <div class="left">
      <div class="brand"><div class="logo"></div> DL Bank</div>
      <nav class="tabs" id="tabs">
        <div class="tab active" data-tab="crash">Crash</div>
        <div class="tab" data-tab="mines">Mines</div>
        <div class="tab" data-tab="chat">Chat</div>
        <div class="tab" data-tab="promos">Promos</div>
        <div class="tab" data-tab="leaderboard">Leaderboard</div>
        <div class="tab" data-tab="profile">Profile</div>
      </nav>
    </div>
    <div class="right">
      <div class="stat" id="statBalance">Balance: <span class="code" id="balanceVal">0.00</span> DL</div>
      <a class="btn" id="btnReferral" title="Copy your referral link">Refer</a>
      <a class="btn primary" id="btnJoin">Join Discord</a>
      <a class="btn" id="btnLogin">Login</a>
      <a class="btn err hidden" id="btnLogout">Logout</a>
    </div>
  </div>
</header>

<main class="container" style="padding-top:20px">
  <!-- Crash -->
  <section id="view-crash" class="card">
    <h2>Crash</h2>
    <div class="row cols-4">
      <div class="field">
        <label>Bet (DL)</label>
        <input id="crashBet" type="number" min="1" step="0.01" placeholder="10.00">
      </div>
      <div class="field">
        <label>Auto Cashout (x)</label>
        <input id="crashCashout" type="number" min="1.01" step="0.01" placeholder="2.00">
      </div>
      <div class="field">
        <label>&nbsp;</label>
        <button class="btn primary" id="btnCrashBet">Place bet</button>
      </div>
      <div class="field">
        <label>&nbsp;</label>
        <button class="btn ok" id="btnCrashCashout">Cash out now</button>
      </div>
    </div>
    <div class="notice" id="crashStatus">Waiting…</div>
    <div class="row cols-2" style="margin-top:12px">
      <div class="card">
        <div>Round: <span id="crashRound">—</span></div>
        <div>State: <span id="crashState">—</span></div>
        <div>Time: <span id="crashTime">—</span></div>
        <div>Multiplier: <span class="code" id="crashMult">1.00x</span></div>
      </div>
      <div class="card">
        <div class="small">Last busts</div>
        <div id="lastBusts" class="list"></div>
      </div>
    </div>
    <div class="card" style="margin-top:12px">
      <div class="small">Your recent games</div>
      <div id="crashHistory" class="list"></div>
    </div>
  </section>

  <!-- Mines -->
  <section id="view-mines" class="card hidden">
    <h2>Mines</h2>
    <div class="row cols-3">
      <div class="field">
        <label>Bet (DL)</label>
        <input id="minesBet" type="number" min="1" step="0.01" placeholder="5.00">
      </div>
      <div class="field">
        <label>Mines (1–24)</label>
        <input id="minesCount" type="number" min="1" max="24" value="3">
      </div>
      <div class="field">
        <label>&nbsp;</label>
        <button class="btn primary" id="btnMinesStart">Start</button>
      </div>
    </div>
    <div class="notice" id="minesStatus">No active game.</div>
    <div id="minesControls" class="row cols-2 hidden" style="margin-top:12px">
      <div>
        <div>Bet: <span class="code" id="minesBetShow">—</span></div>
        <div>Mines: <span class="code" id="minesCountShow">—</span></div>
        <div>Potential win: <span class="code" id="minesPotential">—</span></div>
      </div>
      <div class="right" style="justify-content:flex-end">
        <button class="btn ok" id="btnMinesCashout">Cashout</button>
      </div>
    </div>
    <div class="mines-board" id="minesBoard"></div>
    <div class="card" style="margin-top:12px">
      <div class="small">Your Mines history</div>
      <div id="minesHistory" class="list"></div>
    </div>
  </section>

  <!-- Chat -->
  <section id="view-chat" class="card hidden">
    <h2>Chat</h2>
    <div class="chat-box" id="chatBox"></div>
    <div class="row cols-2" style="margin-top:8px">
      <input id="chatInput" placeholder="Say something (Level 5+ for public chat)">
      <button class="btn primary" id="btnChatSend">Send</button>
    </div>
  </section>

  <!-- Promos -->
  <section id="view-promos" class="card hidden">
    <h2>Promos</h2>
    <div class="row cols-3">
      <input id="promoCode" placeholder="Enter code…">
      <button class="btn primary" id="btnRedeem">Redeem</button>
      <div id="promoResult" class="stat">—</div>
    </div>
    <div class="notice" style="margin-top:10px">
      Share your referral link: <span class="code" id="refLink">—</span>
      <button class="btn" id="btnCopyRef">Copy</button>
      <div class="small">When someone signs in via your link, they’re marked as invited by you.</div>
    </div>
  </section>

  <!-- Leaderboard -->
  <section id="view-leaderboard" class="card hidden">
    <div class="row cols-2">
      <h2>Leaderboard</h2>
      <div class="right">
        <select id="lbPeriod">
          <option value="daily">Daily</option>
          <option value="monthly">Monthly</option>
          <option value="all">All-time</option>
        </select>
        <button class="btn" id="btnLbRefresh">Refresh</button>
      </div>
    </div>
    <table class="table" id="lbTable">
      <thead><tr><th>#</th><th>User</th><th>Total Wagered (DL)</th></tr></thead>
      <tbody></tbody>
    </table>
  </section>

  <!-- Profile -->
  <section id="view-profile" class="card hidden">
    <h2>Profile</h2>
    <div id="meInfo" class="list"></div>
    <div class="row cols-3" style="margin-top:10px">
      <input id="newName" placeholder="Set display name (3–20)">
      <button class="btn" id="btnSetName">Save</button>
      <button class="btn" id="btnToggleAnon">Toggle Anon</button>
    </div>
  </section>
</main>

<script>
const $ = sel => document.querySelector(sel);
const $$ = sel => document.querySelectorAll(sel);
const fmt = n => (Number(n||0).toFixed(2));

let ME = null;
let CHAT_SINCE = 0;
let CRASH_TICK = null;

function showTab(id){
  $$("#tabs .tab").forEach(t => t.classList.toggle("active", t.dataset.tab===id));
  ["crash","mines","chat","promos","leaderboard","profile"].forEach(k=>{
    $("#view-"+k).classList.toggle("hidden", k!==id);
  });
  if(id==="chat"){ loadChat(); }
  if(id==="leaderboard"){ loadLeaderboard(); }
  if(id==="crash"){ startCrashLoop(); }
}

$("#tabs").addEventListener("click", (e)=>{
  const t = e.target.closest(".tab"); if(!t) return;
  showTab(t.dataset.tab);
});

async function jfetch(path, opts={}){
  const r = await fetch(path, Object.assign({headers:{'Content-Type':'application/json'}}, opts));
  if(!r.ok){
    const text = await r.text();
    throw new Error(text || (r.status+" "+r.statusText));
  }
  const ctype = r.headers.get("content-type") || "";
  if(ctype.includes("application/json")) return r.json();
  return r.text();
}

async function refreshMe(){
  try{
    ME = await jfetch("/me");
    $("#btnLogin").classList.add("hidden");
    $("#btnLogout").classList.remove("hidden");
    $("#balanceVal").textContent = fmt(ME.balance);
    $("#btnReferral").classList.remove("hidden");
    $("#refLink").textContent = location.origin + "/ref/" + (ME.name_lower || "me");
  }catch{
    ME = null;
    $("#btnLogin").classList.remove("hidden");
    $("#btnLogout").classList.add("hidden");
    $("#btnReferral").classList.add("hidden");
    $("#balanceVal").textContent = "0.00";
    $("#refLink").textContent = "—";
  }
}

$("#btnLogin").onclick = ()=> location.href = "/login";
$("#btnLogout").onclick = async ()=>{ await jfetch("/logout"); await refreshMe(); }
$("#btnJoin").onclick = ()=> location.href = "/login/join";

$("#btnCopyRef").onclick = async ()=>{
  const text = $("#refLink").textContent;
  try{ await navigator.clipboard.writeText(text); alert("Copied!"); }catch(e){ alert(text); }
};
$("#btnReferral").onclick = ()=> $("#view-promos").scrollIntoView({behavior:"smooth"});

async function loadLeaderboard(){
  const period = $("#lbPeriod").value;
  const rows = await jfetch(`/leaderboard?period=${encodeURIComponent(period)}`);
  const tbody = $("#lbTable tbody");
  tbody.innerHTML = "";
  rows.forEach((r,i)=>{
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${i+1}</td><td>${r.is_anon?"Anonymous":r.display_name}</td><td>${fmt(r.total_wagered)}</td>`;
    tbody.appendChild(tr);
  });
}
$("#btnLbRefresh").onclick = loadLeaderboard;

async function loadChat(){
  try{
    const lines = await jfetch(`/chat?since_id=${CHAT_SINCE}`);
    const box = $("#chatBox");
    lines.forEach(l=>{
      CHAT_SINCE = Math.max(CHAT_SINCE, l.id);
      const div = document.createElement("div");
      div.className = "chat-line" + (ME && ME.id === l.user_id ? " me":"");
      div.innerHTML = `<div><div class="chat-meta">[Lv${l.level}] ${(l.role||"").toUpperCase()} • ${l.username}</div><div>${escapeHtml(l.text)}</div></div>`;
      box.appendChild(div);
    });
    box.scrollTop = box.scrollHeight;
  }catch(e){}
}
setInterval(loadChat, 2000);

$("#btnChatSend").onclick = async ()=>{
  const t = $("#chatInput").value.trim(); if(!t) return;
  $("#chatInput").value = "";
  try{ await jfetch("/chat", {method:"POST", body:JSON.stringify({text:t})}); }catch(e){ alert(e.message); }
};

function escapeHtml(s){ return s.replace(/[&<>"]/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }

// Promos
$("#btnRedeem").onclick = async ()=>{
  const code = $("#promoCode").value.trim(); if(!code) return;
  try{
    const r = await jfetch("/promo/redeem", {method:"POST", body:JSON.stringify({code})});
    $("#promoResult").textContent = `Redeemed ${code}. New balance: ${fmt(r.balance)} DL`;
    $("#balanceVal").textContent = fmt(r.balance);
  }catch(e){ $("#promoResult").textContent = e.message; }
};

// Crash
async function updateCrash(){
  const s = await jfetch("/crash/state");
  $("#crashRound").textContent = s.id || "—";
  $("#crashState").textContent = s.status;
  $("#crashTime").textContent = s.time_label;
  $("#crashMult").textContent = (s.multiplier? s.multiplier.toFixed(2):1).toString()+"x";
  $("#crashStatus").textContent = s.message;
  const list = $("#lastBusts"); list.innerHTML = "";
  s.last_busts.forEach(v=>{
    const b = document.createElement("span");
    b.className="badge"; b.textContent = v.toFixed(2)+"x";
    list.appendChild(b);
  });
  const hist = $("#crashHistory"); hist.innerHTML="";
  s.history.forEach(h=>{
    const item = document.createElement("div");
    item.className="stat";
    item.textContent = `Bet ${fmt(h.bet)} @ ${h.cashout.toFixed(2)}x • Bust ${h.bust.toFixed(2)}x • Win ${fmt(h.win)} DL`;
    hist.appendChild(item);
  });
}
function startCrashLoop(){
  if(CRASH_TICK) clearInterval(CRASH_TICK);
  updateCrash();
  CRASH_TICK = setInterval(updateCrash, 1000);
}
$("#btnCrashBet").onclick = async ()=>{
  const bet = Number($("#crashBet").value||0);
  const cashout = Number($("#crashCashout").value||0);
  try{
    const r = await jfetch("/crash/bet", {method:"POST", body:JSON.stringify({bet, cashout})});
    $("#crashStatus").textContent = `Bet placed on round ${r.round_id}`;
  }catch(e){ alert(e.message); }
};
$("#btnCrashCashout").onclick = async ()=>{
  try{
    const r = await jfetch("/crash/cashout", {method:"POST"});
    alert(`Cashed out at ${r.multiplier.toFixed(2)}x for ${fmt(r.win)} DL`);
  }catch(e){ alert(e.message); }
};

// Mines
function renderMinesBoard(revealedMask, boardString, active=true){
  const board = $("#minesBoard");
  board.innerHTML="";
  for(let i=0;i<25;i++){
    const btn = document.createElement("button");
    btn.className = "tile";
    const revealed = (revealedMask & (1<<i))!==0;
    if(revealed) btn.classList.add("revealed");
    if(!active && boardString[i]==='1') btn.classList.add("mine");
    btn.textContent = revealed ? "✓" : "";
    btn.disabled = !active || revealed;
    btn.onclick = ()=> pickMines(i);
    board.appendChild(btn);
  }
}
async function startMines(){
  const bet = Number($("#minesBet").value||0);
  const mines = Number($("#minesCount").value||3);
  try{
    const r = await jfetch("/mines/start", {method:"POST", body:JSON.stringify({bet, mines})});
    $("#minesStatus").textContent = `Game #${r.id} started • hash ${r.hash}`;
    await minesState();
  }catch(e){ alert(e.message); }
}
$("#btnMinesStart").onclick = startMines;
async function pickMines(index){
  try{
    const r = await jfetch("/mines/pick", {method:"POST", body:JSON.stringify({index})});
    if(r.status==="lost"){
      $("#minesStatus").textContent = "Boom! You hit a mine.";
      renderMinesBoard(0, r.board, false);
      await minesHistory();
      await refreshMe();
      $("#minesControls").classList.add("hidden");
    }else{
      $("#minesPotential").textContent = fmt(r.potential_win);
      renderMinesBoard(r.picks, "0".repeat(25), true);
    }
  }catch(e){ alert(e.message); }
}
$("#btnMinesCashout").onclick = async ()=>{
  try{
    const r = await jfetch("/mines/cashout", {method:"POST"});
    $("#minesStatus").textContent = `Cashed: +${fmt(r.win)} DL`;
    renderMinesBoard(0, r.board, false);
    await refreshMe();
    await minesHistory();
    $("#minesControls").classList.add("hidden");
  }catch(e){ alert(e.message); }
};
async function minesState(){
  const s = await jfetch("/mines/state");
  if(!s){ $("#minesStatus").textContent = "No active game."; $("#minesControls").classList.add("hidden"); $("#minesBoard").innerHTML=""; return;}
  $("#minesControls").classList.remove("hidden");
  $("#minesBetShow").textContent = fmt(s.bet);
  $("#minesCountShow").textContent = s.mines;
  $("#minesStatus").textContent = `Active game #${s.id} • hash ${s.hash}`;
  renderMinesBoard(s.picks, "0".repeat(25), true);
}
async function minesHistory(){
  const list = $("#minesHistory"); list.innerHTML="";
  const rows = await jfetch("/mines/history");
  rows.forEach(r=>{
    const d = document.createElement("div");
    d.className="stat";
    d.textContent = `#${r.id} • ${r.status.toUpperCase()} • Bet ${fmt(r.bet)} • Mines ${r.mines} • Win ${fmt(r.win)} DL`;
    list.appendChild(d);
  });
}

// Profile
$("#btnSetName").onclick = async ()=>{
  const name = $("#newName").value.trim();
  if(!name) return;
  try{
    const r = await jfetch("/name", {method:"POST", body:JSON.stringify({name})});
    alert("Updated name to "+r.name);
    await refreshMe();
    updateProfile();
  }catch(e){ alert(e.message); }
};
$("#btnToggleAnon").onclick = async ()=>{
  try{
    const r = await jfetch("/anon", {method:"POST", body:JSON.stringify({toggle:true})});
    alert("Anonymous mode: "+(r.is_anon?"ON":"OFF"));
    await updateProfile();
  }catch(e){ alert(e.message); }
};
async function updateProfile(){
  if(!ME){ $("#meInfo").innerHTML = "<div class='notice'>Not logged in.</div>"; return; }
  const fresh = await jfetch("/me");
  $("#balanceVal").textContent = fmt(fresh.balance);
  $("#meInfo").innerHTML = `
    <div>Display name: <span class="code">${fresh.display_name}</span> ${fresh.is_anon?'<span class="badge">Anon</span>':''}</div>
    <div>Level: <span class="code">${fresh.level}</span> (${fresh.xp} XP)</div>
    <div>Your referral link: <span class="code copy" id="copyRef">${location.origin}/ref/${fresh.name_lower}</span></div>
  `;
  $("#copyRef").onclick = async ()=>{
    try{ await navigator.clipboard.writeText(location.origin+"/ref/"+fresh.name_lower); alert("Copied!"); }catch(e){ alert("Copy failed."); }
  };
}

// Init
(async ()=>{
  await refreshMe();
  await loadLeaderboard();
  await minesHistory();
  await minesState();
  startCrashLoop();
  updateProfile();
})();

// Referral cookie helper
(function setRefFromQuery(){
  const m = location.search.match(/[?&]ref=([a-z0-9_-]{3,20})/i);
  if(m){ document.cookie = "ref_src="+encodeURIComponent(m[1])+"; max-age="+(30*86400)+"; path=/"; }
})();

</script>
</body></html>
"""

# ---------- OAuth & Utility ----------

def _discord_oauth_url(scopes: List[str], redirect_uri: str) -> str:
    params = {
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": " ".join(scopes),
        "prompt": "consent"
    }
    return f"{DISCORD_API}/oauth2/authorize?{urlencode(params)}"

async def _exchange_code(code: str, redirect_uri: str) -> dict:
    async with httpx.AsyncClient(timeout=20) as cx:
        resp = await cx.post(f"{DISCORD_API}/oauth2/token",
                             data={
                                 "client_id": CLIENT_ID,
                                 "client_secret": CLIENT_SECRET,
                                 "grant_type": "authorization_code",
                                 "code": code,
                                 "redirect_uri": redirect_uri,
                             },
                             headers={"Content-Type": "application/x-www-form-urlencoded"})
        resp.raise_for_status()
        return resp.json()

async def _discord_me(access_token: str) -> dict:
    async with httpx.AsyncClient(timeout=20) as cx:
        r = await cx.get(f"{DISCORD_API}/users/@me",
                         headers={"Authorization": f"Bearer {access_token}"})
        r.raise_for_status()
        return r.json()

async def _guild_add_member(user_id: str, access_token: str) -> bool:
    if not (GUILD_ID and BOT_TOKEN):
        return False
    async with httpx.AsyncClient(timeout=20) as cx:
        r = await cx.put(f"{DISCORD_API}/guilds/{GUILD_ID}/members/{user_id}",
                         json={"access_token": access_token},
                         headers={"Authorization": f"Bot {BOT_TOKEN}"})
        # 201 created or 204 no content are success; 200 can also occur
        return r.status_code in (200, 201, 204)

# ---------- Routes ----------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return HTML_TEMPLATE

@app.get("/login")
async def login():
    return RedirectResponse(_discord_oauth_url(["identify"], OAUTH_REDIRECT or ""))

@app.get("/login/join")
async def login_join():
    # Ask for identify + guilds.join, callback to OAUTH_JOIN_REDIRECT
    if not (OAUTH_JOIN_REDIRECT and GUILD_ID and BOT_TOKEN):
        # Fallback to invite if configured
        if DISCORD_INVITE:
            return RedirectResponse(DISCORD_INVITE)
        raise HTTPException(400, "Join flow is not configured.")
    return RedirectResponse(_discord_oauth_url(["identify","guilds.join"], OAUTH_JOIN_REDIRECT))

@app.get("/oauth/callback")
async def oauth_callback(request: Request, code: str = Query(...)):
    tok = await _exchange_code(code, OAUTH_REDIRECT or "")
    user = await _discord_me(tok["access_token"])
    uid = str(user["id"])
    uname = user.get("username","user")
    ensure_profile_row(uid)
    # If referral cookie exists, try to attach
    ref_src = request.cookies.get("ref_src")
    if ref_src:
        ref_uid = find_user_id_by_name_lower(ref_src.lower())
        if ref_uid and ref_uid != uid:
            add_referral(ref_uid, get_profile_name(ref_uid) or ref_src, uid)
    resp = RedirectResponse("/")
    _set_session(resp, {"id": uid, "username": uname, "token_hint": "set"})
    return resp

@app.get("/oauth/join")
async def oauth_join(request: Request, code: str = Query(...)):
    if not (OAUTH_JOIN_REDIRECT and GUILD_ID and BOT_TOKEN):
        raise HTTPException(400, "Join flow is not configured.")
    tok = await _exchange_code(code, OAUTH_JOIN_REDIRECT)
    user = await _discord_me(tok["access_token"])
    uid = str(user["id"])
    uname = user.get("username","user")
    ensure_profile_row(uid)
    joined = await _guild_add_member(uid, tok["access_token"])
    resp = RedirectResponse("/?joined="+("1" if joined else "0"))
    _set_session(resp, {"id": uid, "username": uname, "token_hint": "set"})
    return resp

@app.get("/logout")
async def logout():
    resp = RedirectResponse("/")
    _clear_session(resp)
    return resp

# -------- Helper: auto-advance crash state on poll --------
def _advance_crash_state():
    r = load_round()
    if not r:
        ensure_betting_round()
        return
    now = now_utc()
    if r["status"] == "betting" and now >= r["betting_ends_at"]:
        begin_running(r["id"])
        r = load_round()
    if r["status"] == "running" and now >= r["expected_end_at"]:
        finish_round(r["id"])
        create_next_betting()

# -------- API: Me / Profile --------
@app.get("/me")
async def me(request: Request):
    try:
        sess = _require_session(request)
    except HTTPException:
        raise
    uid = str(sess["id"])
    ensure_profile_row(uid)
    name = get_profile_name(uid) or f"user_{uid[-4:]}"
    info = profile_info(uid)
    # Include name_lower for referral link convenience
    return {
        "id": uid,
        "username": sess.get("username") or name,
        "display_name": name,
        "name_lower": name.lower(),
        **info
    }

class NameModel(BaseModel):
    name: str

@app.post("/name")
async def set_name(request: Request, body: NameModel):
    sess = _require_session(request)
    r = set_profile_name(str(sess["id"]), body.name.strip())
    return r

class AnonModel(BaseModel):
    toggle: bool = True

@app.post("/anon")
async def anon(request: Request, body: AnonModel):
    sess = _require_session(request)
    cur = public_profile(str(sess["id"]))  # ensures row exists
    new = set_profile_is_anon(str(sess["id"]), not bool(cur and cur.get("is_anon")))
    return new

# -------- API: Promo --------
class PromoRedeem(BaseModel):
    code: str

@app.post("/promo/redeem")
async def promo_redeem(request: Request, body: PromoRedeem):
    sess = _require_session(request)
    try:
        bal = redeem_promo(str(sess["id"]), body.code)
        return {"ok": True, "balance": float(bal)}
    except PromoAlreadyRedeemed as e:
        raise HTTPException(400, str(e))
    except PromoInvalid as e:
        raise HTTPException(400, str(e))
    except PromoExpired as e:
        raise HTTPException(400, str(e))
    except PromoExhausted as e:
        raise HTTPException(400, str(e))

# -------- API: Referrals --------
@app.get("/ref/{name_lower}")
async def ref_touch(name_lower: str):
    # Set a cookie and bounce home
    resp = RedirectResponse("/")
    resp.set_cookie("ref_src", value=name_lower.lower(), max_age=30*86400, samesite="lax")
    return resp

@app.get("/referrals/count")
async def referrals_count(request: Request):
    sess = _require_session(request)
    cnt = referrals_count_for(str(sess["id"]))
    return {"count": cnt}

# -------- API: Chat --------
@app.get("/chat")
async def chat_get(request: Request, since_id: int = 0, limit: int = 50):
    uid = None
    try:
        uid = str(_require_session(request)["id"])
    except HTTPException:
        uid = None
    return chat_fetch(since_id, min(200, max(1, limit)), uid)

class ChatMsg(BaseModel):
    text: str
    private_to: Optional[str] = None

@app.post("/chat")
async def chat_post(request: Request, body: ChatMsg):
    sess = _require_session(request)
    name = get_profile_name(str(sess["id"])) or sess.get("username") or "user"
    try:
        r = chat_insert(str(sess["id"]), name, body.text, body.private_to)
        return r
    except PermissionError as e:
        raise HTTPException(403, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))

# -------- API: Crash --------
@app.get("/crash/state")
async def crash_state(request: Request):
    _advance_crash_state()
    r = load_round()
    if not r:
        rid, _ = ensure_betting_round()
        r = load_round()
    now = now_utc()
    status = r["status"]
    mult = None
    if status == "running":
        mult = current_multiplier(r["started_at"], r["expected_end_at"], r["bust"], now)
    msg = "Place your bets!" if status=="betting" else ("Flying…" if status=="running" else f"Busted at {r['bust']:.2f}x")
    time_label = ""
    if status=="betting":
        secs = int((r["betting_ends_at"] - now).total_seconds())
        time_label = f"bets close in {max(0,secs)}s"
    elif status=="running":
        secs = int((r["expected_end_at"] - now).total_seconds())
        time_label = f"~{max(0,secs)}s to bust"
    else:
        time_label = "ended"
    sess = None
    try: sess = _require_session(request)
    except HTTPException: pass
    history = your_history(str(sess["id"])) if sess else []
    return {
        "id": r["id"],
        "status": status,
        "multiplier": mult,
        "message": msg,
        "time_label": time_label,
        "last_busts": last_busts(),
        "history": history
    }

class CrashBet(BaseModel):
    bet: float
    cashout: float

@app.post("/crash/bet")
async def crash_bet(request: Request, body: CrashBet):
    sess = _require_session(request)
    try:
        r = place_bet(str(sess["id"]), D(body.bet), float(body.cashout))
        return r
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.post("/crash/cashout")
async def crash_cashout(request: Request):
    sess = _require_session(request)
    try:
        r = cashout_now(str(sess["id"]))
        return r
    except ValueError as e:
        raise HTTPException(400, str(e))

# -------- API: Mines --------
class MinesStart(BaseModel):
    bet: float
    mines: int

class MinesPick(BaseModel):
    index: int

@app.post("/mines/start")
async def api_mines_start(request: Request, body: MinesStart):
    sess = _require_session(request)
    try:
        r = mines_start(str(sess["id"]), D(body.bet), int(body.mines))
        return r
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.post("/mines/pick")
async def api_mines_pick(request: Request, body: MinesPick):
    sess = _require_session(request)
    try:
        r = mines_pick(str(sess["id"]), int(body.index))
        return r
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.post("/mines/cashout")
async def api_mines_cashout(request: Request):
    sess = _require_session(request)
    try:
        r = mines_cashout(str(sess["id"]))
        return r
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.get("/mines/state")
async def api_mines_state(request: Request):
    try:
        sess = _require_session(request)
    except HTTPException:
        raise HTTPException(401, "Login required")
    return mines_state(str(sess["id"]))

@app.get("/mines/history")
async def api_mines_history(request: Request, limit: int = 10):
    try:
        sess = _require_session(request)
    except HTTPException:
        raise HTTPException(401, "Login required")
    return mines_history(str(sess["id"]), min(50, max(1, limit)))

# ---------- Static referral utility (already handled via /ref/{name}) ----------

# ---------- Run ----------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=bool(os.getenv("RELOAD")))
