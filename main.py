# app/main.py
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
.tab:hover{opacity:1;transform:translateY(-1px)}.tab.active{background:linear-gradient(135deg,#3b82f6,#22c1dc);color:#051326;box-shadow:0 6px 16px rgba(59,130,246,.25);opacity:1}
.right{display:flex;gap:8px;align-items:center;margin-left:12px}.chip{background:#0c1631;border:1px solid var(--border);color:#dce7ff;padding:6px 10px;border-radius:999px;font-size:12px;white-space:nowrap;cursor:pointer}
.avatar{width:34px;height:34px;border-radius:50%;object-fit:cover;border:1px solid var(--border);cursor:pointer}
.btn{display:inline-flex;align-items:center;gap:8px;padding:10px 14px;border-radius:12px;border:1px solid var(--border);background:linear-gradient(180deg,#0e1833,#0b1326);cursor:pointer;font-weight:600}
.btn.primary{background:linear-gradient(135deg,#3b82f6,#22c1dc);border-color:transparent}.btn.ghost{background:#162a52;border:1px solid var(--border);color:#eaf2ff}
.btn.cashout{background:linear-gradient(135deg,#22c55e,#16a34a);border-color:transparent;box-shadow:0 6px 14px rgba(34,197,94,.25);font-weight:800}
.btn.cashout[disabled]{filter:grayscale(.5) brightness(.8);opacity:.8;cursor:not-allowed}
.big{font-size:22px;font-weight:900}.label{font-size:12px;color:var(--muted);letter-spacing:.2px;text-transform:uppercase}.muted{color:var(--muted)}
.games-grid{display:grid;gap:14px;grid-template-columns:1fr}@media(min-width:700px){.games-grid{grid-template-columns:1fr 1fr}}@media(min-width:1020px){.games-grid{grid-template-columns:1fr 1fr 1fr}}
.game-card{position:relative;min-height:130px;display:flex;flex-direction:column;justify-content:flex-end;gap:4px;background:linear-gradient(180deg,#0f1a33,#0c152a);border:1px solid var(--border);border-radius:16px;padding:16px;cursor:pointer;transition:transform .08s ease, box-shadow .12s ease, border-color .12s ease, background .18s ease;overflow:hidden}
.game-card:hover{transform:translateY(-2px);box-shadow:0 8px 18px rgba(0,0,0,.25)}.game-card .title{font-size:20px;font-weight:800}
.ribbon{position:absolute;top:12px;right:-32px;transform:rotate(35deg);background:linear-gradient(135deg,#f59e0b,#fb923c);color:#1a1206;font-weight:900;padding:6px 50px;border:1px solid rgba(0,0,0,.2);text-shadow:0 1px 0 rgba(255,255,255,.2)}
.cr-graph-wrap{position:relative;height:240px;background:#0e1833;border:1px solid var(--border);border-radius:16px;overflow:hidden}
canvas{display:block;width:100%;height:100%}.boom{position:absolute;inset:0;pointer-events:none;opacity:0}
.boom.bang{animation:bang .6s ease-out}@keyframes bang{0%{opacity:.95;background:radial-gradient(350px 350px at var(--x,50%) var(--y,50%), rgba(255,255,255,.9), rgba(239,68,68,.6) 40%, transparent 70%)}100%{opacity:0;background:radial-gradient(800px 800px at var(--x,50%) var(--y,50%), rgba(255,255,255,0), rgba(239,68,68,0) 40%, transparent 75%)}}
.mines-two{grid-template-columns:360px 1fr!important;align-items:stretch;display:grid;gap:16px}
.mines-wrap{display:grid;place-items:center;min-height:420px;padding:6px}
.mines-grid{--cell:clamp(48px, min(calc((100vw - 440px)/5), calc((100vh - 320px)/5)), 110px);display:grid;gap:10px;grid-template-columns:repeat(5, var(--cell));justify-content:center;align-content:center;padding:6px;width:100%}
.tile{position:relative;width:var(--cell);aspect-ratio:1/1;border-radius:clamp(10px, calc(var(--cell)*0.18), 16px);border:1px solid var(--border);background:radial-gradient(120% 120% at 30% 0%, #19264f 0%, #0c152a 55%), linear-gradient(180deg,#0f1936,#0c152a);display:flex;align-items:center;justify-content:center;font-weight:900;font-size:clamp(13px, calc(var(--cell)*0.34), 22px);cursor:pointer;user-select:none;transition:transform .09s ease, box-shadow .14s ease, background .18s ease, border-color .14s ease, opacity .14s ease;box-shadow:0 8px 22px rgba(0,0,0,.35), inset 0 0 0 1px rgba(255,255,255,.03);overflow:hidden}
.tile::after{content:"";position:absolute;inset:0;background:linear-gradient(145deg, rgba(255,255,255,.18), transparent 40%);mix-blend-mode:soft-light;opacity:.22;transition:opacity .2s ease}
.tile:hover{transform:translateY(-1px);box-shadow:0 10px 26px rgba(0,0,0,.45), inset 0 0 0 1px rgba(255,255,255,.05)}
.tile .icon{filter:drop-shadow(0 2px 6px rgba(0,0,0,.45))}
.tile.safe{background:linear-gradient(135deg,#16a34a 0%, #22c55e 70%);border-color:transparent;color:#06240f}
.tile.mine{background:linear-gradient(135deg,#ef4444 0%, #b91c1c 70%);border-color:transparent;color:#260808}
.tile.revealed{cursor:default}.tile.pop{animation:pop .2s ease}
@keyframes pop{from{transform:scale(.92);opacity:.7}to{transform:scale(1);opacity:1}}
.tile.explode{animation:shake .4s ease-in-out}
.tile.explode::before{content:"";position:absolute;inset:-2px;border-radius:inherit;background:radial-gradient(circle, rgba(255,255,255,.85), rgba(239,68,68,.6) 40%, transparent 70%);opacity:0;animation:exflash .6s ease-out}
@keyframes exflash{0%{opacity:.95;transform:scale(.9)}80%{opacity:.15;transform:scale(1.05)}100%{opacity:0;transform:scale(1)}}
@keyframes shake{0%,100%{transform:translate(0,0)}20%{transform:translate(-2px,-1px)}40%{transform:translate(3px,1px)}60%{transform:translate(-2px,2px)}80%{transform:translate(1px,-2px)}}
.mines-stats{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}.stat{background:#0c1631;border:1px solid var(--border);color:#dce7ff;padding:6px 10px;border-radius:999px;font-size:12px;white-space:nowrap}
.modal{position:fixed;inset:0;display:none;align-items:center;justify-content:center;background:rgba(3,6,12,.6);z-index:50}
.modal .box{width:min(640px, 92vw);background:linear-gradient(180deg,#0f1a33,#0c1429);border:1px solid var(--border);border-radius:18px;padding:16px;box-shadow:0 10px 30px rgba(0,0,0,.4)}
.chat-drawer{position:fixed;right:0;top:64px;bottom:0;width:var(--chatW);max-width:90vw;transform:translateX(100%);transition:transform .2s ease-out;background:linear-gradient(180deg,#0f1a33,#0b1326);border-left:1px solid var(--border);display:flex;flex-direction:column;z-index:40}
.chat-drawer.open{transform:translateX(0)}.chat-head{display:flex;align-items:center;justify-content:space-between;padding:10px 12px;border-bottom:1px solid var(--border)}
.chat-body{flex:1;overflow:auto;padding:10px 12px}.chat-input{display:flex;gap:8px;padding:10px 12px;border-top:1px solid var(--border)}
.chat-input input{flex:1}.msg{margin-bottom:12px;padding-bottom:8px;border-bottom:1px dashed rgba(255,255,255,.04);position:relative}
.msghead{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.msghead .time{margin-left:auto;color:var(--muted);font-size:12px}
.badge{font-size:10px;padding:3px 7px;border-radius:999px;border:1px solid var(--border);letter-spacing:.2px}
.badge.member{background:#0c1631;color:#cfe6ff}.badge.admin{background:linear-gradient(135deg,#f59e0b,#fb923c);color:#1a1206;border-color:rgba(0,0,0,.2);font-weight:900}.badge.owner{background:linear-gradient(135deg,#3b82f6,#22c1dc);color:#041018;border-color:transparent;font-weight:900}
.level{font-size:10px;padding:3px 7px;border-radius:999px;background:#0b1f3a;color:#cfe6ff;border:1px solid var(--border)}
.user-link{cursor:pointer;font-weight:800;padding:2px 6px;border-radius:8px;background:#0b1f3a;border:1px solid var(--border)}
.disabled-note{padding:8px 12px;font-size:13px;color:#dbe6ff;background:#0c1631;border-bottom:1px solid var(--border)}
.kebab{position:absolute;right:0;top:0;opacity:.75}.kebab button{background:transparent;border:none;color:#cfe3ff;cursor:pointer;padding:6px}
.menu{position:absolute;right:6px;top:24px;background:#0c1631;border:1px solid var(--border);border-radius:12px;padding:6px;display:none;min-width:160px;z-index:5}
.menu.open{display:block}.menu .item{padding:8px 10px;border-radius:8px;cursor:pointer;font-size:14px}.menu .item:hover{background:#11234a}.menu .warn{color:#ffd7d7}.menu .danger{color:#ffb3b3}
.owner{margin-top:16px;border-top:1px dashed var(--border);padding-top:12px}.owner .panel{display:grid;gap:12px}.owner .panel .card{background:linear-gradient(180deg,#0f1833,#0c1428)}
.section-title{font-size:14px;font-weight:800;margin-bottom:8px}
.soon-hero{position:relative;overflow:hidden;border-radius:16px;border:1px solid var(--border);background:radial-gradient(1200px 500px at -10% -20%, rgba(59,130,246,.25), transparent 60%), linear-gradient(180deg,#0f1a33,#0b1326);padding:22px}
.soon-hero h2{margin:0;font-size:28px}.soon-badge{position:absolute;top:14px;right:14px;background:linear-gradient(135deg,#f59e0b,#fb923c);color:#1a1206;padding:6px 10px;border-radius:999px;font-weight:900;border:1px solid rgba(0,0,0,.25)}
.soon-grid{display:grid;gap:12px;grid-template-columns:1fr;margin-top:12px}@media(min-width:800px){.soon-grid{grid-template-columns:1fr 1fr}}
.soon-card{background:linear-gradient(180deg,#0f1a33,#0b1326);border:1px solid var(--border);border-radius:16px;padding:14px}
.fab{position:fixed;right:18px;bottom:18px;width:56px;height:56px;border-radius:50%;background:linear-gradient(135deg,#3b82f6,#22c1dc);border:none;cursor:pointer;display:flex;align-items:center;justify-content:center;box-shadow:0 14px 30px rgba(59,130,246,.35), 0 4px 10px rgba(0,0,0,.35);z-index:45;transition:right .2s ease}
.fab:hover{transform:translateY(-1px);box-shadow:0 18px 40px rgba(59,130,246,.45), 0 6px 14px rgba(0,0,0,.45)}
.fab svg{width:26px;height:26px;fill:#041018}
.lb-controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
.seg{display:flex;border:1px solid var(--border);border-radius:12px;overflow:hidden}
.seg button{padding:8px 12px;background:#0c1631;color:#dce7ff;border:none;cursor:pointer}
.seg button.active{background:linear-gradient(135deg,#3b82f6,#22c1dc);color:#051326;font-weight:800}
table{width:100%;border-collapse:collapse}th,td{padding:10px;border-bottom:1px solid rgba(255,255,255,.06);text-align:left}
tr.me-row{background:linear-gradient(90deg, rgba(34,197,94,.12), transparent 60%)}tr.anon td.name{color:#9db4e4;font-style:italic}
.countdown{font-size:12px;color:var(--muted)}.hint{font-size:12px;color:var(--muted);margin-top:6px}
.switch{display:inline-flex;align-items:center;gap:10px}.switch input{width:auto}
</style>
</head>
<body>
  <div class="header">
    <div class="header-inner container">
      <div class="left">
        <a class="brand" href="#" id="homeLink"><span class="logo"></span> 💎 DL Bank</a>
        <div class="tabs">
          <a class="tab active" id="tab-games">Games</a>
          <a class="tab" id="tab-ref">Referral</a>
          <a class="tab" id="tab-promo">Promo Codes</a>
          <a class="tab" id="tab-lb">Leaderboard</a>
          <a class="tab" id="tab-settings">Settings</a>
        </div>
      </div>
      <div class="right" id="authArea"></div>
    </div>
  </div>

  <div class="container" style="padding-top:16px">
    <div id="page-games">
      <div class="card">
        <div class="games-grid">
          <div class="game-card" id="openCrash" style="background-image: radial-gradient(600px 280px at 10% -10%, rgba(59,130,246,.25), transparent 60%);">
            <div class="title">🚀 Crash</div><div class="muted">Shared rounds • 10s betting • Live cashout</div>
          </div>
          <div class="game-card" id="openMines" style="background-image: radial-gradient(600px 280px at 85% -20%, rgba(34,197,94,.25), transparent 60%);">
            <div class="title">💣 Mines</div><div class="muted">5×5 board • Choose mines • Cash out anytime</div>
          </div>
          <div class="game-card"><div class="ribbon">COMING SOON</div><div class="title">🎯 Limbo</div><div class="muted">Pick a multiplier and pray</div></div>
          <div class="game-card"><div class="ribbon">COMING SOON</div><div class="title">🗼 Towers</div><div class="muted">Climb floors • Avoid the trap</div></div>
          <div class="game-card"><div class="ribbon">COMING SOON</div><div class="title">🎲 Keno</div><div class="muted">Pick numbers • Big hits</div></div>
          <div class="game-card"><div class="ribbon">COMING SOON</div><div class="title">🟡 Plinko</div><div class="muted">Edge pockets go brr</div></div>
          <div class="game-card"><div class="ribbon">COMING SOON</div><div class="title">🃏 Blackjack</div><div class="muted">21 or bust • Skill + luck</div></div>
          <div class="game-card"><div class="ribbon">COMING SOON</div><div class="title">📈 Pump</div><div class="muted">Like Crash… crankier</div></div>
          <div class="game-card"><div class="ribbon">COMING SOON</div><div class="title">🪙 Coinflip</div><div class="muted">50/50 • Double up</div></div>
        </div>
      </div>
    </div>

    <div id="page-crash" style="display:none">
      <div class="card">
        <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap">
          <div style="display:flex;align-items:baseline;gap:10px">
            <div class="big" id="crNow">0.00×</div><div class="muted" id="crHint">Loading…</div>
          </div>
          <button class="chip" id="backToGames">← Games</button>
        </div>
        <div class="cr-graph-wrap" style="margin-top:10px"><canvas id="crCanvas"></canvas><div id="crBoom" class="boom"></div></div>
        <div style="margin-top:12px"><div class="label" style="margin-bottom:4px">Previous Busts</div><div id="lastBusts" class="muted">Loading last rounds…</div></div>
        <div class="games-grid" style="grid-template-columns:1fr 1fr;gap:12px;margin-top:8px">
          <div class="field"><div class="label">Bet (DL)</div><input id="crBet" type="number" min="1" step="0.01" placeholder="min 1.00"/></div>
          <div class="field"><div class="label">Auto Cashout (×) — optional</div><input id="crCash" type="number" min="1.01" step="0.01" placeholder="e.g. 2.00"/></div>
        </div>
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:8px">
          <button class="btn primary" id="crPlace">Place Bet</button>
          <button class="btn cashout" id="crCashout" style="display:none">💸 Cash Out</button>
          <span id="crMsg" class="muted"></span>
        </div>
        <div class="card" style="margin-top:14px"><div class="label">Your recent rounds</div><div id="crLast" class="muted">—</div></div>
      </div>
    </div>

    <div id="page-mines" style="display:none">
      <div class="card">
        <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap">
          <div class="big">💣 Mines</div><button class="chip" id="backToGames2">← Games</button>
        </div>
        <div class="mines-two" style="margin-top:12px">
          <div>
            <div class="field"><div class="label">Bet (DL)</div><input id="mBet" type="number" min="1" step="0.01" placeholder="min 1.00"/></div>
            <div class="field" style="margin-top:10px"><div class="label">Mines (1–24)</div><input id="mMines" type="number" min="1" max="24" step="1" value="3"/></div>
            <div class="mines-stats"><span class="stat" id="mHash">Commit: —</span><span class="stat" id="mStatus">Status: —</span><span class="stat" id="mPicks">Picks: 0</span><span class="stat" id="mBombs">Mines: 3</span></div>
            <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:12px">
              <button class="btn primary" id="mStart">Start Game</button>
              <button class="btn cashout" id="mCash" style="display:none">💸 Cash Out</button>
              <span id="mMsg" class="muted"></span>
            </div>
            <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:8px"><span class="stat" id="mMult">Multiplier: 1.0000×</span><span class="stat" id="mPotential">Potential: —</span></div>
            <div class="card" style="margin-top:14px"><div class="label">Recent Mines Games</div><div id="mHist" class="muted">—</div></div>
          </div>
          <div class="mines-wrap"><div class="mines-grid" id="mGrid"></div></div>
        </div>
      </div>
    </div>

    <div id="page-ref" style="display:none"><div class="card"><div class="label">Referral</div><div id="refContent">Loading…</div></div></div>
    <div id="page-promo" style="display:none">
      <div class="card">
        <div class="label">Promo Codes</div>
        <div class="games-grid" style="grid-template-columns:1fr 1fr">
          <div>
            <div class="label">Redeem a code</div>
            <div style="display:flex;gap:8px;align-items:center">
              <input id="promoInput" placeholder="e.g. WELCOME10"/><button class="btn primary" id="redeemBtn">Redeem</button>
            </div>
            <div id="promoMsg" class="muted" style="margin-top:8px"></div>
          </div>
          <div>
            <div class="label">Your redemptions</div><div id="myCodes" class="muted">—</div>
          </div>
        </div>
      </div>
    </div>

    <div id="page-lb" style="display:none">
      <div class="card">
        <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap">
          <div class="big">🏆 Leaderboard — Top Wagered</div><div class="countdown" id="lbCountdown">—</div>
        </div>
        <div class="lb-controls" style="margin-top:10px">
          <div class="seg" id="lbSeg"><button data-period="daily" class="active">Daily</button><button data-period="monthly">Monthly</button><button data-period="alltime">All-time</button></div>
          <span class="hint">Anonymous players are shown as “Anonymous”. Amounts are hidden for anonymous users.</span>
        </div>
        <div id="lbWrap" class="muted">Loading…</div>
      </div>
    </div>

    <div id="page-settings" style="display:none">
      <div class="card">
        <div class="label">Settings</div>
        <div style="margin-top:8px">
          <label class="switch">
            <input type="checkbox" id="anonToggle"/>
            <span><strong>Anonymous Mode</strong> — hide your name & wager amounts from others. You’ll still appear on leaderboards as “Anonymous”.</span>
          </label>
          <div class="hint">Takes effect immediately. Admins cannot override this for public views.</div>
          <div id="setMsg" class="muted" style="margin-top:8px"></div>
        </div>
      </div>
    </div>

    <div id="page-profile" style="display:none">
      <div class="card">
        <div class="label">Profile</div><div id="profileBox">Loading…</div>
        <div id="ownerPanel" class="owner" style="display:none">
          <div class="panel">
            <div class="card">
              <div class="section-title">Adjust Balance</div>
              <div class="row cols-4">
                <div class="field"><div class="label">Discord ID or &lt;@mention&gt;</div><input id="tIdent" placeholder="ID or <@id>"/></div>
                <div class="field"><div class="label">Amount (+/- DL)</div><input id="tAmt" type="text" placeholder="10 or -5.25"/></div>
                <div class="field"><div class="label">Reason (optional)</div><input id="tReason" placeholder="promo/correction/prize"/></div>
                <div style="align-self:end"><button class="btn primary" id="tApply">Apply</button></div>
              </div>
              <div id="tMsg" class="muted" style="margin-top:8px"></div>
            </div>
            <div class="card">
              <div class="section-title">Roles</div>
              <div class="row cols-3">
                <div class="field"><div class="label">Target</div><input id="rIdent" placeholder="ID or <@id>"/></div>
                <button class="btn" id="rAdmin">Make ADMIN</button><button class="btn" id="rMember">Make MEMBER</button>
              </div>
              <div id="rMsg" class="muted" style="margin-top:8px"></div>
            </div>
            <div class="card">
              <div class="section-title">Timeouts</div>
              <div class="row cols-5">
                <div class="field"><div class="label">Target</div><input id="xIdent" placeholder="ID or <@id>"/></div>
                <div class="field"><div class="label">Seconds</div><input id="xSecs" type="number" value="600"/></div>
                <div class="field"><div class="label">Reason</div><input id="xReason" placeholder="spam / rude / etc"/></div>
                <button class="btn" id="xSite">Site Only</button><button class="btn" id="xBoth">Site + Discord</button>
              </div>
              <div id="xMsg" class="muted" style="margin-top:8px"></div>
            </div>
            <div class="card">
              <div class="section-title">Create Promo Code</div>
              <div class="row cols-3">
                <div class="field"><div class="label">Code (optional)</div><input id="cCode" placeholder="auto-generate if empty"/></div>
                <div class="field"><div class="label">Amount (DL)</div><input id="cAmount" type="text" placeholder="e.g. 10 or 1.24"/></div>
                <div class="field"><div class="label">Max Uses</div><input id="cMax" type="number" placeholder="e.g. 100"/></div>
              </div>
              <div style="margin-top:8px"><button class="btn primary" id="cMake">Create</button> <span id="cMsg" class="muted"></span></div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div id="page-limbo" style="display:none"><div class="soon-hero"><div class="soon-badge">Coming Soon</div><h2>🎯 Limbo</h2><p>Higher risk, higher reward.</p></div></div>
    <div id="page-towers" style="display:none"><div class="soon-hero"><div class="soon-badge">Coming Soon</div><h2>🗼 Towers</h2><p>Pick a safe tile per floor.</p></div></div>
    <div id="page-keno" style="display:none"><div class="soon-hero"><div class="soon-badge">Coming Soon</div><h2>🎲 Keno</h2><p>Choose numbers and hope.</p></div></div>
    <div id="page-plinko" style="display:none"><div class="soon-hero"><div class="soon-badge">Coming Soon</div><h2>🟡 Plinko</h2><p>Edge pockets pay big.</p></div></div>
    <div id="page-blackjack" style="display:none"><div class="soon-hero"><div class="soon-badge">Coming Soon</div><h2>🃏 Blackjack</h2><p>Beat 21.</p></div></div>
    <div id="page-pump" style="display:none"><div class="soon-hero"><div class="soon-badge">Coming Soon</div><h2>📈 Pump</h2><p>Ride the curve, cash before pop.</p></div></div>
    <div id="page-coinflip" style="display:none"><div class="soon-hero"><div class="soon-badge">Coming Soon</div><h2>🪙 Coinflip</h2><p>50/50. Double up.</p></div></div>
  </div>

  <button class="fab" id="fabChat" title="Open chat"><svg viewBox="0 0 24 24"><path d="M4 4h16v12H7l-3 3V4z"/></svg></button>
  <div class="chat-drawer" id="chatDrawer">
    <div class="chat-head"><div>Global Chat <span id="chatNote" class="muted"></span></div><button class="chip" id="chatClose">Close</button></div>
    <div class="disabled-note" id="chatDisabled" style="display:none"></div>
    <div class="chat-body" id="chatBody"></div>
    <div class="chat-input"><input id="chatText" placeholder="Say something…  (tip: .tip <UserID> <amount>)"/><button class="btn primary" id="chatSend">Send</button></div>
  </div>

  <div class="modal" id="pm"><div class="box"><div style="display:flex;align-items:center;justify-content:space-between"><div class="big" id="pmTitle">Profile</div><button class="chip" id="pmClose">Close</button></div><div id="pmBody" style="margin-top:10px">Loading…</div></div></div>

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
const pages = ['page-games','page-crash','page-mines','page-ref','page-promo','page-lb','page-settings','page-profile','page-limbo','page-towers','page-keno','page-plinko','page-blackjack','page-pump','page-coinflip'];
function showOnly(id){
  for(const p of pages){ const el = qs(p); if(el) el.style.display = (p===id) ? '' : 'none'; }
  const map = {'page-games':'tab-games','page-ref':'tab-ref','page-promo':'tab-promo','page-lb':'tab-lb','page-settings':'tab-settings'};
  for(const t of ['tab-games','tab-ref','tab-promo','tab-lb','tab-settings']){
    const el = qs(t); if(el) el.classList.toggle('active', map[id]===t);
  }
}
async function renderHeader(){
  try{
    const me = await j('/api/me');
    const bal = await j('/api/balance');
    qs('authArea').innerHTML = `<span class="chip">Balance: <strong>${fmtDL(bal.balance)}</strong></span><a class="btn ghost" id="btnProfile">Profile</a><img class="avatar" id="btnLogout" src="${me.avatar_url||''}" title="${me.username||'user'}"/>`;
    qs('btnProfile').onclick = ()=>{ showOnly('page-profile'); renderProfile(); };
    qs('btnLogout').onclick = ()=>{ location.href = '/logout'; };
  }catch(_){
    qs('authArea').innerHTML = `<a class="btn primary" href="/login">Login with Discord</a>`;
  }
}
qs('homeLink').onclick = (e)=>{ e.preventDefault(); showOnly('page-games'); };
qs('tab-games').onclick = ()=> showOnly('page-games');
qs('tab-ref').onclick = ()=> showOnly('page-ref');
qs('tab-promo').onclick = ()=> { showOnly('page-promo'); renderPromo(); };
qs('tab-lb').onclick = ()=> { showOnly('page-lb'); refreshLeaderboard(); };
qs('tab-settings').onclick = ()=> { showOnly('page-settings'); loadSettings(); };

// Promo
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

// Referral
(async ()=>{ try{
  const st = await j('/api/referral/state');
  qs('refContent').innerHTML = `Your referral name: <strong>${(st && st.name) ? st.name : '—'}</strong>`;
}catch(_){ qs('refContent').textContent='—'; } })();

// Profile & Owner Panel
async function renderProfile(){
  try{
    const p = await j('/api/profile');
    const role = p.role || 'member';
    const isOwner = (role==='owner') || (String(p.id||'') === String('__OWNER_ID__'));
    qs('profileBox').innerHTML = `
      <div class="games-grid" style="grid-template-columns:1fr 1fr 1fr">
        <div class="card"><div class="label">Level</div><div class="big">Lv ${p.level}</div><div class="muted">${p.xp} XP • ${p.progress_pct}% to next</div></div>
        <div class="card"><div class="label">Balance</div><div class="big">${fmtDL(p.balance)}</div></div>
        <div class="card"><div class="label">Role</div><div class="big" style="text-transform:uppercase">${role}</div></div>
      </div>
      <div class="hint" style="margin-top:8px">Visit Settings to enable Anonymous mode.</div>
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

// Settings
async function loadSettings(){
  qs('setMsg').textContent='';
  try{ const r = await j('/api/settings/get'); qs('anonToggle').checked = !!(r && r.is_anon); }catch(_){}
}
qs('anonToggle').addEventListener('change', async (e)=>{
  qs('setMsg').textContent='';
  try{
    const r = await j('/api/settings/set_anon', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ is_anon: !!e.target.checked }) });
    qs('setMsg').textContent = r && r.ok ? 'Saved.' : 'Updated.';
  }catch(err){ qs('setMsg').textContent = err.message; }
});

// Leaderboard
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
  return new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth()+1, 1, 0,0,0,0));
}
function renderLbCountdown(){
  const el = qs('lbCountdown');
  if(lbPeriod==='alltime'){ el.textContent='No expiry'; return; }
  const target = (lbPeriod==='daily') ? nextUtcMidnight() : endOfUtcMonth();
  const tick = ()=>{
    const diff = target - new Date();
    if(diff <= 0){ el.textContent = 'Resets soon'; return; }
    const s = Math.floor(diff/1000);
    const h = Math.floor(s/3600), m = Math.floor((s%3600)/60), ss = s%60;
    el.textContent = `Resets in ${h}h ${m}m ${ss}s (UTC)`;
  };
  tick(); if(window._lbTimer) clearInterval(window._lbTimer); window._lbTimer = setInterval(tick, 1000);
}
async function refreshLeaderboard(){
  try{
    const me = await j('/api/me').catch(()=>null);
    lbMe = me && me.id ? String(me.id) : null;
  }catch(_){}
  setLbButtons(); renderLbCountdown();
  lbWrap().textContent = 'Loading…';
  try{
    const r = await j(`/api/leaderboard?period=${lbPeriod}`);
    const rows = r.rows || [];
    if(!rows.length){ lbWrap().textContent='—'; return; }
    lbWrap().innerHTML = `
      <table>
        <thead><tr><th>#</th><th>User</th><th>Total Wagered</th></tr></thead>
        <tbody>
          ${rows.map((row, i)=>{
            const meRow = (lbMe && String(row.user_id)===lbMe);
            const cls = [ meRow ? 'me-row' : '', row.is_anon ? 'anon' : '' ].filter(Boolean).join(' ');
            const name = row.is_anon && !meRow ? 'Anonymous' : row.display_name;
            const amt = (row.is_anon && !meRow) ? '—' : fmtDL(row.total_wagered);
            return `<tr class="${cls}"><td>${i+1}</td><td class="name">${name}</td><td>${amt}</td></tr>`;
          }).join('')}
        </tbody>
      </table>
    `;
  }catch(e){ lbWrap().textContent = e.message || 'Error'; }
}
qs('lbSeg').addEventListener('click', (e)=>{
  const b = e.target.closest('button'); if(!b) return;
  lbPeriod = b.dataset.period; refreshLeaderboard();
});

// Modal helpers
const pm = qs('pm'); const openPm = ()=> pm.style.display='flex'; const closePm = ()=> pm.style.display='none';
qs('pmClose').onclick = closePm; pm.addEventListener('click',(e)=>{ if(e.target===pm) closePm(); });

// Crash JS
const pgCrash = qs('page-crash'); const crNow = qs('crNow'), crHint = qs('crHint'), lastBusts = qs('lastBusts'), crMsg = qs('crMsg');
const crCanvas = qs('crCanvas'), crBoom = qs('crBoom'); let haveActiveBet = false
# (continuation of the HTML <script>… you pasted)
function drawCurve(mult){
  const ctx = crCanvas.getContext('2d');
  const w = crCanvas.width = crCanvas.clientWidth * window.devicePixelRatio;
  const h = crCanvas.height = crCanvas.clientHeight * window.devicePixelRatio;
  ctx.clearRect(0,0,w,h);
  ctx.lineWidth = 3; ctx.strokeStyle = '#6aa6ff';
  ctx.beginPath();
  const maxX = Math.min(mult, 10.0);
  for(let i=0;i<=200;i++){
    const t = i/200 * maxX;
    const x = (i/200) * w;
    const y = h - (Math.log(1+t) / Math.log(1+maxX)) * h * 0.92 - h*0.04;
    if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  }
  ctx.stroke();
}
function boomAt(x,y){
  crBoom.style.setProperty('--x', (x*100)+'%');
  crBoom.style.setProperty('--y', (y*100)+'%');
  crBoom.classList.remove('bang'); void crBoom.offsetWidth; crBoom.classList.add('bang');
}
async function refreshCrash(){
  try{
    const s = await j('/api/crash/state');
    if(s.phase==='betting'){
      crHint.textContent = 'Betting… next round soon';
      crNow.textContent = '1.00×'; drawCurve(1.0);
    }else if(s.phase==='running'){
      const m = s.current_multiplier || 1.0;
      crHint.textContent = 'Running';
      crNow.textContent = m.toFixed(2) + '×';
      drawCurve(m);
    }else{
      crHint.textContent = 'Ended';
      crNow.textContent = (s.bust||0).toFixed(2)+'×';
      drawCurve(s.bust || 1.0); boomAt(0.92, 0.2);
    }
    if(s.last_busts && s.last_busts.length){
      lastBusts.innerHTML = s.last_busts.map(b=>`<span class="chip" style="margin-right:6px">${b.toFixed(2)}×</span>`).join('');
    }
    if(s.your_bet){
      haveActiveBet = !s.your_bet.resolved && (s.your_bet.cashed_out==null);
      qs('crCashout').style.display = haveActiveBet ? '' : 'none';
    }else qs('crCashout').style.display='none';

    try{
      const h = await j('/api/crash/history');
      const rows = h.rows||[];
      qs('crLast').innerHTML = rows.length
        ? `<table><thead><tr><th>When</th><th>Bet</th><th>Cashout</th><th>Bust</th><th>Win</th></tr></thead>
           <tbody>${rows.map(r=>`<tr><td>${new Date(r.created_at).toLocaleString()}</td><td>${fmtDL(r.bet)}</td><td>${r.cashout.toFixed(2)}×</td><td>${r.bust.toFixed(2)}×</td><td>${fmtDL(r.win)}</td></tr>`).join('')}</tbody></table>`
        : '—';
    }catch(_){}
  }catch(e){
    crHint.textContent = 'Error loading state';
  }
}
qs('crPlace').onclick = async ()=>{
  crMsg.textContent = '';
  try{
    const bet = parseFloat(qs('crBet').value||'0');
    const cashStr = qs('crCash').value.trim();
    const cashout = cashStr ? parseFloat(cashStr) : 2.0;
    const body = { bet: String(bet), cashout };
    await j('/api/crash/place', { method:'POST', headers:{'Content-Type': 'application/json'}, body: JSON.stringify(body) });
    crMsg.textContent = 'Bet placed. Good luck!';
    refreshCrash(); renderHeader();
  }catch(e){ crMsg.textContent = e.message; }
};
qs('crCashout').onclick = async ()=>{
  crMsg.textContent = '';
  try{
    const r = await j('/api/crash/cashout', { method:'POST' });
    crMsg.textContent = `Cashed out at ${r.multiplier.toFixed(2)}× for ${fmtDL(r.win)}.`; haveActiveBet = false; renderHeader();
  }catch(e){ crMsg.textContent = e.message; }
};
setInterval(()=>{ if(pgCrash.style.display !== 'none') refreshCrash(); }, 1000);

// Mines
const mGrid = qs('mGrid'); let mState = null;
function renderGridTiles(disabled=false){
  if(!mGrid) return;
  mGrid.innerHTML = '';
  for(let i=0;i<25;i++){
    const el = document.createElement('div');
    el.className = 'tile'; el.dataset.index = i;
    el.onclick = async ()=>{
      if(disabled) return;
      try{
        const r = await j(`/api/mines/pick?index=${i}`, { method: 'POST' });
        if(r.status === 'lost'){
          el.classList.add('mine','explode','revealed');
          setTimeout(()=>{ revealBoard(r.board); }, 150);
          qs('mStatus').textContent = 'Status: Lost';
          qs('mCash').style.display='none';
          renderHeader(); renderMines();
        }else{
          el.classList.add('safe','revealed','pop');
          qs('mPicks').textContent = 'Picks: ' + (r.picks.toString(2).match(/1/g)||[]).length;
          qs('mMult').textContent = 'Multiplier: ' + r.multiplier.toFixed(4) + '×';
          qs('mPotential').textContent = 'Potential: ' + fmtDL(r.potential_win);
          qs('mStatus').textContent = 'Status: Active';
          qs('mCash').style.display='';
        }
      }catch(e){ qs('mMsg').textContent = e.message; }
    };
    mGrid.appendChild(el);
  }
}
function markPicked(mask){
  for(let i=0;i<25;i++){
    const el = mGrid.children[i];
    if((mask>>i) & 1){ el.classList.add('safe','revealed'); el.textContent = '✓'; }
  }
}
function revealBoard(board){
  for(let i=0;i<25;i++){
    const el = mGrid.children[i];
    if(board[i] === '1'){ el.classList.add('mine','revealed','explode'); el.textContent = '💣'; }
    else if(!el.classList.contains('revealed')){ el.classList.add('safe','revealed'); el.textContent = '✓'; }
  }
}
async function renderMines(){
  qs('mMsg').textContent = '';
  try{
    const s = await j('/api/mines/state');
    if(s){
      mState = s;
      qs('mHash').textContent = 'Commit: ' + s.hash.slice(0,12) + '…';
      qs('mStatus').textContent = 'Status: ' + s.status;
      qs('mPicks').textContent = 'Picks: ' + (s.picks.toString(2).match(/1/g)||[]).length;
      qs('mBombs').textContent = 'Mines: ' + s.mines;
      qs('mCash').style.display = (s.status==='active') ? '' : 'none';
      renderGridTiles(s.status!=='active'); markPicked(s.picks);
    }else{
      mState = null;
      qs('mHash').textContent = 'Commit: —';
      qs('mStatus').textContent = 'Status: —';
      qs('mPicks').textContent = 'Picks: 0';
      qs('mBombs').textContent = 'Mines: ' + (parseInt(qs('mMines').value||'3',10));
      qs('mCash').style.display='none';
      renderGridTiles(false);
    }
  }catch(_){}
  try{
    const h = await j('/api/mines/history');
    const rows = h.rows || [];
    qs('mHist').innerHTML = rows.length
      ? `<table><thead><tr><th>ID</th><th>When</th><th>Mines</th><th>Bet</th><th>Win</th><th>Status</th></tr></thead>
         <tbody>${rows.map(r=>`<tr><td>${r.id}</td><td>${new Date(r.started_at).toLocaleString()}</td><td>${r.mines}</td><td>${fmtDL(r.bet)}</td><td>${fmtDL(r.win)}</td><td>${r.status}</td></tr>`).join('')}</tbody></table>`
      : '—';
  }catch(_){}
}
qs('mStart').onclick = async ()=>{
  const bet = qs('mBet').value.trim();
  const mines = parseInt(qs('mMines').value||'3',10);
  qs('mMsg').textContent = '';
  try{
    const r = await j('/api/mines/start',{ method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ bet, mines }) });
    qs('mHash').textContent = 'Commit: ' + r.hash.slice(0,12) + '…';
    qs('mStatus').textContent = 'Status: active';
    qs('mPicks').textContent = 'Picks: 0';
    qs('mBombs').textContent = 'Mines: ' + mines;
    qs('mCash').style.display = '';
    renderGridTiles(false); renderHeader();
  }catch(e){ qs('mMsg').textContent = e.message; }
};
qs('mCash').onclick = async ()=>{
  qs('mMsg').textContent = '';
  try{
    const r = await j('/api/mines/cashout', { method:'POST' });
    qs('mStatus').textContent = 'Status: cashed';
    qs('mMsg').textContent = 'Cashed out for ' + fmtDL(r.win);
    revealBoard(r.board);
    qs('mCash').style.display='none';
    renderHeader(); renderMines();
  }catch(e){ qs('mMsg').textContent = e.message; }
};

// Chat
const chatDrawer = qs('chatDrawer'), fabChat = qs('fabChat'), chatBody = qs('chatBody'), chatDisabled = qs('chatDisabled');
const chatText = qs('chatText'); const chatSend = qs('chatSend'); const chatNote = qs('chatNote');
const chatClose = qs('chatClose');
fabChat.onclick = ()=>{ chatDrawer.classList.add('open'); };
chatClose.onclick = ()=>{ chatDrawer.classList.remove('open'); };
let chatSince = 0; let chatTimer = null; let meCache = null;
function badge(role){ role = (role||'member').toLowerCase(); return `<span class="badge ${role}">${role.toUpperCase()}</span>`; }
function levelChip(lv){ return `<span class="level">Lv ${lv||1}</span>`; }
function appendMessages(list){
  for(const m of (list||[])){
    const row = document.createElement('div'); row.className = 'msg'; row.dataset.id = m.id;
    row.innerHTML = `
      <div class="msghead">
        <span class="user-link" data-uid="${m.user_id}">${m.username}</span>
        ${badge(m.role)} ${levelChip(m.level)}
        <span class="time">${new Date(m.created_at).toLocaleTimeString()}</span>
      </div>
      <div class="text">${escapeHtml(m.text)}</div>
    `;
    chatBody.appendChild(row); chatBody.scrollTop = chatBody.scrollHeight;
    chatSince = Math.max(chatSince, m.id);
  }
}
function escapeHtml(s){ return (s||'').replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
async function chatPoll(){ try{ const r = await j(`/api/chat/fetch?since=${chatSince}`); appendMessages(r.rows || r); }catch(_){} }
async function renderChatHeader(){ try{ const prof = await j('/api/profile'); meCache = prof; chatNote.textContent = `Lv ${prof.level} — ${prof.role.toUpperCase()}`; }catch(_){} }
chatTimer = setInterval(()=>{ if(chatDrawer.classList.contains('open')) chatPoll(); }, 1500);
chatBody.addEventListener('click',(e)=>{ const el = e.target.closest('.user-link'); if(!el) return; openProfileModal(el.dataset.uid); });
async function openProfileModal(uid){
  try{
    const p = await j(`/api/public_profile?user_id=${encodeURIComponent(uid)}`);
    qs('pmTitle').textContent = 'Profile — ' + p.name;
    qs('pmBody').innerHTML = `
      <div class="games-grid" style="grid-template-columns:1fr 1fr">
        <div class="card"><div class="label">Role</div><div>${(p.role||'member').toUpperCase()}</div></div>
        <div class="card"><div class="label">Level / XP</div><div>Lv ${p.level} — ${p.xp} XP</div></div>
        <div class="card"><div class="label">Balance</div><div>${fmtDL(p.balance)}</div></div>
        <div class="card"><div class="label">Stats</div><div>Crash: ${p.crash_games} • Mines: ${p.mines_games}</div></div>
      </div>
      <div class="muted" style="margin-top:8px">Joined: ${new Date(p.created_at).toLocaleString()}</div>
    `;
    openPm();
  }catch(_){}
}
chatSend.onclick = async ()=>{
  try{
    const txt = (chatText.value||'').trim(); if(!txt) return;
    await j('/api/chat/send', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ text: txt }) });
    chatText.value = ''; chatPoll();
  }catch(e){
    chatDisabled.style.display = ''; chatDisabled.textContent = e.message || 'Cannot send';
    setTimeout(()=>{ chatDisabled.style.display='none'; }, 3500);
  }
};

// Navigation
qs('openCrash').onclick=()=>{ showOnly('page-crash'); refreshCrash(); };
qs('openMines').onclick=()=>{ showOnly('page-mines'); renderMines(); };
qs('backToGames').onclick=()=> showOnly('page-games');
qs('backToGames2').onclick=()=> showOnly('page-games');

// Boot
renderHeader(); renderChatHeader();
</script>
</body></html>
"""
HTML_TEMPLATE = HTML_TEMPLATE.replace("__OWNER_ID__", str(OWNER_ID))

# ---------- OAuth & Session Routes ----------
def _discord_avatar_url(user: dict) -> str:
    uid = user.get("id")
    avatar = user.get("avatar")
    if uid and avatar:
        return f"https://cdn.discordapp.com/avatars/{uid}/{avatar}.png"
    # default avatar (computed from discriminator hash if absent)
    try:
        i = int(uid) % 5 if uid else 0
    except Exception:
        i = 0
    return f"https://cdn.discordapp.com/embed/avatars/{i}.png"

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_TEMPLATE

@app.get("/login")
async def login():
    if not CLIENT_ID or not OAUTH_REDIRECT:
        raise HTTPException(500, "OAuth not configured")
    params = {
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": OAUTH_REDIRECT,
        "scope": "identify",
        "prompt": "none"
    }
    return RedirectResponse(f"{DISCORD_API}/oauth2/authorize?{urlencode(params)}")

@app.get("/oauth/callback")
async def oauth_callback(request: Request, code: Optional[str] = None, error: Optional[str] = None):
    if error: raise HTTPException(400, f"OAuth error: {error}")
    if not code: raise HTTPException(400, "Missing code")
    async with httpx.AsyncClient(timeout=15) as client:
        token_r = await client.post(f"{DISCORD_API}/oauth2/token",
            data={
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": OAUTH_REDIRECT,
            },
            headers={"Content-Type":"application/x-www-form-urlencoded"})
        token_r.raise_for_status()
        tok = token_r.json()
        me_r = await client.get(f"{DISCORD_API}/users/@me", headers={"Authorization": f"Bearer {tok['access_token']}"})
        me_r.raise_for_status()
        user = me_r.json()
    uid = str(user["id"])
    ensure_profile_row(uid)
    resp = RedirectResponse("/")
    _set_session(resp, {"id": uid, "username": user.get("username"), "avatar_url": _discord_avatar_url(user)})
    return resp

@app.get("/logout")
async def logout():
    resp = RedirectResponse("/")
    _clear_session(resp)
    return resp

# ---------- API route helpers ----------
def _get_me(request: Request) -> dict:
    sess = _require_session(request)
    return {"id": str(sess["id"]), "username": sess.get("username","user"), "avatar_url": sess.get("avatar_url","")}

def _id_from_identifier(identifier: str) -> str:
    identifier = (identifier or "").strip()
    m = re.match(r"<@!?(\d+)>", identifier)
    if m: return m.group(1)
    m = re.match(r"^(\d{6,30})$", identifier)
    if m: return m.group(1)
    raise HTTPException(400, "Invalid identifier")

# ---------- API: Me / Balance / Profile ----------
@app.get("/api/me")
async def api_me(request: Request):
    return _get_me(request)

@app.get("/api/balance")
async def api_balance(request: Request):
    uid = _require_session(request)["id"]
    return {"balance": float(get_balance(uid))}

@app.get("/api/profile")
async def api_profile(request: Request):
    uid = _require_session(request)["id"]
    return profile_info(uid)

@app.get("/api/public_profile")
async def api_public_profile(user_id: str = Query(...)):
    p = public_profile(user_id)
    if not p: raise HTTPException(404, "Not found")
    return p

# ---------- API: Settings ----------
class AnonIn(BaseModel):
    is_anon: bool

@app.get("/api/settings/get")
async def api_settings_get(request: Request):
    uid = _require_session(request)["id"]
    cur = profile_info(uid)  # includes is_anon
    return {"is_anon": bool(cur.get("is_anon", False))}

@app.post("/api/settings/set_anon")
async def api_settings_set_anon(request: Request, body: AnonIn):
    uid = _require_session(request)["id"]
    return set_profile_is_anon(uid, body.is_anon)

# ---------- API: Promo ----------
class PromoIn(BaseModel):
    code: str

@app.post("/api/promo/redeem")
async def api_promo_redeem(request: Request, body: PromoIn):
    uid = _require_session(request)["id"]
    try:
        new_bal = redeem_promo(uid, body.code)
        return {"ok": True, "new_balance": float(new_bal)}
    except PromoInvalid as e:
        raise HTTPException(400, str(e))
    except PromoExpired as e:
        raise HTTPException(400, str(e))
    except PromoExhausted as e:
        raise HTTPException(400, str(e))
    except PromoAlreadyRedeemed as e:
        raise HTTPException(400, str(e))

@app.get("/api/promo/my")
async def api_promo_my(request: Request):
    uid = _require_session(request)["id"]
    @with_conn
    def _rows(cur, user_id):
        cur.execute("SELECT code, redeemed_at FROM promo_redemptions WHERE user_id=%s ORDER BY redeemed_at DESC LIMIT 50", (user_id,))
        return [{"code": r[0], "redeemed_at": r[1].isoformat()} for r in cur.fetchall()]
    return {"rows": _rows(uid)}

# ---------- API: Referral (stub) ----------
@app.get("/api/referral/state")
async def api_referral_state(request: Request):
    uid = _require_session(request)["id"]
    name = get_profile_name(uid)
    return {"name": name}

# ---------- API: Crash ----------
class PlaceBetIn(BaseModel):
    bet: str
    cashout: Optional[float] = None

@app.get("/api/crash/state")
async def api_crash_state(request: Request):
    uid = _require_session(request)["id"]
    # progress engine
    rd = load_round()
    now = now_utc()
    if not rd:
        ensure_betting_round()  # creates a betting round
        rd = load_round()
    # transition if needed
    if rd["status"] == "betting" and now >= rd["betting_ends_at"]:
        begin_running(rd["id"])
        rd = load_round()
    if rd["status"] == "running" and now >= rd["expected_end_at"]:
        finish_round(rd["id"])
        create_next_betting()
        rd = load_round()

    phase = "betting" if rd["status"]=="betting" else ("running" if rd["status"]=="running" else "ended")
    current = None
    if phase=="running":
        current = current_multiplier(rd["started_at"], rd["expected_end_at"], rd["bust"], now)
    yb = your_bet(rd["id"], uid) if rd["id"] else None
    return {
        "phase": phase,
        "bust": rd.get("bust"),
        "current_multiplier": current,
        "last_busts": last_busts(15),
        "your_bet": yb
    }

@app.post("/api/crash/place")
async def api_crash_place(request: Request, body: PlaceBetIn):
    uid = _require_session(request)["id"]
    bet = q2(D(body.bet))
    cash = float(body.cashout or 2.0)
    if cash < 1.01: raise HTTPException(400, "Cashout must be ≥ 1.01×")
    try:
        r = place_bet(uid, bet, cash)
        return {"ok": True, "round_id": r["round_id"]}
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.post("/api/crash/cashout")
async def api_crash_cashout(request: Request):
    uid = _require_session(request)["id"]
    try:
        r = cashout_now(uid)
        return r
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.get("/api/crash/history")
async def api_crash_history(request: Request):
    uid = _require_session(request)["id"]
    return {"rows": your_history(uid)}

# ---------- API: Mines ----------
class MinesStartIn(BaseModel):
    bet: str
    mines: int

@app.get("/api/mines/state")
async def api_mines_state(request: Request):
    uid = _require_session(request)["id"]
    s = mines_state(uid)
    return s or {}

@app.post("/api/mines/start")
async def api_mines_start(request: Request, body: MinesStartIn):
    uid = _require_session(request)["id"]
    try:
        r = mines_start(uid, q2(D(body.bet)), int(body.mines))
        return r
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.post("/api/mines/pick")
async def api_mines_pick(request: Request, index: int = Query(..., ge=0, le=24)):
    uid = _require_session(request)["id"]
    try:
        return mines_pick(uid, index)
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.post("/api/mines/cashout")
async def api_mines_cashout(request: Request):
    uid = _require_session(request)["id"]
    try:
        return mines_cashout(uid)
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.get("/api/mines/history")
async def api_mines_history(request: Request, limit: int = Query(15, ge=1, le=100)):
    uid = _require_session(request)["id"]
    return {"rows": mines_history(uid, limit)}

# ---------- API: Chat ----------
class ChatIn(BaseModel):
    text: str

@app.get("/api/chat/fetch")
async def api_chat_fetch(request: Request, since: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)):
    maybe = request.cookies.get("session")
    uid = None
    if maybe:
        try:
            uid = SER.loads(maybe).get("id")
        except BadSignature:
            uid = None
    rows = chat_fetch(since, limit, uid)
    return {"rows": rows}

@app.post("/api/chat/send")
async def api_chat_send(request: Request, body: ChatIn):
    sess = _require_session(request)
    uid = sess["id"]; username = sess.get("username","user")
    txt = (body.text or "").strip()
    # Quick command: .tip <uid> <amount>
    if txt.startswith(".tip "):
        parts = txt.split()
        if len(parts) >= 3:
            target = parts[1]; amt = parts[2]
            try:
                tip_transfer(uid, _id_from_identifier(target), q2(D(amt)))
                return {"ok": True, "tip": True}
            except Exception as e:
                raise HTTPException(400, f"Tip failed: {e}")
    try:
        r = chat_insert(uid, username, txt)
        return r
    except PermissionError as e:
        raise HTTPException(403, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.post("/api/chat/delete")
async def api_chat_delete(request: Request, id: int = Query(..., ge=1)):
    uid = _require_session(request)["id"]
    role = get_role(uid)
    if role not in ("admin","owner"): raise HTTPException(403, "Admin only")
    return chat_delete(id)

# ---------- API: Admin ----------
class AdjustIn(BaseModel):
    identifier: str
    amount: str
    reason: Optional[str] = None

class RoleIn(BaseModel):
    identifier: str
    role: str

class TimeoutIn(BaseModel):
    identifier: str
    seconds: int
    reason: Optional[str] = None

class PromoCreateIn(BaseModel):
    code: Optional[str] = None
    amount: str
    max_uses: int = 1
    expires_at: Optional[str] = None

def _assert_owner(request: Request) -> str:
    uid = _require_session(request)["id"]
    if str(uid) != str(OWNER_ID): raise HTTPException(403, "Owner only")
    return uid

@app.post("/api/admin/adjust")
async def api_admin_adjust(request: Request, body: AdjustIn):
    actor = _assert_owner(request)
    target = _id_from_identifier(body.identifier)
    try:
        new_bal = adjust_balance(actor, target, q2(D(body.amount)), body.reason)
        return {"ok": True, "new_balance": float(new_bal)}
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/api/admin/role")
async def api_admin_role(request: Request, body: RoleIn):
    _assert_owner(request)
    target = _id_from_identifier(body.identifier)
    return set_role(target, body.role.lower())

@app.post("/api/admin/timeout_site")
async def api_admin_timeout_site(request: Request, body: TimeoutIn):
    actor = _assert_owner(request)
    target = _id_from_identifier(body.identifier)
    return chat_timeout_set(actor, target, max(1, int(body.seconds)), body.reason)

@app.post("/api/admin/timeout_both")
async def api_admin_timeout_both(request: Request, body: TimeoutIn):
    # For now identical to site timeout; Discord moderation not implemented here.
    actor = _assert_owner(request)
    target = _id_from_identifier(body.identifier)
    return chat_timeout_set(actor, target, max(1, int(body.seconds)), (body.reason or "") + " (site+discord requested)")

@app.post("/api/admin/promo/create")
async def api_admin_promo_create(request: Request, body: PromoCreateIn):
    actor = _assert_owner(request)
    r = create_promo(actor, body.code, body.amount, max_uses=int(body.max_uses or 1), expires_at=body.expires_at)
    return r

# ---------- API: Leaderboard ----------
@app.get("/api/leaderboard")
async def api_leaderboard(period: str = Query("daily"), limit: int = Query(50, ge=1, le=200)):
    rows = leaderboard_rows(period, limit)
    return {"rows": rows}

# ---------- Run ----------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT)
