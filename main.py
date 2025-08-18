import os, json, asyncio, re, random, string, math, secrets, datetime, hashlib
from urllib.parse import urlencode
from typing import Optional, Tuple, Dict, List
from decimal import Decimal, ROUND_DOWN, getcontext

import httpx
import psycopg
import discord
from discord.ext import commands
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from itsdangerous import URLSafeSerializer, BadSignature
import uvicorn
from pydantic import BaseModel

# ---------- Config ----------
getcontext().prec = 28  # high precision for Decimal math

PREFIX = "."
BOT_TOKEN = os.getenv("DISCORD_TOKEN")
CLIENT_ID = os.getenv("DISCORD_CLIENT_ID")
CLIENT_SECRET = os.getenv("DISCORD_CLIENT_SECRET") or os.getenv("CLIENT_SECRET")
OAUTH_REDIRECT = os.getenv("OAUTH_REDIRECT")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")
PORT = int(os.getenv("PORT", "8080"))
DISCORD_API = "https://discord.com/api"
OWNER_ID = 1128658280546320426
GUILD_ID = int(os.getenv("GUILD_ID", "0"))
DATABASE_URL = os.getenv("DATABASE_URL")

GEM = "💎"
MIN_BET = Decimal("1.00")        # 1.00 DL minimum
MAX_BET = Decimal("1000000.00")
BETTING_SECONDS = 10             # Crash betting window

# House edges (env-tunable)
HOUSE_EDGE_CRASH = Decimal(os.getenv("HOUSE_EDGE_CRASH", "0.06"))  # 6% default for Crash
HOUSE_EDGE_MINES = Decimal(os.getenv("HOUSE_EDGE_MINES", "0.03"))  # 3% per safe reveal for Mines

TWO = Decimal("0.01")

def D(x) -> Decimal:
    if isinstance(x, Decimal): return x
    return Decimal(str(x))

def q2(x: Decimal) -> Decimal:
    return D(x).quantize(TWO, rounding=ROUND_DOWN)

# ---------- Time helpers ----------
UTC = datetime.timezone.utc
def now_utc() -> datetime.datetime: return datetime.datetime.now(UTC)
def iso(dt: Optional[str|datetime.datetime]) -> Optional[str]:
    if dt is None: return None
    if isinstance(dt, str): return dt
    return dt.astimezone(UTC).isoformat()

# ---------- FastAPI ----------
app = FastAPI()

# Ensure /static exists so StaticFiles won't crash
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

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

    # profiles / levels / roles
    cur.execute("""
        CREATE TABLE IF NOT EXISTS profiles (
            user_id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            name_lower TEXT NOT NULL UNIQUE,
            xp INTEGER NOT NULL DEFAULT 0,
            role TEXT NOT NULL DEFAULT 'member', -- member|admin|owner
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # add role if missing
    try:
        cur.execute("ALTER TABLE profiles ADD COLUMN role TEXT NOT NULL DEFAULT 'member'")
    except Exception:
        pass

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
            status TEXT NOT NULL, -- 'betting' | 'running' | 'ended'
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

    # global chat
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id BIGSERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            username TEXT NOT NULL,
            text TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # chat timeouts
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_timeouts (
            user_id TEXT PRIMARY KEY,
            until TIMESTAMPTZ NOT NULL,
            reason TEXT,
            created_by TEXT,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # MINES
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
            status TEXT NOT NULL DEFAULT 'active',  -- active|lost|cashed
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

NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{3,20}$")

@with_conn
def ensure_profile_row(cur, user_id: str):
    # owner gets role=owner
    role = 'owner' if str(user_id) == str(OWNER_ID) else 'member'
    cur.execute("""
        INSERT INTO profiles(user_id, display_name, name_lower, role)
        VALUES (%s,%s,%s,%s)
        ON CONFLICT (user_id) DO NOTHING
    """, (user_id, f"user_{user_id[-4:]}", f"user_{user_id[-4:]}", role))

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
def profile_info(cur, user_id: str):
    ensure_profile_row(user_id)
    cur.execute("SELECT xp, role FROM profiles WHERE user_id=%s", (user_id,))
    xp, role = cur.fetchone()
    level = 1 + int(xp) // 100
    base = (level - 1) * 100; need = level * 100 - base
    progress = int(xp) - base; pct = 0 if need==0 else int(progress*100/need)
    bal = get_balance(user_id)
    return {"xp": int(xp), "level": level, "progress": progress, "next_needed": need, "progress_pct": pct, "balance": float(bal), "role": role}

@with_conn
def public_profile(cur, user_id: str):
    ensure_profile_row(user_id)
    cur.execute("SELECT display_name, xp, role, created_at FROM profiles WHERE user_id=%s", (user_id,))
    r = cur.fetchone()
    if not r: return None
    display_name, xp, role, created_at = r
    level = 1 + int(xp)//100
    cur.execute("SELECT COUNT(*) FROM crash_games WHERE user_id=%s", (user_id,)); crash_count = int(cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM mines_games WHERE user_id=%s AND status<>'active'", (user_id,)); mines_count = int(cur.fetchone()[0])
    bal = get_balance(user_id)
    return {
        "id": str(user_id), "name": display_name, "role": role,
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

# ---- Crash math & DB ----
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
    if not cur.fetchone()[0]:
        raise ValueError("Betting just closed")

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
    resolve_round_end(round_id, bust)
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
def chat_send(cur, user_id: str, username: str, text: str):
    text = (text or "").strip()
    if not text: raise ValueError("Message is empty")
    if len(text) > 300: raise ValueError("Message is too long (max 300)")
    ensure_profile_row(user_id)
    # level gate
    cur.execute("SELECT xp FROM profiles WHERE user_id=%s", (user_id,))
    xp = int(cur.fetchone()[0])
    lvl = 1 + xp // 100
    if lvl < 5: raise PermissionError("You must be level 5 to chat")
    # timeout gate
    cur.execute("SELECT until FROM chat_timeouts WHERE user_id=%s", (user_id,))
    r = cur.fetchone()
    if r and r[0] > now_utc():
        raise PermissionError("You are timed out")

    cur.execute("INSERT INTO chat_messages(user_id, username, text) VALUES (%s,%s,%s) RETURNING id, created_at",
                (user_id, username, text))
    row = cur.fetchone()
    return {"id": int(row[0]), "created_at": str(row[1])}

@with_conn
def chat_fetch(cur, since_id: int, limit: int = 50):
    if since_id <= 0:
        cur.execute("""SELECT id, user_id, username, text, created_at
                       FROM chat_messages ORDER BY id DESC LIMIT %s""", (limit,))
        rows = list(reversed(cur.fetchall()))
    else:
        cur.execute("""SELECT id, user_id, username, text, created_at
                       FROM chat_messages WHERE id > %s ORDER BY id ASC LIMIT %s""", (since_id, limit))
        rows = cur.fetchall()
    uids = list({r[1] for r in rows})
    levels: Dict[str, int] = {}
    roles: Dict[str, str] = {}
    if uids:
        cur.execute("SELECT user_id, xp, role FROM profiles WHERE user_id = ANY(%s)", (uids,))
        for uid, xp, role in cur.fetchall():
            levels[str(uid)] = 1 + int(xp) // 100
            roles[str(uid)] = role or "member"
    out = []
    for mid, uid, uname, txt, ts in rows:
        uid = str(uid)
        lvl = levels.get(uid, 1)
        role = roles.get(uid, "member")
        out.append({"id": int(mid), "user_id": uid, "username": uname, "level": int(lvl), "role": role, "text": txt, "created_at": str(ts)})
    return out

# ---------- MINES helpers ----------
def mines_random_board(mines: int) -> str:
    idxs = list(range(25))
    random.shuffle(idxs)
    mines_set = set(idxs[:mines])
    return ''.join('1' if i in mines_set else '0' for i in range(25))

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def picks_count_from_bitmask(mask: int) -> int:
    return mask.bit_count()

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

# ---------- Discord bot ----------
def fmt_dl(n) -> str:
    v = q2(D(n))
    return f"{GEM} {v:,.2f} DL"

def embed(title: str, desc: Optional[str] = None, color: int = 0x00C2FF) -> discord.Embed:
    return discord.Embed(title=title, description=desc or "", color=color)
def avatar_url_from(id_str: str, avatar_hash: Optional[str]) -> str:
    if avatar_hash: return f"https://cdn.discordapp.com/avatars/{id_str}/{avatar_hash}.png?size=64"
    idx = int(id_str) % 6; return f"https://cdn.discordapp.com/embed/avatars/{idx}.png?size=64"

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix=PREFIX, intents=intents, help_command=None)

@bot.event
async def on_ready():
    g = bot.get_guild(GUILD_ID) if GUILD_ID else None
    print(f"Logged in as {bot.user} (id={bot.user.id}). Guild set: {bool(g)}")
    if g: print(f"Guild: {g.name} ({g.id}) — cached members: {len(g.members)}")

@bot.command(name="help")
async def help_command(ctx: commands.Context):
    is_owner = (ctx.author.id == OWNER_ID)
    e = embed(title="💎 DL Bank — Help", desc=f"Prefix: `{PREFIX}`", color=0x60A5FA)
    e.add_field(name="General",
                value=(f"**{PREFIX}help** — Show this help\n"
                       f"**{PREFIX}bal** — Show **your** balance\n"
                       f"**{PREFIX}bal @User** — Show **someone else’s** balance\n"
                       f"**{PREFIX}tip <amount> <@User/ID>** — Send DL to another user"),
                inline=False)
    owner_line = (f"**{PREFIX}addbal @User <amount>** — Adjust balance\n"
                  f"**Mods**: website/discord timeouts via Owner Panel")
    if is_owner: owner_line += " ✅"
    e.add_field(name="Admin", value=owner_line, inline=False)
    e.set_footer(text="Currency: Growtopia Diamond Locks (DL)")
    await ctx.reply(embed=e, mention_author=False)

@with_conn
def get_user_balance(cur, uid: str) -> Decimal:
    cur.execute("SELECT balance FROM balances WHERE user_id=%s", (uid,))
    r = cur.fetchone(); return q2(r[0]) if r else Decimal("0.00")

@bot.command(name="bal")
async def bal(ctx: commands.Context, user: discord.User | None = None):
    target = user or ctx.author
    bal_value = get_user_balance(str(target.id))
    e = embed(title="Balance", desc=f"{target.mention}\n**{fmt_dl(bal_value)}**", color=0x34D399)
    e.set_footer(text="Currency: Growtopia Diamond Locks (DL)")
    await ctx.reply(embed=e, mention_author=False)

@bot.command(name="addbal")
async def addbal(ctx: commands.Context, user: discord.User | None = None, amount: str | None = None):
    if ctx.author.id != OWNER_ID:
        return await ctx.reply(embed=embed("Not allowed","Only the owner can adjust balances.",0xEF4444), mention_author=False)
    if user is None or amount is None:
        return await ctx.reply(embed=embed("Usage", f"`{PREFIX}addbal @User <amount>` (e.g. 1.24)", 0xF59E0B), mention_author=False)
    try:
        delta = q2(D(amount))
    except Exception:
        return await ctx.reply(embed=embed("Invalid amount","Use a number like 1 or 1.24",0xEF4444), mention_author=False)
    if delta == 0:
        return await ctx.reply(embed=embed("Invalid amount","Amount cannot be zero.",0xEF4444), mention_author=False)
    new_balance = adjust_balance(str(ctx.author.id), str(user.id), delta, reason="bot addbal")
    sign = "+" if delta > 0 else ""
    e = embed("Balance Updated", f"**Target:** {user.mention}\n**Change:** `{sign}{delta}` → {fmt_dl(new_balance)}", 0x60A5FA)
    e.set_footer(text="Currency: Growtopia Diamond Locks (DL)")
    await ctx.reply(embed=e, mention_author=False)

def parse_target(val: str) -> Optional[str]:
    if not val: return None
    m = re.match(r"^<@!?(\d+)>$", val)
    if m: return m.group(1)
    if val.isdigit(): return val
    return None

@bot.command(name="tip")
async def tip(ctx: commands.Context, amount: str | None = None, target: str | None = None):
    if amount is None or target is None:
        return await ctx.reply(embed=embed("Usage", f"`{PREFIX}tip <amount> <@User/ID>`", 0xF59E0B), mention_author=False)
    try:
        amt = q2(D(amount))
    except Exception:
        return await ctx.reply(embed=embed("Invalid amount","Use 1 or 1.24 etc.",0xEF4444), mention_author=False)
    if amt <= 0: return await ctx.reply(embed=embed("Invalid amount","Must be > 0.",0xEF4444), mention_author=False)
    tid = parse_target(target)
    if not tid: return await ctx.reply(embed=embed("Invalid target","Mention a user or provide their ID.",0xEF4444), mention_author=False)
    if str(ctx.author.id) == tid:
        return await ctx.reply(embed=embed("Nope","You can’t tip yourself.",0xEF4444), mention_author=False)

    try:
        with psycopg.connect(DATABASE_URL) as con, con.cursor() as cur:
            cur.execute("INSERT INTO balances(user_id,balance) VALUES (%s,0) ON CONFLICT(user_id) DO NOTHING", (str(ctx.author.id),))
            cur.execute("INSERT INTO balances(user_id,balance) VALUES (%s,0) ON CONFLICT(user_id) DO NOTHING", (tid,))
            cur.execute("SELECT balance FROM balances WHERE user_id=%s FOR UPDATE", (str(ctx.author.id),))
            sbal = D(cur.fetchone()[0])
            if sbal < amt: raise ValueError("Insufficient balance")
            cur.execute("UPDATE balances SET balance=balance-%s WHERE user_id=%s", (amt, str(ctx.author.id)))
            cur.execute("UPDATE balances SET balance=balance+%s WHERE user_id=%s", (amt, tid))
            cur.execute("INSERT INTO balance_log(actor_id,target_id,amount,reason) VALUES (%s,%s,%s,%s)",
                        (str(ctx.author.id), tid, -amt, "tip"))
            cur.execute("INSERT INTO balance_log(actor_id,target_id,amount,reason) VALUES (%s,%s,%s,%s)",
                        (str(ctx.author.id), tid, amt, "tip"))
            con.commit()
        await ctx.reply(embed=embed("Tip sent", f"{ctx.author.mention} ➜ <@{tid}> • {fmt_dl(amt)}", 0x34D399), mention_author=False)
    except Exception as e:
        await ctx.reply(embed=embed("Failed", str(e), 0xEF4444), mention_author=False)

# ---------- Web (FastAPI + Frontend) ----------
signer = URLSafeSerializer(SECRET_KEY, salt="session")

def set_session(resp: RedirectResponse, payload: dict):
    resp.set_cookie("session", signer.dumps(payload), httponly=True, samesite="lax", max_age=7*24*3600)
def read_session(request: Request) -> Optional[dict]:
    raw = request.cookies.get("session")
    if not raw: return None
    try: return signer.loads(raw)
    except BadSignature: return None

# ---------- Frontend HTML (with improved chat badges + profile modal + sliding FAB) ----------
HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover" />
  <title>💎 DL Bank</title>
  <style>
    :root{
      --bg:#0a0f1e; --bg2:#0c1428; --card:#111a31; --muted:#9eb3da; --text:#ecf2ff;
      --accent:#6aa6ff; --accent2:#22c1dc; --ok:#34d399; --warn:#f59e0b; --err:#ef4444; --border:#1f2b47;
      --chatW: 300px;
    }
    *{box-sizing:border-box}
    html,body{height:100%}
    body{
      margin:0; color:var(--text); background:radial-gradient(1400px 600px at 20% -10%, #11204d 0%, transparent 60%), linear-gradient(180deg,#0a0f1e,#0a0f1e 60%, #0b1124);
      font-family:Inter, system-ui, Segoe UI, Roboto, Arial, Helvetica, sans-serif;
      -webkit-font-smoothing:antialiased; -moz-osx-font-smoothing:grayscale;
    }
    a{color:inherit; text-decoration:none}
    .container{max-width:1100px; margin:0 auto; padding:16px}

    /* header */
    .header{position:sticky; top:0; z-index:30; backdrop-filter: blur(8px); background:rgba(10,15,30,.7); border-bottom:1px solid var(--border)}
    .header-inner{display:flex; align-items:center; justify-content:space-between; gap:10px; padding:10px 12px}
    .left{display:flex; align-items:center; gap:14px; flex:1; min-width:0}
    .brand{display:flex; align-items:center; gap:10px; font-weight:800; letter-spacing:.2px; white-space:nowrap}
    .brand .logo{width:28px;height:28px;border-radius:8px; background:linear-gradient(135deg,var(--accent),var(--accent2))}
    /* segmented tabs pulled left */
    .tabs{
      display:flex; gap:4px; align-items:center; padding:4px; border-radius:14px;
      background:linear-gradient(180deg,#0f1a33,#0b1326); border:1px solid var(--border);
      overflow:auto; -webkit-overflow-scrolling:touch; max-width:100%;
    }
    .tab{
      padding:8px 12px; border-radius:10px; cursor:pointer; font-weight:700; white-space:nowrap;
      color:#d8e6ff; opacity:.85; transition:all .15s ease; display:flex; align-items:center; gap:8px;
    }
    .tab:hover{opacity:1; transform:translateY(-1px)}
    .tab.active{
      background:linear-gradient(135deg,#3b82f6,#22c1dc); color:#051326; box-shadow:0 6px 16px rgba(59,130,246,.25); opacity:1;
    }
    .right{display:flex; gap:8px; align-items:center; margin-left:12px}
    .chip{background:#0c1631; border:1px solid var(--border); color:#dce7ff; padding:6px 10px; border-radius:999px; font-size:12px; white-space:nowrap; cursor:pointer}
    .avatar{width:34px;height:34px;border-radius:50%;object-fit:cover;border:1px solid var(--border); cursor:pointer}
    .btn{display:inline-flex; align-items:center; gap:8px; padding:10px 14px; border-radius:12px; border:1px solid var(--border); background:linear-gradient(180deg,#0e1833,#0b1326); cursor:pointer; font-weight:600}
    .btn.primary{background:linear-gradient(135deg,#3b82f6,#22c1dc); border-color:transparent}
    .btn.ghost{ background:#162a52; border:1px solid var(--border); color:#eaf2ff; }
    .btn.cashout{
      background: linear-gradient(135deg,#22c55e,#16a34a);
      border-color: transparent;
      box-shadow: 0 6px 14px rgba(34,197,94,.25);
      font-weight:800;
    }
    .btn.cashout[disabled]{ filter:grayscale(.5) brightness(.8); opacity:.8; cursor:not-allowed }

    .big{font-size:22px; font-weight:900}
    .label{font-size:12px; color:var(--muted); letter-spacing:.2px; text-transform:uppercase}
    .muted{color:var(--muted)}

    .card{
      background:linear-gradient(180deg,#0f1a33,#0b1326);
      border:1px solid var(--border); border-radius:16px; padding:16px
    }

    /* games grid/cards */
    .games-grid{display:grid; gap:14px; grid-template-columns:1fr}
    @media(min-width:700px){.games-grid{grid-template-columns:1fr 1fr}}
    @media(min-width:1020px){.games-grid{grid-template-columns:1fr 1fr 1fr}}
    .game-card{
      position:relative;
      min-height:130px;
      display:flex; flex-direction:column; justify-content:flex-end; gap:4px;
      background:linear-gradient(180deg,#0f1a33,#0c152a);
      border:1px solid var(--border); border-radius:16px; padding:16px; cursor:pointer;
      transition:transform .08s ease, box-shadow .12s ease, border-color .12s ease, background .18s ease;
      overflow:hidden;
    }
    .game-card:hover{transform:translateY(-2px); box-shadow:0 8px 18px rgba(0,0,0,.25)}
    .game-card .title{font-size:20px; font-weight:800}
    .game-card .muted{opacity:.9}
    .ribbon{
      position:absolute; top:12px; right:-32px; transform:rotate(35deg);
      background:linear-gradient(135deg,#f59e0b,#fb923c);
      color:#1a1206; font-weight:900; padding:6px 50px; border:1px solid rgba(0,0,0,.2);
      text-shadow:0 1px 0 rgba(255,255,255,.2);
    }

    .owner{margin-top:16px; border-top:1px dashed var(--border); padding-top:12px}

    /* Crash graph */
    .cr-graph-wrap{position:relative; height:240px; background:#0e1833; border:1px solid var(--border); border-radius:16px; overflow:hidden}
    canvas{display:block; width:100%; height:100%}
    .boom{ position:absolute; inset:0; pointer-events:none; opacity:0; }
    .boom.bang{ animation: bang .6s ease-out; }
    @keyframes bang{
      0%{ opacity:.95; background: radial-gradient(350px 350px at var(--x,50%) var(--y,50%), rgba(255,255,255,.9), rgba(239,68,68,.6) 40%, transparent 70%); }
      100%{ opacity:0; background: radial-gradient(800px 800px at var(--x,50%) var(--y,50%), rgba(255,255,255,.0), rgba(239,68,68,.0) 40%, transparent 75%); }
    }

    /* ----- MINES layout ----- */
    .mines-two{ grid-template-columns: 360px 1fr !important; align-items: stretch; display:grid; gap:16px }
    .mines-wrap{ display:grid; place-items:center; height: calc(100vh - 180px); min-height: 420px; padding: 6px; }
    .mines-grid{
      --cell: clamp(48px, min( calc((100vw - 440px)/5), calc((100vh - 320px)/5) ), 110px);
      display:grid; gap:10px; grid-template-columns: repeat(5, var(--cell)); justify-content:center; align-content:center; padding: 6px; width: 100%;
    }
    .tile{
      position:relative; width: var(--cell); aspect-ratio: 1/1; border-radius: clamp(10px, calc(var(--cell)*0.18), 16px);
      border:1px solid var(--border);
      background: radial-gradient(120% 120% at 30% 0%, #19264f 0%, #0c152a 55%), linear-gradient(180deg,#0f1936,#0c152a);
      display:flex; align-items:center; justify-content:center; font-weight:900; font-size: clamp(13px, calc(var(--cell)*0.34), 22px);
      cursor:pointer; user-select:none; transition:transform .09s ease, box-shadow .14s ease, background .18s ease, border-color .14s ease, opacity .14s ease;
      box-shadow: 0 8px 22px rgba(0,0,0,.35), inset 0 0 0 1px rgba(255,255,255,.03);
      overflow:hidden;
    }
    .tile::after{ content:""; position:absolute; inset:0; background: linear-gradient(145deg, rgba(255,255,255,.18), transparent 40%); mix-blend-mode: soft-light; opacity:.22; transition:opacity .2s ease; }
    .tile:hover{ transform:translateY(-1px); box-shadow: 0 10px 26px rgba(0,0,0,.45), inset 0 0 0 1px rgba(255,255,255,.05); }
    .tile .icon{ filter: drop-shadow(0 2px 6px rgba(0,0,0,.45)); }
    .tile.safe{ background: linear-gradient(135deg,#16a34a 0%, #22c55e 70%); border-color: transparent; color:#06240f; }
    .tile.mine{ background: linear-gradient(135deg,#ef4444 0%, #b91c1c 70%); border-color: transparent; color:#260808; }
    .tile.revealed{ cursor:default; }
    .tile.pop{ animation: pop .2s ease; }
    @keyframes pop{ from{ transform: scale(.92); opacity:.7 } to{ transform: scale(1); opacity:1 } }
    .tile.explode{ animation: shake .4s ease-in-out; }
    .tile.explode::before{ content:""; position:absolute; inset:-2px; border-radius: inherit; background: radial-gradient(circle, rgba(255,255,255,.85), rgba(239,68,68,.6) 40%, transparent 70%); opacity:0; animation: exflash .6s ease-out; }
    @keyframes exflash{ 0%{ opacity:.95; transform: scale(.9); } 80%{ opacity:.15); transform: scale(1.05); } 100%{ opacity:0; transform: scale(1); } }
    @keyframes shake{ 0%,100%{ transform: translate(0,0) } 20%{ transform: translate(-2px,-1px) } 40%{ transform: translate(3px,1px) } 60%{ transform: translate(-2px,2px) } 80%{ transform: translate(1px,-2px) } }

    .mines-stats{ display:flex; gap:8px; flex-wrap:wrap; margin-top:10px }
    .stat{ background:#0c1631; border:1px solid var(--border); color:#dce7ff; padding:6px 10px; border-radius:999px; font-size:12px; white-space:nowrap }

    @media (max-width: 980px){
      .mines-two{ grid-template-columns: 1fr !important; }
      .mines-wrap{ height:auto; min-height: 360px; }
      .mines-grid{
        --cell: clamp(52px, min( calc((100vw - 48px)/5), calc((100vh - 320px)/5) ), 100px);
        justify-content:center;
      }
    }

    /* Modal */
    .modal{ position:fixed; inset:0; display:none; align-items:center; justify-content:center; background:rgba(3,6,12,.6); z-index:50; }
    .modal .box{ width:min(640px, 92vw); background:linear-gradient(180deg,#0f1a33,#0c1429); border:1px solid var(--border); border-radius:18px; padding:16px; box-shadow:0 10px 30px rgba(0,0,0,.4) }

    /* Chat Drawer + badges */
    .chat-drawer{
      position:fixed; right:0; top:64px; bottom:0; width:var(--chatW); max-width:90vw;
      transform:translateX(100%); transition: transform .2s ease-out;
      background:linear-gradient(180deg,#0f1a33,#0b1326); border-left:1px solid var(--border);
      display:flex; flex-direction:column; z-index:40;
    }
    .chat-drawer.open{ transform:translateX(0); }
    .chat-head{ display:flex; align-items:center; justify-content:space-between; padding:10px 12px; border-bottom:1px solid var(--border) }
    .chat-body{ flex:1; overflow:auto; padding:10px 12px; }
    .chat-input{ display:flex; gap:8px; padding:10px 12px; border-top:1px solid var(--border) }
    .chat-input input{ flex:1 }
    .msg{ margin-bottom:12px; padding-bottom:8px; border-bottom:1px dashed rgba(255,255,255,.04) }
    .msghead{ display:flex; gap:8px; align-items:center; flex-wrap:wrap }
    .msghead .time{ margin-left:auto; color:var(--muted); font-size:12px }
    .badge{ font-size:10px; padding:3px 7px; border-radius:999px; border:1px solid var(--border); letter-spacing:.2px; }
    .badge.member{ background:#0c1631; color:#cfe6ff }
    .badge.admin{ background:linear-gradient(135deg,#f59e0b,#fb923c); color:#1a1206; border-color:rgba(0,0,0,.2); font-weight:900 }
    .badge.owner{ background:linear-gradient(135deg,#3b82f6,#22c1dc); color:#041018; border-color:transparent; font-weight:900 }
    .level{ font-size:10px; padding:3px 7px; border-radius:999px; background:#0b1f3a; color:#cfe6ff; border:1px solid var(--border) }
    .user-link{ cursor:pointer; font-weight:800 }
    .disabled-note{ padding:8px 12px; font-size:13px; color:#dbe6ff; background:#0c1631; border-bottom:1px solid var(--border) }

    /* Coming Soon hero pages */
    .soon-hero{
      position:relative; overflow:hidden; border-radius:16px; border:1px solid var(--border);
      background: radial-gradient(1200px 500px at -10% -20%, rgba(59,130,246,.25), transparent 60%),
                  linear-gradient(180deg,#0f1a33,#0b1326);
      padding:22px;
    }
    .soon-hero h2{ margin:0; font-size:28px }
    .soon-hero p{ margin:.3rem 0 0; color:var(--muted) }
    .soon-badge{
      position:absolute; top:14px; right:14px;
      background:linear-gradient(135deg,#f59e0b,#fb923c);
      color:#1a1206; padding:6px 10px; border-radius:999px; font-weight:900; border:1px solid rgba(0,0,0,.25)
    }
    .soon-grid{ display:grid; gap:12px; grid-template-columns:1fr; margin-top:12px }
    @media(min-width:800px){ .soon-grid{ grid-template-columns: 1fr 1fr } }
    .soon-card{ background:linear-gradient(180deg,#0f1a33,#0b1326); border:1px solid var(--border); border-radius:16px; padding:14px }

    /* Floating chat button */
    .fab{
      position:fixed; right:18px; bottom:18px; width:56px; height:56px; border-radius:50%;
      background:linear-gradient(135deg,#3b82f6,#22c1dc); border:none; cursor:pointer;
      display:flex; align-items:center; justify-content:center;
      box-shadow:0 14px 30px rgba(59,130,246,.35), 0 4px 10px rgba(0,0,0,.35);
      z-index:45;
      transition:right .2s ease;
    }
    .fab:hover{ transform: translateY(-1px); box-shadow:0 18px 40px rgba(59,130,246,.45), 0 6px 14px rgba(0,0,0,.45); }
    .fab svg{ width:26px; height:26px; fill:#041018 }
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
        </div>
      </div>
      <div class="right" id="authArea"><!-- filled by js --></div>
    </div>
  </div>

  <div class="container" style="padding-top:16px">
    <!-- Games -->
    <div id="page-games">
      <div class="card">
        <div class="games-grid">
          <div class="game-card" id="openCrash" style="background-image: radial-gradient(600px 280px at 10% -10%, rgba(59,130,246,.25), transparent 60%);">
            <div class="title">🚀 Crash</div>
            <div class="muted">Shared rounds • 10s betting • Live cashout</div>
          </div>
          <div class="game-card" id="openMines" style="background-image: radial-gradient(600px 280px at 85% -20%, rgba(34,197,94,.25), transparent 60%);">
            <div class="title">💣 Mines</div>
            <div class="muted">5×5 board • Choose mines • Cash out anytime</div>
          </div>

          <!-- Coming Soon games -->
          <div class="game-card" id="openLimbo">
            <div class="ribbon">COMING SOON</div>
            <div class="title">🎯 Limbo</div>
            <div class="muted">Pick a multiplier and pray</div>
          </div>
          <div class="game-card" id="openTowers">
            <div class="ribbon">COMING SOON</div>
            <div class="title">🗼 Towers</div>
            <div class="muted">Climb floors • Avoid the trap</div>
          </div>
          <div class="game-card" id="openKeno">
            <div class="ribbon">COMING SOON</div>
            <div class="title">🎲 Keno</div>
            <div class="muted">Pick numbers • Big hits</div>
          </div>
          <div class="game-card" id="openPlinko">
            <div class="ribbon">COMING SOON</div>
            <div class="title">🟡 Plinko</div>
            <div class="muted">Drop the puck • Aim for edge</div>
          </div>
          <div class="game-card" id="openBlackjack">
            <div class="ribbon">COMING SOON</div>
            <div class="title">🃏 Blackjack</div>
            <div class="muted">21 or bust • Skill + luck</div>
          </div>
          <div class="game-card" id="openPump">
            <div class="ribbon">COMING SOON</div>
            <div class="title">📈 Pump</div>
            <div class="muted">Ride the wave • Cash before pop</div>
          </div>
          <div class="game-card" id="openCoinflip">
            <div class="ribbon">COMING SOON</div>
            <div class="title">🪙 Coinflip</div>
            <div class="muted">50/50 • Double or none</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Crash -->
    <div id="page-crash" style="display:none">
      <div class="card">
        <div style="display:flex; align-items:center; justify-content:space-between; gap:10px; flex-wrap:wrap">
          <div style="display:flex; align-items:baseline; gap:10px">
            <div class="big" id="crNow">0.00×</div>
            <div class="muted" id="crHint">Loading…</div>
          </div>
          <button class="chip" id="backToGames">← Games</button>
        </div>

        <div class="cr-graph-wrap" style="margin-top:10px">
          <canvas id="crCanvas"></canvas>
          <div id="crBoom" class="boom"></div>
        </div>

        <div style="margin-top:12px">
          <div class="label" style="margin-bottom:4px">Previous Busts</div>
          <div id="lastBusts" class="muted">Loading last rounds…</div>
        </div>

        <div class="games-grid" style="grid-template-columns:1fr 1fr; gap:12px; margin-top:8px">
          <div>
            <div class="label">Bet (DL)</div>
            <input id="crBet" type="number" min="1" step="0.01" placeholder="min 1.00"/>
          </div>
          <div>
            <div class="label">Auto Cashout (×) — optional</div>
            <input id="crCash" type="number" min="1.01" step="0.01" placeholder="e.g. 2.00"/>
          </div>
        </div>

        <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-top:8px">
          <button class="btn primary" id="crPlace">Place Bet</button>
          <button class="btn cashout" id="crCashout" style="display:none">💸 Cash Out</button>
          <span id="crMsg" class="muted"></span>
        </div>

        <div class="card" style="margin-top:14px">
          <div class="label">Your recent rounds</div>
          <div id="crLast" class="muted">—</div>
        </div>
      </div>
    </div>

    <!-- Mines -->
    <div id="page-mines" style="display:none">
      <div class="card">
        <div style="display:flex; align-items:center; justify-content:space-between; gap:10px; flex-wrap:wrap">
          <div class="big">💣 Mines</div>
          <button class="chip" id="backToGames2">← Games</button>
        </div>

        <div class="mines-two" style="margin-top:12px">
          <!-- LEFT: settings & stats + recent games under -->
          <div>
            <div class="label">Bet (DL)</div>
            <input id="mBet" type="number" min="1" step="0.01" placeholder="min 1.00"/>

            <div class="label" style="margin-top:10px">Mines (1–24)</div>
            <input id="mMines" type="number" min="1" max="24" step="1" value="3"/>

            <div class="mines-stats">
              <span class="stat" id="mHash">Commit: —</span>
              <span class="stat" id="mStatus">Status: —</span>
              <span class="stat" id="mPicks">Picks: 0</span>
              <span class="stat" id="mBombs">Mines: 3</span>
            </div>

            <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-top:12px">
              <button class="btn primary" id="mStart">Start Game</button>
              <button class="btn cashout" id="mCash" style="display:none">💸 Cash Out</button>
              <span id="mMsg" class="muted"></span>
            </div>

            <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-top:8px">
              <span class="stat" id="mMult">Multiplier: 1.0000×</span>
              <span class="stat" id="mPotential">Potential: —</span>
            </div>

            <div class="card" style="margin-top:14px">
              <div class="label">Recent Mines Games</div>
              <div id="mHist" class="muted">—</div>
            </div>
          </div>

          <!-- RIGHT: board fills remaining space -->
          <div class="mines-wrap">
            <div class="mines-grid" id="mGrid"></div>
          </div>
        </div>
      </div>
    </div>

    <!-- Referral -->
    <div id="page-ref" style="display:none">
      <div class="card">
        <div class="label">Referral</div>
        <div id="refContent">Loading…</div>
      </div>
    </div>

    <!-- Promo Codes -->
    <div id="page-promo" style="display:none">
      <div class="card">
        <div class="label">Promo Codes</div>
        <div class="games-grid" style="grid-template-columns:1fr 1fr">
          <div>
            <div class="label">Redeem a code</div>
            <div style="display:flex; gap:8px; align-items:center">
              <input id="promoInput" placeholder="e.g. WELCOME10" />
              <button class="btn primary" id="redeemBtn">Redeem</button>
            </div>
            <div id="promoMsg" class="muted" style="margin-top:8px"></div>
          </div>
          <div>
            <div class="label">Your redemptions</div>
            <div id="myCodes" class="muted">—</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Profile -->
    <div id="page-profile" style="display:none">
      <div class="card">
        <div class="label">Profile</div>
        <div id="profileBox">Loading…</div>

        <div id="ownerPanel" class="owner" style="display:none">
          <div class="label">Owner Panel</div>
          <div class="games-grid" style="grid-template-columns:2fr 1fr 2fr auto">
            <div><div class="label">Discord ID or &lt;@mention&gt;</div><input id="tIdent" placeholder="ID or <@id>"/></div>
            <div><div class="label">Amount (+/- DL)</div><input id="tAmt" type="text" placeholder="10 or -5.25"/></div>
            <div><div class="label">Reason (optional)</div><input id="tReason" placeholder="promo/correction/prize"/></div>
            <div style="align-self:end"><button class="btn primary" id="tApply">Apply</button></div>
          </div>
          <div id="tMsg" class="muted" style="margin-top:8px"></div>

          <div class="label" style="margin-top:12px">Roles</div>
          <div class="games-grid" style="grid-template-columns:2fr auto auto">
            <div><div class="label">Target</div><input id="rIdent" placeholder="ID or <@id>"/></div>
            <button class="btn" id="rAdmin">Make ADMIN</button>
            <button class="btn" id="rMember">Make MEMBER</button>
          </div>
          <div id="rMsg" class="muted" style="margin-top:8px"></div>

          <div class="label" style="margin-top:12px">Timeouts</div>
          <div class="games-grid" style="grid-template-columns:2fr 1fr 1fr auto auto">
            <div><div class="label">Target</div><input id="xIdent" placeholder="ID or <@id>"/></div>
            <div><div class="label">Seconds</div><input id="xSecs" type="number" value="600"/></div>
            <div><div class="label">Reason</div><input id="xReason" placeholder="spam / rude / etc"/></div>
            <button class="btn" id="xSite">Website Timeout</button>
            <button class="btn" id="xBoth">Site + Discord</button>
          </div>
          <div id="xMsg" class="muted" style="margin-top:8px"></div>

          <div class="label" style="margin-top:12px">Create Promo Code</div>
          <div class="games-grid" style="grid-template-columns:1fr 1fr 1fr">
            <div><div class="label">Code (optional)</div><input id="cCode" placeholder="auto-generate if empty"/></div>
            <div><div class="label">Amount (DL)</div><input id="cAmount" type="text" placeholder="e.g. 10 or 1.24"/></div>
            <div><div class="label">Max Uses</div><input id="cMax" type="number" placeholder="e.g. 100"/></div>
          </div>
          <div style="margin-top:8px"><button class="btn primary" id="cMake">Create</button> <span id="cMsg" class="muted"></span></div>
        </div>
      </div>
    </div>

    <!-- Coming Soon Pages (unchanged labels) -->
    <div id="page-limbo" style="display:none"><div class="soon-hero"><div class="soon-badge">Coming Soon</div><h2>🎯 Limbo</h2><p>Set your multiplier and hold your breath. Higher risk, higher reward.</p><div class="soon-grid"><div class="soon-card"><div class="label">RTP</div><div class="big">~96–98%</div></div><div class="soon-card"><div class="label">Min Bet</div><div class="big">1.00 DL</div></div></div><button class="chip" id="backFromLimbo" style="margin-top:12px">← Back to Games</button></div></div>
    <div id="page-towers" style="display:none"><div class="soon-hero"><div class="soon-badge">Coming Soon</div><h2>🗼 Towers</h2><p>Pick one safe tile per floor. Each floor multiplies your win—miss and you fall.</p><div class="soon-grid"><div class="soon-card"><div class="label">Floors</div><div class="big">8–12</div></div><div class="soon-card"><div class="label">House Edge</div><div class="big">Fair & transparent</div></div></div><button class="chip" id="backFromTowers" style="margin-top:12px">← Back to Games</button></div></div>
    <div id="page-keno" style="display:none"><div class="soon-hero"><div class="soon-badge">Coming Soon</div><h2>🎲 Keno</h2><p>Choose numbers and let the draw decide. More hits = bigger payouts.</p><div class="soon-grid"><div class="soon-card"><div class="label">Board</div><div class="big">10×10</div></div><div class="soon-card"><div class="label">Picks</div><div class="big">1–10</div></div></div><button class="chip" id="backFromKeno" style="margin-top:12px">← Back to Games</button></div></div>
    <div id="page-plinko" style="display:none"><div class="soon-hero"><div class="soon-badge">Coming Soon</div><h2>🟡 Plinko</h2><p>Drop balls through a peg board—pray for the edges!</p><div class="soon-grid"><div class="soon-card"><div class="label">Rows</div><div class="big">8–16</div></div><div class="soon-card"><div class="label">Risk</div><div class="big">Low / Med / High</div></div></div><button class="chip" id="backFromPlinko" style="margin-top:12px">← Back to Games</button></div></div>
    <div id="page-blackjack" style="display:none"><div class="soon-hero"><div class="soon-badge">Coming Soon</div><h2>🃏 Blackjack</h2><p>Beat the dealer to 21. Strategy meets luck—splits & doubles supported.</p><div class="soon-grid"><div class="soon-card"><div class="label">Decks</div><div class="big">4–6</div></div><div class="soon-card"><div class="label">Payout</div><div class="big">Blackjack 3:2</div></div></div><button class="chip" id="backFromBlackjack" style="margin-top:12px">← Back to Games</button></div></div>
    <div id="page-pump" style="display:none"><div class="soon-hero"><div class="soon-badge">Coming Soon</div><h2>📈 Pump</h2><p>Ride a volatile curve—cash before the pop. Like Crash… crankier.</p><div class="soon-grid"><div class="soon-card"><div class="label">Mode</div><div class="big">Volatility+</div></div><div class="soon-card"><div class="label">Max Mult</div><div class="big">Huge 👀</div></div></div><button class="chip" id="backFromPump" style="margin-top:12px">← Back to Games</button></div></div>
    <div id="page-coinflip" style="display:none"><div class="soon-hero"><div class="soon-badge">Coming Soon</div><h2>🪙 Coinflip</h2><p>50/50 double-up. Create or join flips with friends.</p><div class="soon-grid"><div class="soon-card"><div class="label">Edge</div><div class="big">Low, fair fee</div></div><div class="soon-card"><div class="label">Min Bet</div><div class="big">1.00 DL</div></div></div><button class="chip" id="backFromCoinflip" style="margin-top:12px">← Back to Games</button></div></div>

    <!-- Login card -->
    <div id="loginCard" class="card" style="display:none">
      <div class="label">Get Started</div>
      <p>Use Discord to log in and see your balance, play games, and chat.</p>
      <div style="display:flex; gap:8px; flex-wrap:wrap">
        <a class="btn primary" href="/login">Login with Discord</a>
      </div>
    </div>
  </div>

  <!-- Balance Modal -->
  <div class="modal" id="modal">
    <div class="box">
      <div style="display:flex; justify-content:space-between; align-items:center; gap:10px">
        <div class="big">Balance — Deposit / Withdraw</div>
        <button class="btn ghost" id="mClose">Close</button>
      </div>
      <div class="muted" style="margin-top:8px">
        <p><b>Deposit:</b> Join our Discord and type <code>.deposit</code> in the <i>#deposit</i> channel.</p>
        <p><b>Withdraw:</b> In Discord type <code>.withdraw</code>. (We’ll wire up fulfillment later.)</p>
      </div>
    </div>
  </div>

  <!-- Profile Modal (click usernames in chat) -->
  <div class="modal" id="profileModal">
    <div class="box" id="profileModalBox">
      <div style="display:flex; justify-content:space-between; align-items:center; gap:10px">
        <div class="big" id="pmTitle">Profile</div>
        <button class="btn ghost" id="pmClose">Close</button>
      </div>
      <div id="pmBody" style="margin-top:10px">Loading…</div>
    </div>
  </div>

  <!-- Global Chat Drawer -->
  <div class="chat-drawer" id="chatDrawer">
    <div class="chat-head">
      <div><b>Global Chat</b></div>
      <div style="display:flex; gap:8px; align-items:center">
        <span id="chatNote" class="level"></span>
        <button class="btn ghost" id="chatClose">Close</button>
      </div>
    </div>
    <div class="disabled-note" id="chatDisabled" style="display:none"></div>
    <div class="chat-body" id="chatBody"></div>
    <div class="chat-input">
      <input id="chatText" placeholder="Type a message (Lv 5+)" maxlength="300"/>
      <button class="btn primary" id="chatSend">Send</button>
    </div>
  </div>

  <!-- Floating Chat Button -->
  <button class="fab" id="fabChat" title="Open chat">
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M20 2H4a2 2 0 0 0-2 2v18l4-4h14a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2zM6 9h12v2H6V9zm0-4h12v2H6V5zm0 8h8v2H6v-2z"/>
    </svg>
  </button>

  <script>
    const HOUSE_EDGE_MINES = __HOUSE_EDGE_MINES__;
    const OWNER_ID = "__OWNER_ID__";
    function qs(id){return document.getElementById(id)}
    const tabGames=qs('tab-games'), tabRef=qs('tab-ref'), tabPromo=qs('tab-promo');
    const pages=['page-games','page-crash','page-mines','page-ref','page-promo','page-profile','page-limbo','page-towers','page-keno','page-plinko','page-blackjack','page-pump','page-coinflip'].map(id=>qs(id));
    const pgGames=qs('page-games'), pgCrash=qs('page-crash'), pgMines=qs('page-mines'), pgRef=qs('page-ref'), pgPromo=qs('page-promo'), pgProfile=qs('page-profile');

    function fmtDL(n){ const v=Number(n||0); return `💎 ${v.toLocaleString(undefined,{minimumFractionDigits:2, maximumFractionDigits:2})} DL`; }
    async function j(u, opt={}){ const r=await fetch(u, Object.assign({credentials:'include'},opt)); if(!r.ok) throw new Error((await r.text())||r.statusText); return r.json(); }

    function showOnly(id){
      for(const p of pages) if(p) p.style.display = (p.id===id?'':'none');
      [tabGames,tabRef,tabPromo].forEach(t=>t.classList.remove('active'));
      if(id==='page-games') tabGames.classList.add('active');
      if(id==='page-ref') tabRef.classList.add('active');
      if(id==='page-promo') tabPromo.classList.add('active');
      window.scrollTo({top:0, behavior:'smooth'});
    }

    qs('homeLink').onclick=(e)=>{ e.preventDefault(); showOnly('page-games'); };
    tabGames.onclick=()=>showOnly('page-games');
    tabRef.onclick=()=>{ showOnly('page-ref'); renderReferral(); };
    tabPromo.onclick=()=>{ showOnly('page-promo'); renderPromos(); };

    qs('openCrash').onclick=()=>showOnly('page-crash');
    qs('openMines').onclick=()=>{ showOnly('page-mines'); renderMines(); };
    const openers=[['openLimbo','page-limbo','backFromLimbo'],['openTowers','page-towers','backFromTowers'],['openKeno','page-keno','backFromKeno'],['openPlinko','page-plinko','backFromPlinko'],['openBlackjack','page-blackjack','backFromBlackjack'],['openPump','page-pump','backFromPump'],['openCoinflip','page-coinflip','backFromCoinflip']];
    for(const [btn,page,back] of openers){ const b=qs(btn), bk=qs(back); if(b) b.onclick=()=>showOnly(page); if(bk) bk.onclick=()=>showOnly('page-games'); }
    qs('backToGames').onclick=()=>showOnly('page-games');
    qs('backToGames2').onclick=()=>showOnly('page-games');

    // modal
    function openModal(){ qs('modal').style.display='flex'; }
    function closeModal(){ qs('modal').style.display='none'; }
    qs('mClose').onclick = closeModal;
    qs('modal').addEventListener('click', (e)=>{ if(e.target.id==='modal') closeModal(); });

    // profile modal
    function openPm(){ qs('profileModal').style.display='flex'; }
    function closePm(){ qs('profileModal').style.display='none'; }
    qs('pmClose').onclick = closePm;
    qs('profileModal').addEventListener('click', (e)=>{ if(e.target.id==='profileModal') closePm(); });

    function safeAvatar(me){
      if(!me) return 'https://cdn.discordapp.com/embed/avatars/1.png?size=64';
      if(typeof me === 'string') return me;
      return me.avatar_url || 'https://cdn.discordapp.com/embed/avatars/1.png?size=64';
    }

    // Header/auth
    async function renderHeader(){
      const auth = qs('authArea');
      try{
        const me = await j('/api/me');
        const bal = await j('/api/balance');
        auth.innerHTML = `
          <span class="chip" id="balanceBtn">${fmtDL(bal.balance)}</span>
          <img class="avatar" id="avatarBtn" src="${safeAvatar(me)}" title="${me.username}" onerror="this.src='https://cdn.discordapp.com/embed/avatars/1.png?size=64'">
        `;
        qs('balanceBtn').onclick = openModal;
        qs('avatarBtn').onclick = ()=>{ showOnly('page-profile'); renderProfile(); };
        qs('loginCard').style.display='none';
      }catch(e){
        auth.innerHTML = `<a class="btn primary" href="/login">Login with Discord</a>`;
        qs('loginCard').style.display='';
      }
    }

    // Profile / Referral / Promos
    async function renderProfile(){
      try{
        const me = await j('/api/me'); const prof = await j('/api/profile');
        const lvl=prof.level, pct=Math.max(0,Math.min(100,prof.progress_pct));
        const roleBadge = `<span class="badge ${prof.role}">${prof.role.toUpperCase()}</span>`;
        qs('profileBox').innerHTML = `
          <div style="display:flex; gap:12px; align-items:center; flex-wrap:wrap">
            <img src="${safeAvatar(me)}" class="avatar" style="width:56px; height:56px">
            <div><div class="big">${me.username} ${roleBadge}</div><div class="muted">ID: ${me.id}</div></div>
            <div style="margin-left:auto; display:flex; gap:8px; align-items:center"><a class="btn" href="/logout">Logout</a></div>
          </div>
          <div class="games-grid" style="grid-template-columns:1fr 1fr; margin-top:12px">
            <div class="card" style="padding:12px">
              <div class="label">Balance</div><div class="big">${fmtDL(prof.balance)}</div>
              <div class="muted" style="margin-top:6px">Click your balance in the header for Deposit/Withdraw instructions.</div>
            </div>
            <div class="card" style="padding:12px">
              <div class="label">Level</div><div><b>Level ${lvl}</b> — XP ${prof.xp} / ${((lvl-1)*100)+100}</div>
              <div style="height:10px; background:#0e1833; border:1px solid var(--border); border-radius:999px; overflow:hidden; margin-top:8px">
                <div style="height:100%; width:${pct}%; background:linear-gradient(90deg,#22c1dc,#3b82f6)"></div>
              </div><div class="muted" style="margin-top:6px">${prof.progress}/${prof.next_needed} XP to next level</div>
            </div>
          </div>
        `;
        const ownerPanel = qs('ownerPanel');
        if(me.role === 'owner' || me.role === 'admin' || me.id === OWNER_ID){ ownerPanel.style.display=''; } else ownerPanel.style.display='none';

        // Balance adjust
        const tApply = qs('tApply'); if(tApply){
          tApply.onclick = async ()=>{
            const identifier = qs('tIdent').value.trim();
            const amount = qs('tAmt').value.trim();
            const reason = qs('tReason').value.trim() || null;
            const msg = qs('tMsg'); msg.textContent = '';
            try{
              const r = await j('/api/admin/adjust', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({identifier, amount, reason})});
              msg.textContent = 'Updated. New balance for ' + identifier + ' = ' + fmtDL(r.new_balance);
              renderHeader();
            }catch(e){ msg.textContent = 'Error: '+e.message; }
          };
        }
        // Roles
        const rAdmin = qs('rAdmin'), rMember = qs('rMember');
        if(rAdmin) rAdmin.onclick = async ()=>{
          const identifier = qs('rIdent').value.trim(); const msg=qs('rMsg'); msg.textContent='';
          try{ const r=await j('/api/admin/role',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({identifier, role:'admin'})});
               msg.textContent='Set role to ADMIN'; }catch(e){ msg.textContent='Error: '+e.message; }
        };
        if(rMember) rMember.onclick = async ()=>{
          const identifier = qs('rIdent').value.trim(); const msg=qs('rMsg'); msg.textContent='';
          try{ const r=await j('/api/admin/role',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({identifier, role:'member'})});
               msg.textContent='Set role to MEMBER'; }catch(e){ msg.textContent='Error: '+e.message; }
        };

        // Timeouts
        const xSite = qs('xSite'), xBoth = qs('xBoth');
        if(xSite) xSite.onclick = async ()=>{
          const identifier=qs('xIdent').value.trim(); const seconds=parseInt(qs('xSecs').value,10)||600; const reason=qs('xReason').value.trim()||null;
          const msg=qs('xMsg'); msg.textContent='';
          try{ await j('/api/admin/timeout_site',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({identifier,seconds,reason})});
               msg.textContent='Website chat timeout applied.'; }catch(e){ msg.textContent='Error: '+e.message; }
        };
        if(xBoth) xBoth.onclick = async ()=>{
          const identifier=qs('xIdent').value.trim(); const seconds=parseInt(qs('xSecs').value,10)||600; const reason=qs('xReason').value.trim()||null;
          const msg=qs('xMsg'); msg.textContent='';
          try{ await j('/api/admin/timeout_both',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({identifier,seconds,reason})});
               msg.textContent='Website + Discord timeout requested.'; }catch(e){ msg.textContent='Error: '+e.message; }
        };

        // Promo codes
        const cMake = qs('cMake'); if(cMake){
          cMake.onclick = async ()=>{
            const code = qs('cCode').value.trim() || null;
            const amount = qs('cAmount').value.trim() || "0";
            const max_uses = parseInt(qs('cMax').value,10) || 1;
            const msg = qs('cMsg'); msg.textContent='';
            try{
              const r = await j('/api/admin/promo/create',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({code, amount, max_uses})});
              msg.textContent = 'Created: '+r.code+' for '+amount+' DL';
            }catch(e){ msg.textContent = 'Error: '+e.message; }
          };
        }
      }catch(e){}
    }

    async function renderReferral(){
      try{
        const ref = await j('/api/referral/state');
        if(ref.name){
          const link = location.origin + '/?ref=' + encodeURIComponent(ref.name);
          qs('refContent').innerHTML = `<p>Your referral name: <b>${ref.name}</b></p><p>Share this link:</p><p><code>${link}</code></p>`;
        }else{
          qs('refContent').innerHTML = `
            <p>Claim a referral name to get your link.</p>
            <div style="display:flex; gap:8px; align-items:center; max-width:420px">
              <input id="refName" placeholder="3-20 letters/numbers/_-" />
              <button class="btn primary" id="claimBtn">Claim</button>
            </div><div id="refMsg" class="muted" style="margin-top:8px"></div>`;
          qs('claimBtn').onclick = async ()=>{
            const name = document.getElementById('refName').value.trim(), msg=qs('refMsg'); msg.textContent='';
            try{ const r=await j('/api/profile/set_name',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})});
                 msg.textContent='Saved. Your link: '+location.origin+'/?ref='+encodeURIComponent(r.name); }
            catch(e){ msg.textContent='Error: '+e.message; }
          };
        }
      }catch(e){}
    }
    async function renderPromos(){
      try{
        const mine = await j('/api/promo/my');
        document.getElementById('myCodes').innerHTML = mine.rows.length
          ? '<ul>' + mine.rows.map(r=>`<li><code>${r.code}</code> — ${new Date(r.redeemed_at).toLocaleString()}</li>`).join('') + '</ul>'
          : 'No redemptions yet.';
      }catch(e){
        document.getElementById('myCodes').textContent='Failed to load.';
      }
    }

    // -------- Crash (slow, curved line + explosion) --------
    const crNowEl = qs('crNow'), crHint = qs('crHint'), crMsg = qs('crMsg');
    const lastBustsEl = qs('lastBusts'), cashBtn = qs('crCashout'), placeBtn = qs('crPlace');

    let crPhase = 'betting', lastPhase = 'betting';
    let roundId = null;
    let haveActiveBet = false;

    // Canvas
    const canv = qs('crCanvas'); const ctx = canv.getContext('2d');
    const crBoom = qs('crBoom');
    function resizeCanvas(){
      const dpr = window.devicePixelRatio || 1;
      const r = canv.getBoundingClientRect();
      canv.width = Math.floor(r.width*dpr);
      canv.height = Math.floor(r.height*dpr);
      ctx.setTransform(dpr,0,0,dpr,0,0);
      redrawAxis();
    }
    window.addEventListener('resize', resizeCanvas);
    function redrawAxis(){
      const w = canv.clientWidth, h = canv.clientHeight;
      ctx.clearRect(0,0,w,h);
      ctx.globalAlpha = 1;
      ctx.strokeStyle = '#203255';
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.35;
      for(let i=0;i<=5;i++){
        const y = h - (i*h/5);
        ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(w,y); ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }
    resizeCanvas();

    // Multiplier animation (very slow start)
    let clientMult = 0.00;
    let serverTarget = 1.00;
    let rafId = null, runStartTs = 0, lastTs = 0, stepAcc = 0, lastDrawnM = 0.00;

    function mapPoint(mult){
      const w = canv.clientWidth, h = canv.clientHeight;
      const maxM = 20.0;
      const yf = Math.min(1, Math.log(Math.max(1.0001, mult)) / Math.log(maxM));
      const y = h - (h * yf);
      const xf = Math.pow(yf, 1.35);
      const x = (w * (0.12 + 0.76 * xf));
      return [x,y];
    }
    function resetGraph(){ redrawAxis(); lastDrawnM = 0.00; clientMult = 0.00; }
    function drawStep(toMult){
      const [px,py] = mapPoint(lastDrawnM || 1.00);
      const [x,y]   = mapPoint(toMult   || 1.00);
      ctx.lineJoin = 'round'; ctx.lineCap = 'round';
      ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 3.5;
      ctx.shadowColor = 'rgba(255,255,255,.25)'; ctx.shadowBlur = 6;
      ctx.beginPath();
      const cx = (px + x) / 2, cy = Math.min(py, y) - 8;
      ctx.moveTo(px, py);
      ctx.quadraticCurveTo(cx, cy, x, y);
      ctx.stroke();
      ctx.shadowBlur = 0;
      lastDrawnM = toMult;
    }
    function startRunAnim(){
      resetGraph();
      runStartTs = performance.now(); lastTs = 0; stepAcc = 0;
      clientMult = 0.00; serverTarget = 1.00;
      crNowEl.textContent = '0.00×';
      if(rafId) cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(tick);
    }
    function stopRunAnim(){ if(rafId){ cancelAnimationFrame(rafId); rafId=null; } }
    function triggerCrashExplosion(bust){
      const w = canv.clientWidth, h = canv.clientHeight;
      const p = mapPoint(Math.max(1.0, bust));
      const rx = (p[0]/w*100).toFixed(2)+'%';
      const ry = (p[1]/h*100).toFixed(2)+'%';
      crBoom.style.setProperty('--x', rx);
      crBoom.style.setProperty('--y', ry);
      crBoom.classList.remove('bang'); void crBoom.offsetWidth; crBoom.classList.add('bang');
    }
    function tick(ts){
      if(!lastTs) lastTs = ts;
      const dt = Math.min(0.1, (ts - lastTs)/1000); lastTs = ts;
      const tSec = Math.max(0, (ts - runStartTs)/1000);
      // Slow accel curve
      const baseRate = 0.008, maxRate = 0.16, tau = 8.0;
      const ratePerSec = baseRate + (maxRate - baseRate) * (1 - Math.exp(-tSec / tau));
      stepAcc += ratePerSec * dt;
      while(stepAcc >= 0.01 && clientMult < serverTarget){
        const next = Math.min(serverTarget, +(clientMult + 0.01).toFixed(2));
        clientMult = next;
        crNowEl.textContent = clientMult.toFixed(2)+'×';
        drawStep(clientMult);
        stepAcc -= 0.01;
      }
      rafId = requestAnimationFrame(tick);
    }

    function bustChipColor(v){
      if(v < 1.5) return 'linear-gradient(180deg,#3a1020,#2a0b18)';
      if(v < 2.0) return 'linear-gradient(180deg,#3a2d10,#2a200b)';
      if(v < 3.0) return 'linear-gradient(180deg,#0f2f16,#0b2110)';
      return 'linear-gradient(180deg,#0f2b38,#0b1f2a)';
    }

    async function refreshCrash(){
      try{
        const s = await j('/api/crash/state');
        roundId = s.round_id; crPhase = s.phase;

        const lbs = (s.last_busts||[]).slice().reverse(); // oldest -> newest in UI
        lastBustsEl.innerHTML = lbs.length
          ? lbs.map(v=>`<span class="chip" style="border-color:transparent;background:${bustChipColor(v)}">${v.toFixed(2)}×</span>`).join(' ')
          : 'No history yet.';

        haveActiveBet = !!(s.your_bet && !s.your_bet.resolved);

        if(haveActiveBet){
          placeBtn.style.display = 'none';
          cashBtn.style.display = '';
          cashBtn.disabled = (crPhase!=='running');
        }else{
          placeBtn.style.display = '';
          cashBtn.style.display = 'none';
        }

        if(lastPhase !== 'running' && crPhase === 'running'){
          crHint.textContent = 'Round running… Tap Cash Out anytime.';
          startRunAnim();
        }
        if(lastPhase === 'running' && crPhase !== 'running'){
          stopRunAnim();
          if(s.bust){ crNowEl.textContent = s.bust.toFixed(2)+'×'; triggerCrashExplosion(s.bust); }
          crHint.textContent = 'Preparing next round…';
          cashBtn.disabled = true;
        }
        if(crPhase==='betting'){
          const left = Math.max(0, Math.round((Date.parse(s.betting_ends_at) - Date.now())/1000));
          crHint.textContent = `Betting open — closes in ${left}s`;
          crNowEl.textContent = '0.00×';
          resetGraph();
        }

        lastPhase = crPhase;

        // Your recent rounds
        try{
          const rh = await j('/api/crash/history');
          const rows = rh.rows||[];
          qs('crLast').innerHTML = rows.length
            ? `<table><thead><tr><th>When</th><th>Bet</th><th>Cashout</th><th>Bust</th><th>Win</th></tr></thead>
                <tbody>${rows.map(r=>`<tr><td>${new Date(r.created_at).toLocaleString()}</td><td>${fmtDL(r.bet)}</td><td>${r.cashout.toFixed(2)}×</td><td>${r.bust.toFixed(2)}×</td><td style="color:${r.win>0?'#34d399':'#ef4444'}">${r.win>0?fmtDL(r.win):'—'}</td></tr>`).join('')}</tbody></table>`
            : '—';
        }catch{}
      }catch(e){
        crHint.textContent = 'Disconnected. Reconnecting…';
      }
    }
    async function pollNow(){
      if(crPhase!=='running') return;
      try{
        const n = await j('/api/crash/now');
        if(n.phase==='running'){
          serverTarget = Math.max(serverTarget, Math.round(n.multiplier*100)/100);
        }
      }catch(e){}
    }

    qs('crPlace').onclick = async ()=>{
      try{
        const bet = parseFloat(document.getElementById('crBet').value);
        const cashVal = document.getElementById('crCash').value.trim();
        let cash = cashVal==='' ? 1000 : parseFloat(cashVal);
        if(!isFinite(bet) || bet < 1) throw new Error('Enter a bet of at least 1.00 DL.');
        if(!isFinite(cash) || cash < 1.01) throw new Error('Auto cashout must be at least 1.01×, or leave empty.');
        await j('/api/crash/bet', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({bet, cashout: cash})});
        crMsg.textContent = 'Bet placed for this round.';
        await renderHeader();
        haveActiveBet = true;
        qs('crPlace').style.display='none';
        qs('crCashout').style.display=''; qs('crCashout').disabled = (crPhase!=='running');
      }catch(e){ crMsg.textContent = 'Error: '+e.message; }
    };
    qs('crCashout').onclick = async ()=>{
      if(qs('crCashout').disabled) return;
      try{
        const r = await j('/api/crash/cashout', {method:'POST'});
        crMsg.textContent = 'Cashed out at '+r.multiplier.toFixed(2)+'× • Won '+fmtDL(r.win);
        await renderHeader();
        haveActiveBet = false;
        qs('crCashout').style.display = 'none';
        qs('crPlace').style.display = '';
      }catch(e){ crMsg.textContent = 'Cashout failed: '+e.message; }
    };

    // ---- MINES ----
    const mGrid = qs('mGrid'), mStart = qs('mStart'), mCash = qs('mCash'), mMsg = qs('mMsg');
    const mHash = qs('mHash'), mStatus = qs('mStatus'), mPotential = qs('mPotential'), mMult = qs('mMult');
    const mPicksEl = qs('mPicks'), mBombsEl = qs('mBombs');

    let mActive = false, mPicks = 0, mGameId = null, mMines = 3, mBet = 0;

    function buildGrid(){
      mGrid.innerHTML = '';
      for(let i=0;i<25;i++){
        const tile = document.createElement('div');
        tile.className = 'tile';
        tile.dataset.idx = i;
        const icon = document.createElement('div');
        icon.className = 'icon';
        tile.appendChild(icon);
        tile.onclick = onTileClick;
        mGrid.appendChild(tile);
      }
    }

    function updateGridReveal(board=null, explodedIndex=null){
      for(let i=0;i<25;i++){
        const tile = mGrid.children[i];
        const icon = tile.firstChild;
        const revealed = ((mPicks>>i)&1)===1;
        tile.classList.remove('safe','mine','revealed','pop','explode');
        icon.textContent = '';

        if(revealed){
          tile.classList.add('safe','revealed','pop');
          icon.textContent = '💎';
        }
        if(board){
          if(board[i]==='1'){
            tile.classList.remove('safe');
            tile.classList.add('mine','revealed');
            icon.textContent = '💥';
            if(explodedIndex===i){ tile.classList.add('explode'); }
          }
        }
      }
    }

    function countBits(x){ return x.toString(2).split('0').join('').length; }
    function clientMinesMultiplier(mines, picks){
      if(picks<=0) return 1.0;
      let t=1.0;
      for(let i=0;i<picks;i++){
        t *= (25 - i) / Math.max(1, (25 - mines - i));
        t *= (1 - HOUSE_EDGE_MINES);
      }
      return t;
    }
    function updateMinesStats(){
      mPicksEl.textContent = 'Picks: ' + countBits(mPicks);
      mBombsEl.textContent = 'Mines: ' + mMines;
      const p = countBits(mPicks);
      const mult = clientMinesMultiplier(mMines, p);
      mMult.textContent = 'Multiplier: ' + mult.toFixed(4) + '×';
      const potential = (p>0? (mBet * mult) : mBet);
      mPotential.textContent = 'Potential: ' + fmtDL(potential);
    }

    async function onTileClick(e){
      if(!mActive) return;
      const el = e.currentTarget;
      const idx = parseInt(el.dataset.idx,10);
      if(((mPicks>>idx)&1)===1) return;
      try{
        const r = await j('/api/mines/pick',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({index: idx})});
        if(r.status==='lost'){
          mActive=false; mStatus.textContent='Status: Lost';
          updateGridReveal(r.board, r.index);
          mMult.textContent='Multiplier: —';
          mPotential.textContent='Potential: —';
          mMsg.textContent='💥 You hit a mine.';
          mStart.style.display=''; mCash.style.display='none';
          renderHeader();
          renderMinesHistory();
        }else{
          mPicks = r.picks;
          updateGridReveal();
          updateMinesStats();
          mStatus.textContent='Status: Active';
          mMsg.textContent='';
          if(countBits(mPicks)>0){ mCash.disabled=false; }
        }
      }catch(err){ mMsg.textContent='Error: '+err.message; }
    }

    async function minesStart(){
      try{
        const bet = parseFloat(document.getElementById('mBet').value);
        const mines = parseInt(document.getElementById('mMines').value,10);
        if(!isFinite(bet) || bet < 1) throw new Error('Enter a bet of at least 1.00 DL.');
        if(!isFinite(mines) || mines<1 || mines>24) throw new Error('Mines must be 1–24.');
        const r = await j('/api/mines/start',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({bet, mines})});
        mActive=true; mPicks=0; mGameId=r.id; mMines=mines; mBet=bet;
        mHash.textContent = 'Commit: '+r.hash;
        mStatus.textContent = 'Status: Active';
        mMsg.textContent='';
        buildGrid(); updateGridReveal(); updateMinesStats();
        mStart.style.display='none'; mCash.style.display=''; mCash.disabled=true;
        renderHeader();
        renderMinesHistory();
      }catch(err){ mMsg.textContent='Error: '+err.message; }
    }
    async function minesCash(){
      try{
        const r = await j('/api/mines/cashout',{method:'POST'});
        mActive=false;
        mStatus.textContent='Status: Cashed';
        mMsg.textContent='✅ Cashed out: '+fmtDL(r.win);
        updateGridReveal(r.board);
        mMult.textContent='Multiplier: —';
        mPotential.textContent='Potential: —';
        mStart.style.display=''; mCash.style.display='none';
        renderHeader();
        renderMinesHistory();
      }catch(err){ mMsg.textContent='Error: '+err.message; }
    }
    qs('mStart').onclick = minesStart;
    qs('mCash').onclick = minesCash;

    async function renderMines(){
      buildGrid(); updateGridReveal(); updateMinesStats();
      try{
        const s = await j('/api/mines/state');
        if(s && s.status==='active'){
          mActive=true; mPicks=s.picks; mGameId=s.id;
          mMines=s.mines; mBet=s.bet;
          mHash.textContent='Commit: '+s.hash;
          mStatus.textContent='Status: Active';
          qs('mStart').style.display='none'; qs('mCash').style.display=''; qs('mCash').disabled=(countBits(mPicks)<1);
          updateGridReveal(); updateMinesStats();
        }else{
          mActive=false; mPicks=0; mGameId=null;
          mHash.textContent='Commit: —'; mStatus.textContent='Status: —';
          qs('mStart').style.display=''; qs('mCash').style.display='none';
          mMult.textContent='Multiplier: 1.0000×';
          mPotential.textContent='Potential: —';
          updateGridReveal(); updateMinesStats();
        }
      }catch(e){
        mActive=false; mPicks=0; mGameId=null;
        mHash.textContent='Commit: —'; mStatus.textContent='Status: —';
        qs('mStart').style.display=''; qs('mCash').style.display='none';
        mMult.textContent='Multiplier: 1.0000×';
        mPotential.textContent='Potential: —';
      }
      renderMinesHistory();
    }

    async function renderMinesHistory(){
      try{
        const h = await j('/api/mines/history');
        const rows = h.rows;
        document.getElementById('mHist').innerHTML = rows.length
          ? `<table><thead><tr><th>ID</th><th>Bet</th><th>Mines</th><th>Result</th><th>Hash</th><th>Seed</th></tr></thead>
              <tbody>${
                rows.map(r=>`
                  <tr>
                    <td>#${r.id}</td>
                    <td>${fmtDL(r.bet)}</td>
                    <td>${r.mines}</td>
                    <td style="color:${r.win>0?'#34d399':'#ef4444'}">${r.win>0?('Won '+fmtDL(r.win)):'Lost'}</td>
                    <td><code>${r.hash.slice(0,10)}…</code></td>
                    <td><code>${(r.seed||'').slice(0,8)}…</code></td>
                  </tr>`).join('')
              }</tbody></table>`
          : 'No games yet.';
      }catch(e){
        document.getElementById('mHist').textContent = 'Unable to load history.';
      }
    }

    // ---- Chat (with badges + profile modal) ----
    const drawer = qs('chatDrawer'), chatBody=qs('chatBody'), chatText=qs('chatText'), chatSend=qs('chatSend');
    const chatNote = qs('chatNote'), chatDisabled = qs('chatDisabled'), fab = qs('fabChat');
    let chatOpen=false, chatPoll=null, lastChatId=0, myLevel=0, isLogged=false, myRole='member';

    function toggleChat(){ if(chatOpen) closeChat(); else openChat(); }
    qs('fabChat').onclick = toggleChat;

    function scrollChatToBottom(){ chatBody.scrollTop = chatBody.scrollHeight; }
    function renderMsg(m){
      const wrap = document.createElement('div'); wrap.className='msg';
      const head = document.createElement('div'); head.className='msghead';
      const name = document.createElement('span'); name.className='user-link'; name.textContent = m.username; name.dataset.uid = m.user_id;
      name.onclick = ()=>openUserProfile(m.user_id);
      const lvl = document.createElement('span'); lvl.className='level'; lvl.textContent = `Lv ${m.level}`;
      const badge = document.createElement('span'); badge.className='badge '+(m.role||'member'); badge.textContent = (m.role||'member').toUpperCase();
      const ts = document.createElement('span'); ts.className='time'; ts.textContent = new Date(m.created_at).toLocaleTimeString();
      head.appendChild(name); head.appendChild(lvl); head.appendChild(badge); head.appendChild(ts);
      const txt = document.createElement('div'); txt.className='txt'; txt.textContent = m.text;
      wrap.appendChild(head); wrap.appendChild(txt);
      chatBody.appendChild(wrap);
    }
    async function openUserProfile(uid){
      try{
        const p = await j('/api/profile/public?user_id='+encodeURIComponent(uid));
        const body = `
          <div style="display:flex; gap:12px; align-items:center; flex-wrap:wrap">
            <img src="${safeAvatar(p.avatar_url)}" class="avatar" style="width:56px; height:56px">
            <div><div class="big">${p.name} <span class="badge ${p.role}">${p.role.toUpperCase()}</span></div><div class="muted">ID: ${p.id}</div></div>
          </div>
          <div class="games-grid" style="grid-template-columns:1fr 1fr; margin-top:12px">
            <div class="card" style="padding:12px"><div class="label">Balance</div><div class="big">${fmtDL(p.balance)}</div></div>
            <div class="card" style="padding:12px"><div class="label">Level</div><div class="big">Lv ${p.level}</div><div class="muted">XP ${p.xp}</div></div>
          </div>
          <div class="games-grid" style="grid-template-columns:1fr 1fr; margin-top:12px">
            <div class="card" style="padding:12px"><div class="label">Crash games</div><div class="big">${p.crash_games}</div></div>
            <div class="card" style="padding:12px"><div class="label">Mines games</div><div class="big">${p.mines_games}</div></div>
          </div>`;
        qs('pmBody').innerHTML = body;
        qs('pmTitle').textContent = 'Profile — '+p.name;
        openPm();
      }catch(e){}
    }

    async function fetchChat(initial=false){
      try{
        const data = await j('/api/chat/fetch?since_id='+(initial?0:lastChatId));
        if(initial){ chatBody.innerHTML=''; }
        if(data.messages && data.messages.length){
          data.messages.forEach(m => { renderMsg(m); lastChatId = Math.max(lastChatId, m.id); });
          scrollChatToBottom();
        }
      }catch(e){}
    }
    async function updateChatGate(){
      try{ const me=await j('/api/me'); isLogged = true; myRole = me.role||'member'; }
      catch(e){ isLogged = false; myRole='member'; }
      if(isLogged){
        const prof = await j('/api/profile'); myLevel = prof.level || 1;
      }else{ myLevel = 0; }
      const canSend = isLogged && myLevel >= 5;
      chatText.disabled = !canSend; chatSend.disabled = !canSend;
      if(!isLogged){ chatDisabled.style.display=''; chatDisabled.textContent = 'Login with Discord to view and chat.'; }
      else if(myLevel < 5){ chatDisabled.style.display=''; chatDisabled.textContent = `Reach Level 5 to chat (your level: ${myLevel}). You can still read.`; }
      else { chatDisabled.style.display='none'; chatDisabled.textContent=''; }
      chatNote.textContent = isLogged ? `You are Lv ${myLevel}` : 'Not logged in';
    }
    async function openChat(){
      drawer.classList.add('open'); chatOpen=true;
      fab.style.right = 'calc(var(--chatW) + 18px)';
      await updateChatGate(); await fetchChat(true);
      if(chatPoll) clearInterval(chatPoll);
      chatPoll = setInterval(()=>{ if(chatOpen) fetchChat(false); }, 2000);
    }
    function closeChat(){ drawer.classList.remove('open'); chatOpen=false; if(chatPoll){clearInterval(chatPoll); chatPoll=null;} fab.style.right='18px'; }
    qs('chatClose').onclick = closeChat;

    chatSend.onclick = async ()=>{
      const t = chatText.value.trim(); if(!t) return;
      try{
        await j('/api/chat/send',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({text:t})});
        chatText.value=''; await fetchChat(false);
      }catch(e){ alert(e.message); }
    };

    // Header render + initial loads
    (async ()=>{
      await renderHeader();
      setInterval(renderHeader, 12000); // keep header fresh
      setInterval(refreshCrash, 1000);
      setInterval(pollNow, 350);
      await refreshCrash();
    })();
  </script>
</body>
</html>
"""

# ---------- Small HTML renderer ----------
def render_index_html():
    html = HTML_TEMPLATE.replace("__HOUSE_EDGE_MINES__", str(float(HOUSE_EDGE_MINES))).replace("__OWNER_ID__", str(OWNER_ID))
    return HTMLResponse(html)

# ---------- OAuth ----------
DISCORD_AUTH_BASE = "https://discord.com/oauth2/authorize"
DISCORD_TOKEN_URL = f"{DISCORD_API}/oauth2/token"
DISCORD_ME_URL = f"{DISCORD_API}/users/@me"

def oauth_login_url():
    params = {
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": OAUTH_REDIRECT,
        "scope": "identify",
        "prompt": "none",
    }
    return f"{DISCORD_AUTH_BASE}?{urlencode(params)}"

async def exchange_code_for_user(code: str) -> dict:
    async with httpx.AsyncClient(timeout=15.0) as cli:
        data = {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": OAUTH_REDIRECT,
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        tok = (await cli.post(DISCORD_TOKEN_URL, data=data, headers=headers)).json()
        if "access_token" not in tok:
            raise HTTPException(status_code=400, detail="OAuth failed")
        at = tok["access_token"]
        me = (await cli.get(DISCORD_ME_URL, headers={"Authorization": f"Bearer {at}"})).json()
        if "id" not in me: raise HTTPException(status_code=400, detail="OAuth failed")
        # Store only what we need
        return {
            "id": str(me["id"]),
            "username": f'{me.get("username","user")}#{me.get("discriminator","0")}' if "global_name" not in me else me.get("global_name") or me.get("username","user"),
            "avatar": me.get("avatar"),
        }

def avatar_from_session(s: dict) -> str:
    uid = s.get("id")
    ah = s.get("avatar")
    return avatar_url_from(uid, ah)

# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return render_index_html()

@app.get("/login")
async def login():
    return RedirectResponse(oauth_login_url())

@app.get("/oauth/callback")
async def oauth_callback(code: str = Query(...)):
    user = await exchange_code_for_user(code)
    ensure_profile_row(user["id"])
    resp = RedirectResponse(url="/")
    set_session(resp, user)
    return resp

@app.get("/logout")
async def logout():
    resp = RedirectResponse(url="/")
    resp.delete_cookie("session")
    return resp

def require_login(request: Request) -> dict:
    s = read_session(request)
    if not s: raise HTTPException(401, "Not logged in")
    return s

@app.get("/api/me")
async def api_me(request: Request):
    s = require_login(request)
    role = get_role(s["id"])
    return {"id": s["id"], "username": s["username"], "avatar_url": avatar_from_session(s), "role": role}

@app.get("/api/balance")
async def api_balance(request: Request):
    s = require_login(request)
    return {"balance": float(get_balance(s["id"]))}

@app.get("/api/profile")
async def api_profile(request: Request):
    s = require_login(request)
    return profile_info(s["id"])

@app.post("/api/profile/set_name")
async def api_set_name(request: Request):
    s = require_login(request)
    body = await request.json()
    name = (body.get("name") or "").strip()
    try:
        return set_profile_name(s["id"], name)
    except Exception as e:
        raise HTTPException(400, str(e))

@app.get("/api/profile/public")
async def api_public_profile(user_id: str):
    p = public_profile(user_id)
    if not p: raise HTTPException(404, "Not found")
    # include avatar url
    p["avatar_url"] = avatar_url_from(user_id, None)
    return p

@app.get("/api/referral/state")
async def api_ref_state(request: Request):
    s = require_login(request)
    nm = get_profile_name(s["id"])
    return {"name": nm}

# ---- Promo ----
@app.post("/api/promo/redeem")
async def api_promo_redeem(request: Request):
    s = require_login(request)
    code = (await request.json()).get("code","")
    try:
        bal = redeem_promo(s["id"], code)
        return {"ok": True, "balance": float(bal)}
    except PromoAlreadyRedeemed as e:
        raise HTTPException(400, str(e))
    except (PromoInvalid, PromoExpired, PromoExhausted) as e:
        raise HTTPException(400, str(e))

@with_conn
def my_redemptions(cur, user_id: str):
    cur.execute("SELECT code, redeemed_at FROM promo_redemptions WHERE user_id=%s ORDER BY redeemed_at DESC LIMIT 50", (user_id,))
    return [{"code": r[0], "redeemed_at": str(r[1])} for r in cur.fetchall()]

@app.get("/api/promo/my")
async def api_promo_my(request: Request):
    s = require_login(request)
    return {"rows": my_redemptions(s["id"])}

# ---- Admin endpoints ----
def require_role(request: Request, allowed=("owner","admin")) -> dict:
    s = require_login(request)
    role = get_role(s["id"])
    if role not in allowed and str(s["id"]) != str(OWNER_ID):
        raise HTTPException(403, "Not allowed")
    s["role"] = role
    return s

def parse_identifier(identifier: str) -> str:
    if not identifier: raise HTTPException(400, "Missing identifier")
    m = re.match(r"^<@!?(\d+)>$", identifier.strip())
    if m: return m.group(1)
    if identifier.strip().isdigit(): return identifier.strip()
    raise HTTPException(400, "Provide a user ID or <@mention>")

@app.post("/api/admin/adjust")
async def api_admin_adjust(request: Request):
    s = require_role(request, allowed=("owner","admin"))
    body = await request.json()
    target = parse_identifier(body.get("identifier",""))
    amount = body.get("amount","0")
    reason = body.get("reason")
    try:
        nb = adjust_balance(s["id"], target, amount, reason)
        return {"new_balance": float(nb)}
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/api/admin/promo/create")
async def api_admin_promo_create(request: Request):
    s = require_role(request, allowed=("owner","admin"))
    body = await request.json()
    code = body.get("code")
    amount = body.get("amount","0")
    max_uses = int(body.get("max_uses", 1))
    res = create_promo(s["id"], code, amount, max_uses)
    return res

@app.post("/api/admin/role")
async def api_admin_role(request: Request):
    s = require_role(request, allowed=("owner",))
    body = await request.json()
    target = parse_identifier(body.get("identifier",""))
    role = body.get("role","member")
    try:
        return set_role(target, role)
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/api/admin/timeout_site")
async def api_admin_timeout_site(request: Request):
    s = require_role(request, allowed=("owner","admin"))
    body = await request.json()
    target = parse_identifier(body.get("identifier",""))
    seconds = int(body.get("seconds", 600))
    reason = body.get("reason")
    return chat_timeout_set(s["id"], target, seconds, reason)

async def discord_timeout(uid: str, seconds: int, reason: Optional[str]):
    try:
        if not GUILD_ID: return
        guild = bot.get_guild(GUILD_ID)
        if not guild: return
        member = guild.get_member(int(uid))
        if not member: return
        until = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=seconds)
        await member.edit(communication_disabled_until=until, reason=reason or "Website timeout")
    except Exception:
        pass

@app.post("/api/admin/timeout_both")
async def api_admin_timeout_both(request: Request):
    s = require_role(request, allowed=("owner","admin"))
    body = await request.json()
    target = parse_identifier(body.get("identifier",""))
    seconds = int(body.get("seconds", 600))
    reason = body.get("reason")
    site = chat_timeout_set(s["id"], target, seconds, reason)
    # schedule discord timeout
    try:
        bot.loop.create_task(discord_timeout(target, seconds, reason))
    except Exception:
        pass
    return {"ok": True, "site": site}

# ---- Chat API ----
@app.get("/api/chat/fetch")
async def api_chat_fetch(since_id: int = 0):
    return {"messages": chat_fetch(since_id)}

@app.post("/api/chat/send")
async def api_chat_send(request: Request):
    s = require_login(request)
    body = await request.json()
    text = body.get("text","")
    try:
        res = chat_send(s["id"], s["username"], text)
        return res
    except PermissionError as e:
        raise HTTPException(403, str(e))
    except Exception as e:
        raise HTTPException(400, str(e))

# ---- Crash API ----
@app.get("/api/crash/state")
async def api_crash_state(request: Request):
    s = read_session(request)
    rid, info = ensure_betting_round()
    st = load_round()
    lb = last_busts()
    your = None
    if s:
        your = your_bet(st["id"], s["id"])
    phase = st["status"]
    out = {
        "round_id": st["id"],
        "phase": ("betting" if phase=="betting" else ("running" if phase=="running" else "ended")),
        "betting_opens_at": iso(st["betting_opens_at"]),
        "betting_ends_at": iso(st["betting_ends_at"]),
        "started_at": iso(st["started_at"]),
        "expected_end_at": iso(st["expected_end_at"]),
        "last_busts": lb,
        "your_bet": your
    }
    if st["status"] == "ended":
        out["bust"] = st["bust"]
    return out

@app.get("/api/crash/now")
async def api_crash_now():
    st = load_round()
    if not st or st["status"]!="running":
        return {"phase": (st["status"] if st else "betting")}
    m = current_multiplier(st["started_at"], st["expected_end_at"], st["bust"])
    return {"phase":"running","multiplier": float(m)}

@app.post("/api/crash/bet")
async def api_crash_bet(request: Request):
    s = require_login(request)
    body = await request.json()
    bet = q2(D(body.get("bet","0")))
    cashout = float(body.get("cashout", 1000))
    try:
        res = place_bet(s["id"], bet, cashout)
        return res
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/api/crash/cashout")
async def api_crash_cashout(request: Request):
    s = require_login(request)
    try:
        r = cashout_now(s["id"])
        return r
    except Exception as e:
        raise HTTPException(400, str(e))

@app.get("/api/crash/history")
async def api_crash_history(request: Request):
    s = require_login(request)
    return {"rows": your_history(s["id"])}

# ---- Mines API ----
@app.get("/api/mines/state")
async def api_mines_state(request: Request):
    s = require_login(request)
    st = mines_state(s["id"])
    return st or {}

@app.post("/api/mines/start")
async def api_mines_start(request: Request):
    s = require_login(request)
    body = await request.json()
    bet = q2(D(body.get("bet","0")))
    mines = int(body.get("mines",3))
    try:
        return mines_start(s["id"], bet, mines)
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/api/mines/pick")
async def api_mines_pick(request: Request):
    s = require_login(request)
    idx = int((await request.json()).get("index", -1))
    try:
        return mines_pick(s["id"], idx)
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/api/mines/cashout")
async def api_mines_cash(request: Request):
    s = require_login(request)
    try:
        return mines_cashout(s["id"])
    except Exception as e:
        raise HTTPException(400, str(e))

@app.get("/api/mines/history")
async def api_mines_hist(request: Request):
    s = require_login(request)
    return {"rows": mines_history(s["id"])}

# ---------- Crash round loop ----------
async def crash_loop():
    while True:
        try:
            st = load_round()
            if not st:
                ensure_betting_round()
            else:
                now = now_utc()
                if st["status"]=="betting":
                    if now >= st["betting_ends_at"]:
                        begin_running(st["id"])
                elif st["status"]=="running":
                    if now >= st["expected_end_at"]:
                        finish_round(st["id"])
                        create_next_betting()
            await asyncio.sleep(0.25)
        except Exception:
            await asyncio.sleep(1.0)

# ---------- Main ----------
async def run_bot_forever():
    if not BOT_TOKEN:
        print("WARN: DISCORD_TOKEN not set; bot disabled")
        while True: await asyncio.sleep(3600)
    await bot.start(BOT_TOKEN)

async def main():
    init_db()
    config = uvicorn.Config(app, host="0.0.0.0", port=PORT, log_level="info")
    server = uvicorn.Server(config)
    await asyncio.gather(server.serve(), run_bot_forever(), crash_loop())

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

