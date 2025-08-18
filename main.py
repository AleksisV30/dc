# ---------- DL Bank (Full App) — PART 1/2 ----------
# Imports, config, DB schema + helpers, game logic, Discord bot, HTML template

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
OWNER_ID = 1128658280546320426  # owner hard-coded
GUILD_ID = int(os.getenv("GUILD_ID", "0"))
DATABASE_URL = os.getenv("DATABASE_URL")

GEM = "💎"
MIN_BET = Decimal("1.00")
MAX_BET = Decimal("1000000.00")
BETTING_SECONDS = 10

# House edges
HOUSE_EDGE_CRASH = Decimal(os.getenv("HOUSE_EDGE_CRASH", "0.06"))
HOUSE_EDGE_MINES = Decimal(os.getenv("HOUSE_EDGE_MINES", "0.03"))

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
            role TEXT NOT NULL DEFAULT 'member',
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Make the column add no-op if it already exists (prevents transaction abort)
    cur.execute("ALTER TABLE IF EXISTS profiles ADD COLUMN IF NOT EXISTS role TEXT NOT NULL DEFAULT 'member'")

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

    # global chat (+ private)
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
    cur.execute("INSERT INTO balance_log(actor_id,target_id,amount,reason) VALUES (%s,%s,%s,%s)",
                (from_id, to_id, -amount, "tip"))
    cur.execute("INSERT INTO balance_log(actor_id,target_id,amount,reason) VALUES (%s,%s,%s,%s)",
                (from_id, to_id, amount, "tip"))
    return True

NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{3,20}$")

@with_conn
def ensure_profile_row(cur, user_id: str):
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
def chat_insert(cur, user_id: str, username: str, text: str, private_to: Optional[str] = None):
    text = (text or "").strip()
    if not text: raise ValueError("Message is empty")
    if len(text) > 300: raise ValueError("Message is too long (max 300)")
    ensure_profile_row(user_id)
    # level gate for PUBLIC only
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
    out = []
    if since_id <= 0:
        # latest public + latest private to user
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
                       f"**{PREFIX}tip <UserID> <amount>** — Send DL to another user"),
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

def parse_target_any(val: str) -> Optional[str]:
    if not val: return None
    m = re.match(r"^<@!?(\d+)>$", val)
    if m: return m.group(1)
    if val.isdigit(): return val
    return None

@bot.command(name="tip")
async def tip(ctx: commands.Context, target: str | None = None, amount: str | None = None):
    # NEW ORDER: .tip <userID> <amount>
    if target is None or amount is None:
        return await ctx.reply(embed=embed("Usage", f"`{PREFIX}tip <UserID> <amount>`", 0xF59E0B), mention_author=False)
    tid = parse_target_any(target)
    if not tid: return await ctx.reply(embed=embed("Invalid target","Provide a user ID or @mention.",0xEF4444), mention_author=False)
    if str(ctx.author.id) == tid:
        return await ctx.reply(embed=embed("Nope","You can’t tip yourself.",0xEF4444), mention_author=False)
    try:
        amt = q2(D(amount))
    except Exception:
        return await ctx.reply(embed=embed("Invalid amount","Use 1 or 1.24 etc.",0xEF4444), mention_author=False)
    if amt <= 0: return await ctx.reply(embed=embed("Invalid amount","Must be > 0.",0xEF4444), mention_author=False)
    try:
        with psycopg.connect(DATABASE_URL) as con, con.cursor() as cur:
            tip_transfer(cur, str(ctx.author.id), tid, amt)
            con.commit()
        await ctx.reply(embed=embed("Tip sent", f"{ctx.author.mention} ➜ <@{tid}> • {fmt_dl(amt)}", 0x34D399), mention_author=False)
    except Exception as e:
        await ctx.reply(embed=embed("Failed", str(e), 0xEF4444), mention_author=False)

# ---------- Frontend HTML ----------
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
      --input-bg:#0b1430; --input-br:#223457; --input-tx:#e6eeff; --input-ph:#9db4e4;
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

    /* inputs/buttons refresh */
    input, select, textarea{
      width:100%; appearance:none; background:var(--input-bg); color:var(--input-tx);
      border:1px solid var(--input-br); border-radius:12px; padding:10px 12px; outline:none;
      transition:border-color .15s ease, box-shadow .15s ease;
    }
    input::placeholder{ color:var(--input-ph) }
    input:focus{ border-color:#4c78ff; box-shadow:0 0 0 3px rgba(76,120,255,.18) }
    .field{ display:flex; flex-direction:column; gap:6px }
    .row{ display:grid; gap:10px }
    .row.cols-2{ grid-template-columns:1fr 1fr }
    .row.cols-3{ grid-template-columns:1fr 1fr 1fr }
    .row.cols-4{ grid-template-columns:1.6fr 1fr 1fr auto }
    .row.cols-5{ grid-template-columns:2fr 1fr 1fr auto auto }
    .card{
      background:linear-gradient(180deg,#0f1a33,#0b1326);
      border:1px solid var(--border); border-radius:16px; padding:16px
    }

    /* header */
    .header{position:sticky; top:0; z-index:30; backdrop-filter: blur(8px); background:rgba(10,15,30,.7); border-bottom:1px solid var(--border)}
    .header-inner{display:flex; align-items:center; justify-content:space-between; gap:10px; padding:10px 12px}
    .left{display:flex; align-items:center; gap:14px; flex:1; min-width:0}
    .brand{display:flex; align-items:center; gap:10px; font-weight:800; letter-spacing:.2px; white-space:nowrap}
    .brand .logo{width:28px;height:28px;border-radius:8px; background:linear-gradient(135deg,var(--accent),var(--accent2))}
    /* tabs */
    .tabs{ display:flex; gap:4px; align-items:center; padding:4px; border-radius:14px; background:linear-gradient(180deg,#0f1a33,#0b1326); border:1px solid var(--border); }
    .tab{ padding:8px 12px; border-radius:10px; cursor:pointer; font-weight:700; white-space:nowrap; color:#d8e6ff; opacity:.85; transition:all .15s ease; display:flex; align-items:center; gap:8px; }
    .tab:hover{opacity:1; transform:translateY(-1px)}
    .tab.active{ background:linear-gradient(135deg,#3b82f6,#22c1dc); color:#051326; box-shadow:0 6px 16px rgba(59,130,246,.25); opacity:1; }
    .right{display:flex; gap:8px; align-items:center; margin-left:12px}
    .chip{background:#0c1631; border:1px solid var(--border); color:#dce7ff; padding:6px 10px; border-radius:999px; font-size:12px; white-space:nowrap; cursor:pointer}
    .avatar{width:34px;height:34px;border-radius:50%;object-fit:cover;border:1px solid var(--border); cursor:pointer}
    .btn{display:inline-flex; align-items:center; gap:8px; padding:10px 14px; border-radius:12px; border:1px solid var(--border); background:linear-gradient(180deg,#0e1833,#0b1326); cursor:pointer; font-weight:600}
    .btn.primary{background:linear-gradient(135deg,#3b82f6,#22c1dc); border-color:transparent}
    .btn.ghost{ background:#162a52; border:1px solid var(--border); color:#eaf2ff; }
    .btn.cashout{ background: linear-gradient(135deg,#22c55e,#16a34a); border-color: transparent; box-shadow: 0 6px 14px rgba(34,197,94,.25); font-weight:800; }
    .btn.cashout[disabled]{ filter:grayscale(.5) brightness(.8); opacity:.8; cursor:not-allowed }

    .big{font-size:22px; font-weight:900}
    .label{font-size:12px; color:var(--muted); letter-spacing:.2px; text-transform:uppercase}
    .muted{color:var(--muted)}

    /* games grid */
    .games-grid{display:grid; gap:14px; grid-template-columns:1fr}
    @media(min-width:700px){.games-grid{grid-template-columns:1fr 1fr}}
    @media(min-width:1020px){.games-grid{grid-template-columns:1fr 1fr 1fr}}
    .game-card{ position:relative; min-height:130px; display:flex; flex-direction:column; justify-content:flex-end; gap:4px; background:linear-gradient(180deg,#0f1a33,#0c152a); border:1px solid var(--border); border-radius:16px; padding:16px; cursor:pointer; transition:transform .08s ease, box-shadow .12s ease, border-color .12s ease, background .18s ease; overflow:hidden; }
    .game-card:hover{transform:translateY(-2px); box-shadow:0 8px 18px rgba(0,0,0,.25)}
    .game-card .title{font-size:20px; font-weight:800}
    .ribbon{ position:absolute; top:12px; right:-32px; transform:rotate(35deg); background:linear-gradient(135deg,#f59e0b,#fb923c); color:#1a1206; font-weight:900; padding:6px 50px; border:1px solid rgba(0,0,0,.2); text-shadow:0 1px 0 rgba(255,255,255,.2); }

    /* Crash graph */
    .cr-graph-wrap{position:relative; height:240px; background:#0e1833; border:1px solid var(--border); border-radius:16px; overflow:hidden}
    canvas{display:block; width:100%; height:100%}
    .boom{ position:absolute; inset:0; pointer-events:none; opacity:0; }
    .boom.bang{ animation: bang .6s ease-out; }
    @keyframes bang{ 0%{ opacity:.95; background: radial-gradient(350px 350px at var(--x,50%) var(--y,50%), rgba(255,255,255,.9), rgba(239,68,68,.6) 40%, transparent 70%); } 100%{ opacity:0; background: radial-gradient(800px 800px at var(--x,50%) var(--y,50%), rgba(255,255,255,.0), rgba(239,68,68,.0) 40%, transparent 75%); } }

    /* Mines (unchanged styles from last drop) */
    .mines-two{ grid-template-columns: 360px 1fr !important; align-items: stretch; display:grid; gap:16px }
    .mines-wrap{ display:grid; place-items:center; height: calc(100vh - 180px); min-height: 420px; padding: 6px; }
    .mines-grid{
      --cell: clamp(48px, min( calc((100vw - 440px)/5), calc((100vh - 320px)/5) ), 110px);
      display:grid; gap:10px; grid-template-columns: repeat(5, var(--cell)); justify-content:center; align-content:center; padding: 6px; width: 100%;
    }
    .tile{ position:relative; width: var(--cell); aspect-ratio: 1/1; border-radius: clamp(10px, calc(var(--cell)*0.18), 16px); border:1px solid var(--border); background: radial-gradient(120% 120% at 30% 0%, #19264f 0%, #0c152a 55%), linear-gradient(180deg,#0f1936,#0c152a); display:flex; align-items:center; justify-content:center; font-weight:900; font-size: clamp(13px, calc(var(--cell)*0.34), 22px); cursor:pointer; user-select:none; transition:transform .09s ease, box-shadow .14s ease, background .18s ease, border-color .14s ease, opacity .14s ease; box-shadow: 0 8px 22px rgba(0,0,0,.35), inset 0 0 0 1px rgba(255,255,255,.03); overflow:hidden; }
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

    /* Modal */
    .modal{ position:fixed; inset:0; display:none; align-items:center; justify-content:center; background:rgba(3,6,12,.6); z-index:50; }
    .modal .box{ width:min(640px, 92vw); background:linear-gradient(180deg,#0f1a33,#0c1429); border:1px solid var(--border); border-radius:18px; padding:16px; box-shadow:0 10px 30px rgba(0,0,0,.4) }

    /* Chat Drawer + badges + menus */
    .chat-drawer{ position:fixed; right:0; top:64px; bottom:0; width:var(--chatW); max-width:90vw; transform:translateX(100%); transition: transform .2s ease-out; background:linear-gradient(180deg,#0f1a33,#0b1326); border-left:1px solid var(--border); display:flex; flex-direction:column; z-index:40; }
    .chat-drawer.open{ transform:translateX(0); }
    .chat-head{ display:flex; align-items:center; justify-content:space-between; padding:10px 12px; border-bottom:1px solid var(--border) }
    .chat-body{ flex:1; overflow:auto; padding:10px 12px; }
    .chat-input{ display:flex; gap:8px; padding:10px 12px; border-top:1px solid var(--border) }
    .chat-input input{ flex:1 }
    .msg{ margin-bottom:12px; padding-bottom:8px; border-bottom:1px dashed rgba(255,255,255,.04); position:relative }
    .msghead{ display:flex; gap:8px; align-items:center; flex-wrap:wrap }
    .msghead .time{ margin-left:auto; color:#...
""".replace("__HOUSE_EDGE_MINES__", str(float(HOUSE_EDGE_MINES))).replace("__OWNER_ID__", str(OWNER_ID))
# (template continues in this string; full closing tags are present)
# ---------- DL Bank (Full App) — PART 2/2 ----------
# FastAPI app + routes, OAuth/session helpers, background loop, lifespan, runner

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

# ---------- FastAPI app + static ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# we'll define lifespan below to avoid deprecated on_event hooks
app = FastAPI()

# serve /static (not strictly needed for this UI, but handy)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ---------- Signer + session cookies ----------
signer = URLSafeSerializer(SECRET_KEY, salt="session")

def set_session(resp: RedirectResponse, payload: dict):
    resp.set_cookie(
        "session",
        signer.dumps(payload),
        httponly=True,
        samesite="lax",
        max_age=7 * 24 * 3600,
    )

def read_session(request: Request) -> Optional[dict]:
    raw = request.cookies.get("session")
    if not raw:
        return None
    try:
        return signer.loads(raw)
    except BadSignature:
        return None

# ---------- OAuth state helpers ----------
class _State(BaseModel):
    t: int
    n: str

def _new_state() -> str:
    st = _State(t=int(datetime.datetime.now(UTC).timestamp()), n=secrets.token_hex(8)).model_dump()
    return signer.dumps(st, salt="state")

def _check_state(raw: str) -> bool:
    try:
        st = signer.loads(raw, salt="state")
        # 10 minute window
        t = int(st.get("t", 0))
        return (int(datetime.datetime.now(UTC).timestamp()) - t) < 600
    except Exception:
        return False

# keep a shared HTTP client
async_client: httpx.AsyncClient | None = None

def _require_session(request: Request) -> dict:
    sess = read_session(request)
    if not sess:
        raise HTTPException(status_code=401, detail="Not logged in")
    return sess

def _ensure_display_name(uid: str, username: str):
    # ensure a profile row with username as display_name if not set
    try:
        ensure_profile_row(uid)
        name = get_profile_name(uid)
        if not name:
            set_profile_name(uid, username)
    except Exception:
        pass

def _parse_identifier(s: str) -> str:
    s = (s or "").strip()
    m = re.match(r"^<@!?(\d+)>$", s)
    if m:
        return m.group(1)
    if s.isdigit():
        return s
    raise HTTPException(400, "Provide a Discord ID or @mention")

# ---------- Routes: HTML ----------
@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(HTML_TEMPLATE)

# ---------- Routes: OAuth ----------
@app.get("/login")
async def login():
    if not CLIENT_ID or not OAUTH_REDIRECT:
        raise HTTPException(500, "OAuth not configured")
    params = {
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": OAUTH_REDIRECT,
        "scope": "identify",
        "state": _new_state(),
        "prompt": "consent",
    }
    return RedirectResponse(f"{DISCORD_API}/oauth2/authorize?{urlencode(params)}")

@app.get("/callback")
async def callback(request: Request, code: str = Query(...), state: str = Query(...)):
    if not _check_state(state):
        raise HTTPException(400, "Invalid state")
    if not (CLIENT_ID and CLIENT_SECRET and OAUTH_REDIRECT):
        raise HTTPException(500, "OAuth not configured")
    global async_client
    if async_client is None:
        async_client = httpx.AsyncClient(timeout=15)
    try:
        # Exchange code for token
        token_r = await async_client.post(
            f"{DISCORD_API}/oauth2/token",
            data={
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": OAUTH_REDIRECT,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        token_r.raise_for_status()
        token = token_r.json()
        access = token["access_token"]

        # Fetch user
        me_r = await async_client.get(
            f"{DISCORD_API}/users/@me",
            headers={"Authorization": f"Bearer {access}"},
        )
        me_r.raise_for_status()
        me = me_r.json()
        uid = str(me["id"])
        username = (
            f'{me.get("username","user")}#{me.get("discriminator","0")}'
            if me.get("discriminator") not in (None, "0")
            else me.get("username", "user")
        )
        avatar_hash = me.get("avatar")
        avatar = avatar_url_from(uid, avatar_hash)

        # Ensure DB rows + display name
        ensure_profile_row(uid)
        _ensure_display_name(uid, username)

        # Set session + redirect
        resp = RedirectResponse("/")
        set_session(resp, {"id": uid, "username": username, "avatar_url": avatar})
        return resp
    except httpx.HTTPError as e:
        raise HTTPException(400, f"OAuth failed: {e}")

@app.get("/logout")
async def logout():
    resp = RedirectResponse("/")
    resp.delete_cookie("session")
    return resp

# ---------- Routes: Identity/Profile/Balance ----------
@app.get("/api/me")
async def api_me(request: Request):
    sess = _require_session(request)
    uid = sess["id"]
    role = get_role(uid)
    return {
        "id": uid,
        "username": sess.get("username", "user"),
        "avatar_url": sess.get("avatar_url"),
        "role": role,
    }

@app.get("/api/balance")
async def api_balance(request: Request):
    sess = _require_session(request)
    bal = get_balance(sess["id"])
    return {"balance": float(bal)}

@app.get("/api/profile")
async def api_profile(request: Request):
    sess = _require_session(request)
    data = profile_info(sess["id"])
    data["id"] = sess["id"]
    return data

class NameIn(BaseModel):
    name: str

@app.post("/api/profile/set_name")
async def api_set_name(request: Request, body: NameIn):
    sess = _require_session(request)
    return set_profile_name(sess["id"], body.name)

@app.get("/api/public_profile")
async def api_public_profile(user_id: str = Query(...)):
    p = public_profile(user_id)
    if not p:
        raise HTTPException(404, "Not found")
    return p

@app.get("/api/referral/state")
async def api_ref_state(request: Request):
    sess = _require_session(request)
    return {"name": get_profile_name(sess["id"])}

# ---------- Routes: Promo ----------
class PromoIn(BaseModel):
    code: str

@app.post("/api/promo/redeem")
async def api_promo_redeem(request: Request, body: PromoIn):
    sess = _require_session(request)
    try:
        new_bal = redeem_promo(sess["id"], body.code)
        return {"ok": True, "new_balance": float(new_bal)}
    except PromoAlreadyRedeemed as e:
        raise HTTPException(400, str(e))
    except PromoInvalid as e:
        raise HTTPException(400, str(e))
    except PromoExpired as e:
        raise HTTPException(400, str(e))
    except PromoExhausted as e:
        raise HTTPException(400, str(e))

@app.get("/api/promo/my")
async def api_promo_my(request: Request):
    sess = _require_session(request)

    @with_conn
    def _rows(cur):
        cur.execute(
            "SELECT code, redeemed_at FROM promo_redemptions WHERE user_id=%s ORDER BY redeemed_at DESC",
            (sess["id"],),
        )
        return [{"code": r[0], "redeemed_at": str(r[1])} for r in cur.fetchall()]

    return {"rows": _rows()}

# ---------- Routes: Admin ----------
class AdjustIn(BaseModel):
    identifier: str
    amount: str
    reason: str | None = None

def _must_owner(request: Request) -> str:
    uid = _require_session(request)["id"]
    role = get_role(uid)
    if role != "owner" and str(uid) != str(OWNER_ID):
        raise HTTPException(403, "Owner only")
    return uid

def _must_admin_or_owner(request: Request) -> str:
    uid = _require_session(request)["id"]
    role = get_role(uid)
    if role not in ("owner", "admin") and str(uid) != str(OWNER_ID):
        raise HTTPException(403, "Admin only")
    return uid

@app.post("/api/admin/adjust")
async def api_admin_adjust(request: Request, body: AdjustIn):
    actor = _must_owner(request)
    tid = _parse_identifier(body.identifier)
    try:
        new_bal = adjust_balance(actor, tid, body.amount, body.reason or "admin adjust")
        return {"ok": True, "new_balance": float(new_bal)}
    except Exception as e:
        raise HTTPException(400, str(e))

class RoleIn(BaseModel):
    identifier: str
    role: str

@app.post("/api/admin/role")
async def api_admin_role(request: Request, body: RoleIn):
    _must_owner(request)
    tid = _parse_identifier(body.identifier)
    return set_role(tid, body.role.lower())

class TimeoutIn(BaseModel):
    identifier: str
    seconds: int
    reason: str | None = None

@app.post("/api/admin/timeout_site")
async def api_admin_timeout_site(request: Request, body: TimeoutIn):
    actor = _must_admin_or_owner(request)
    tid = _parse_identifier(body.identifier)
    return chat_timeout_set(
        actor, tid, max(1, int(body.seconds)), body.reason or "moderation"
    )

async def _timeout_discord(uid: int, seconds: int, reason: str | None = None):
    try:
        if not GUILD_ID:
            return False
        guild = bot.get_guild(GUILD_ID)
        if guild is None:
            guild = await bot.fetch_guild(GUILD_ID)
        member = guild.get_member(uid)
        if member is None:
            member = await guild.fetch_member(uid)
        until = discord.utils.utcnow() + datetime.timedelta(seconds=seconds)
        await member.edit(
            communication_disabled_until=until, reason=reason or "moderation"
        )
        return True
    except Exception:
        return False

@app.post("/api/admin/timeout_both")
async def api_admin_timeout_both(request: Request, body: TimeoutIn):
    actor = _must_admin_or_owner(request)
    tid = _parse_identifier(body.identifier)
    out = chat_timeout_set(
        actor, tid, max(1, int(body.seconds)), body.reason or "moderation"
    )
    # fire-and-forget Discord timeout
    try:
        asyncio.create_task(_timeout_discord(int(tid), int(body.seconds), body.reason))
    except Exception:
        pass
    return out

class PromoCreateIn(BaseModel):
    code: str | None = None
    amount: str
    max_uses: int = 1
    expires_at: str | None = None

@app.post("/api/admin/promo/create")
async def api_admin_promo_create(request: Request, body: PromoCreateIn):
    actor = _must_owner(request)
    try:
        return create_promo(
            actor, body.code, body.amount, body.max_uses, body.expires_at
        )
    except Exception as e:
        raise HTTPException(400, str(e))

# ---------- Routes: Chat ----------
class ChatIn(BaseModel):
    text: str
    private_to: str | None = None

@app.get("/api/chat/fetch")
async def api_chat_fetch(
    request: Request, since: int = Query(0), limit: int = Query(60)
):
    user_id = None
    sess = read_session(request)
    if sess:
        user_id = sess["id"]
    rows = chat_fetch(since, min(120, max(1, limit)), user_id)
    return {"rows": rows}

@app.post("/api/chat/send")
async def api_chat_send(request: Request, body: ChatIn):
    sess = _require_session(request)
    uid = sess["id"]
    uname = sess.get("username") or f"user_{uid[-4:]}"
    txt = (body.text or "").strip()

    # Inline command: .tip <uid> <amount>
    if txt.startswith(".tip "):
        try:
            _, rest = txt.split(".tip", 1)
            parts = rest.strip().split()
            if len(parts) != 2:
                raise ValueError("Usage: .tip <UserID> <amount>")
            tid = _parse_identifier(parts[0])
            amt = q2(D(parts[1]))
            with psycopg.connect(DATABASE_URL) as con, con.cursor() as cur:
                tip_transfer(cur, uid, tid, amt)
                con.commit()
            # Announce
            chat_insert(uid, uname, f"Sent {fmt_dl(amt)} to <@{tid}>", None)
            return {"ok": True, "message": "Tip sent"}
        except Exception as e:
            raise HTTPException(400, str(e))

    try:
        row = chat_insert(uid, uname, txt, body.private_to)
        return {"ok": True, "id": row["id"], "created_at": row["created_at"]}
    except PermissionError as e:
        raise HTTPException(403, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.post("/api/chat/delete")
async def api_chat_delete(request: Request, id: int = Query(...)):
    _must_admin_or_owner(request)
    return chat_delete(id)

@app.get("/api/chat/timeout")
async def api_chat_timeout(request: Request):
    sess = _require_session(request)
    t = chat_timeout_get(sess["id"])
    return t or {"ok": True, "seconds_left": 0}

# ---------- Routes: Crash ----------
@app.get("/api/crash/state")
async def api_crash_state(request: Request):
    sess = read_session(request)
    uid = sess["id"] if sess else None
    rid, rd = ensure_betting_round()
    out = {
        "round_id": rid,
        "phase": rd["status"],
        "betting_opens_at": iso(rd["betting_opens_at"]),
        "betting_ends_at": iso(rd["betting_ends_at"]),
        "started_at": iso(rd["started_at"]),
        "expected_end_at": iso(rd["expected_end_at"]),
        "bust": rd["bust"],
        "last_busts": last_busts(),
    }
    if (
        rd["status"] == "running"
        and rd["started_at"]
        and rd["expected_end_at"]
        and rd["bust"]
    ):
        out["current_multiplier"] = current_multiplier(
            rd["started_at"], rd["expected_end_at"], rd["bust"]
        )
    if uid:
        yr = your_bet(rid, uid)
        if yr:
            out["your_bet"] = yr
    return out

class CrashPlaceIn(BaseModel):
    bet: str
    cashout: float | None = None

@app.post("/api/crash/place")
async def api_crash_place(request: Request, body: CrashPlaceIn):
    sess = _require_session(request)
    try:
        bet = q2(D(body.bet))
        co = float(body.cashout or 0) or 2.0
        if co < 1.01:
            raise ValueError("Cashout must be ≥ 1.01×")
        return place_bet(sess["id"], bet, co)
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/api/crash/cashout")
async def api_crash_cashout(request: Request):
    sess = _require_session(request)
    try:
        return cashout_now(sess["id"])
    except Exception as e:
        raise HTTPException(400, str(e))

@app.get("/api/crash/history")
async def api_crash_history(request: Request):
    sess = _require_session(request)
    return {"rows": your_history(sess["id"], 10)}

# ---------- Routes: Mines ----------
class MinesStartIn(BaseModel):
    bet: str
    mines: int

@app.post("/api/mines/start")
async def api_mines_start(request: Request, body: MinesStartIn):
    sess = _require_session(request)
    try:
        return mines_start(sess["id"], q2(D(body.bet)), int(body.mines))
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/api/mines/pick")
async def api_mines_pick(request: Request, index: int = Query(...)):
    sess = _require_session(request)
    try:
        return mines_pick(sess["id"], int(index))
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/api/mines/cashout")
async def api_mines_cashout(request: Request):
    sess = _require_session(request)
    try:
        return mines_cashout(sess["id"])
    except Exception as e:
        raise HTTPException(400, str(e))

@app.get("/api/mines/state")
async def api_mines_state(request: Request):
    sess = _require_session(request)
    s = mines_state(sess["id"])
    return s or {}

@app.get("/api/mines/history")
async def api_mines_history(request: Request):
    sess = _require_session(request)
    return {"rows": mines_history(sess["id"], 15)}

# ---------- Background: Crash engine loop ----------
async def crash_engine_loop():
    await asyncio.sleep(0.5)
    while True:
        try:
            rid, rd = ensure_betting_round()
            now = now_utc()
            st = rd["status"]
            if st == "betting":
                if now >= rd["betting_ends_at"]:
                    begin_running(rid)
            elif st == "running":
                if now >= rd["expected_end_at"]:
                    finish_round(rid)
                    create_next_betting()
        except Exception:
            # swallow to keep loop alive (you can add logging)
            pass
        await asyncio.sleep(0.5)

# ---------- Lifespan (startup/shutdown) ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    init_db()
    global async_client
    if async_client is None:
        async_client = httpx.AsyncClient(timeout=15)

    # start crash engine
    crash_task = asyncio.create_task(crash_engine_loop())

    # start Discord bot if configured (non-blocking)
    bot_task = None
    if BOT_TOKEN:
        bot_task = asyncio.create_task(bot.start(BOT_TOKEN))

    try:
        yield
    finally:
        # shutdown
        if async_client:
            try:
                await async_client.aclose()
            except Exception:
                pass

        # close discord bot
        try:
            if bot and bot.is_ready():
                await bot.close()
        except Exception:
            pass

        # cancel background task
        try:
            crash_task.cancel()
        except Exception:
            pass
        if bot_task:
            try:
                bot_task.cancel()
            except Exception:
                pass

# rebind app with lifespan (to avoid on_event deprecation)
app.router.lifespan_context = lifespan  # type: ignore[attr-defined]

# ---------- Main ----------
if __name__ == "__main__":
    # local dev run
    init_db()
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
