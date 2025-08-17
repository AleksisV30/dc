import os, json, asyncio, re, random, string, math, secrets, datetime
from urllib.parse import urlencode
from typing import Optional, Tuple, Dict

import httpx
import psycopg
import discord
from discord.ext import commands
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from itsdangerous import URLSafeSerializer, BadSignature
import uvicorn
from pydantic import BaseModel

# ---------- Config ----------
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
HOUSE_EDGE = 0.01      # 1% house edge
MIN_BET = 1            # DL
MAX_BET = 1_000_000
BETTING_SECONDS = 10   # time to place bets between rounds

# ---------- Time helpers ----------
UTC = datetime.timezone.utc
def now_utc() -> datetime.datetime: return datetime.datetime.now(UTC)
def iso(dt: Optional[datetime.datetime]) -> Optional[str]:
    if dt is None: return None
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
            balance INTEGER NOT NULL DEFAULT 0
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS balance_log (
            id BIGSERIAL PRIMARY KEY,
            actor_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            amount INTEGER NOT NULL,
            reason TEXT,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # profiles / levels
    cur.execute("""
        CREATE TABLE IF NOT EXISTS profiles (
            user_id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            name_lower TEXT NOT NULL UNIQUE,
            xp INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("ALTER TABLE profiles ADD COLUMN IF NOT EXISTS xp INTEGER NOT NULL DEFAULT 0")

    # promos
    cur.execute("""
        CREATE TABLE IF NOT EXISTS promo_codes (
            code TEXT PRIMARY KEY,
            amount INTEGER NOT NULL,
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

    # crash (multiplayer)
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
            bet INTEGER NOT NULL,
            cashout NUMERIC(8,2) NOT NULL, -- auto cashout goal
            cashed_out NUMERIC(8,2),       -- live cashout multiplier (optional)
            cashed_out_at TIMESTAMPTZ,
            win INTEGER NOT NULL DEFAULT 0,
            resolved BOOLEAN NOT NULL DEFAULT FALSE,
            PRIMARY KEY(round_id, user_id)
        )
    """)
    # ensure new columns exist if migrating
    cur.execute("ALTER TABLE crash_bets ADD COLUMN IF NOT EXISTS cashed_out NUMERIC(8,2)")
    cur.execute("ALTER TABLE crash_bets ADD COLUMN IF NOT EXISTS cashed_out_at TIMESTAMPTZ")
    cur.execute("ALTER TABLE crash_bets ADD COLUMN IF NOT EXISTS resolved BOOLEAN NOT NULL DEFAULT FALSE")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS crash_games (
            id BIGSERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            bet INTEGER NOT NULL,
            cashout NUMERIC(8,2) NOT NULL,
            bust NUMERIC(10,2) NOT NULL,
            win INTEGER NOT NULL,
            xp_gain INTEGER NOT NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
    """)

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

# ---- balances / profiles ----
@with_conn
def get_balance(cur, user_id: str) -> int:
    cur.execute("SELECT balance FROM balances WHERE user_id = %s", (user_id,))
    row = cur.fetchone(); return int(row[0]) if row else 0

@with_conn
def adjust_balance(cur, actor_id: str, target_id: str, amount: int, reason: Optional[str]) -> int:
    cur.execute("INSERT INTO balances (user_id, balance) VALUES (%s, 0) ON CONFLICT (user_id) DO NOTHING", (target_id,))
    cur.execute("UPDATE balances SET balance = balance + %s WHERE user_id = %s", (amount, target_id))
    cur.execute("INSERT INTO balance_log(actor_id, target_id, amount, reason) VALUES (%s, %s, %s, %s)",
                (actor_id, target_id, amount, reason))
    cur.execute("SELECT balance FROM balances WHERE user_id = %s", (target_id,))
    return int(cur.fetchone()[0])

NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{3,20}$")

@with_conn
def ensure_profile_row(cur, user_id: str):
    cur.execute("INSERT INTO profiles(user_id, display_name, name_lower) VALUES (%s,%s,%s) ON CONFLICT (user_id) DO NOTHING",
                (user_id, f"user_{user_id[-4:]}", f"user_{user_id[-4:]}"))

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
    cur.execute("SELECT xp FROM profiles WHERE user_id=%s", (user_id,))
    xp = int(cur.fetchone()[0])
    level = 1 + xp // 100
    base = (level - 1) * 100; need = level * 100 - base
    progress = xp - base; pct = 0 if need==0 else int(progress*100/need)
    bal = get_balance(user_id)
    return {"xp": xp, "level": level, "progress": progress, "next_needed": need, "progress_pct": pct, "balance": bal}

# ---- promos ----
class PromoError(Exception): ...
class PromoAlreadyRedeemed(PromoError): ...
class PromoInvalid(PromoError): ...
class PromoExpired(PromoError): ...
class PromoExhausted(PromoError): ...

@with_conn
def redeem_promo(cur, user_id: str, code: str) -> int:
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
    return int(cur.fetchone()[0])

def _rand_code(n=8): return ''.join(random.choices(string.ascii_uppercase+string.digits, k=n))

@with_conn
def create_promo(cur, actor_id: str, code: Optional[str], amount: int, max_uses: int = 1, expires_at: Optional[str] = None):
    code = (code.strip().upper() if code else _rand_code())
    cur.execute("""
        INSERT INTO promo_codes(code,amount,max_uses,expires_at,created_by)
        VALUES (%s,%s,%s,%s,%s)
        ON CONFLICT (code) DO UPDATE SET amount=EXCLUDED.amount, max_uses=EXCLUDED.max_uses, expires_at=EXCLUDED.expires_at
    """, (code, amount, max_uses, expires_at, actor_id))
    return {"ok": True, "code": code}

# ---- Crash math & DB (multiplayer) ----
def _u(): return (secrets.randbelow(1_000_000_000)+1)/1_000_000_001.0
def gen_bust(edge: float = HOUSE_EDGE) -> float:
    # P(B>=x) = (1-edge)/x
    u = _u(); B = max(1.0, (1.0-edge)/u)
    return math.floor(B*100)/100.0

def run_duration_for(bust: float) -> float:
    # ~1.2s .. 7s depending on bust
    return min(7.0, 1.2 + math.log(bust+1.0)*1.6)

def current_multiplier(started_at: datetime.datetime, expected_end_at: datetime.datetime, bust: float, at: Optional[datetime.datetime] = None) -> float:
    """Deterministic server multiplier based on time.
       m(t) grows from 1.00 at start to bust at expected_end."""
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
def place_bet(cur, user_id: str, bet: int, cashout: float):
    # must be during betting
    cur.execute("""SELECT id, betting_ends_at FROM crash_rounds
                   WHERE status='betting'
                   ORDER BY id DESC LIMIT 1""")
    row = cur.fetchone()
    if not row: raise ValueError("Betting is closed")
    round_id, ends_at = int(row[0]), row[1]
    cur.execute("SELECT NOW() < %s", (ends_at,))
    if not cur.fetchone()[0]:
        raise ValueError("Betting just closed")

    # balance & debit
    cur.execute("INSERT INTO balances(user_id,balance) VALUES (%s,0) ON CONFLICT(user_id) DO NOTHING", (user_id,))
    cur.execute("SELECT balance FROM balances WHERE user_id=%s FOR UPDATE", (user_id,))
    bal = int(cur.fetchone()[0])
    if bet < MIN_BET: raise ValueError(f"Min bet is {MIN_BET} DL")
    if bet > MAX_BET: raise ValueError(f"Max bet is {MAX_BET} DL")
    if bal < bet: raise ValueError("Insufficient balance")
    cur.execute("UPDATE balances SET balance=balance-%s WHERE user_id=%s", (bet, user_id))

    # one bet per user per round
    try:
        cur.execute("""INSERT INTO crash_bets(round_id,user_id,bet,cashout)
                       VALUES(%s,%s,%s,%s)""",
                    (round_id, user_id, bet, float(cashout)))
    except Exception:
        cur.execute("UPDATE balances SET balance=balance+%s WHERE user_id=%s", (bet, user_id))
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

    bust = gen_bust(HOUSE_EDGE)
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
    # cash out current running round for this user (if possible)
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
    bet, cash_goal, cashed_out, resolved = int(b[0]), float(b[1]), b[2], bool(b[3])
    if resolved or cashed_out is not None:
        raise ValueError("Already cashed out")

    m = current_multiplier(started_at, exp_end, bust, now)
    if m >= bust: raise ValueError("Too late — crashed")
    # Credit immediately
    win = int(math.floor(bet * m + 1e-9))
    cur.execute("UPDATE balances SET balance=balance+%s WHERE user_id=%s", (win, user_id))
    cur.execute("""UPDATE crash_bets
                   SET cashed_out=%s, cashed_out_at=%s, win=%s, resolved=TRUE
                   WHERE round_id=%s AND user_id=%s""",
                (float(m), now, int(win), rid, user_id))
    return {"round_id": rid, "multiplier": m, "win": win}

@with_conn
def resolve_round_end(cur, round_id: int, bust: float):
    # For all bets, finalize results and write history; avoid double-crediting those who already cashed out
    cur.execute("""SELECT user_id, bet, cashout, cashed_out, resolved, win
                   FROM crash_bets WHERE round_id=%s""", (round_id,))
    bets = cur.fetchall()
    for uid, bet, goal, cashed, resolved, win in bets:
        uid = str(uid); bet=int(bet); goal=float(goal); win=int(win); resolved=bool(resolved)
        # Compute final state:
        if resolved and cashed is not None:
            # already paid at live cashout; just record history + xp
            xp_gain = max(1, min(bet, 50))
            cur.execute("""INSERT INTO crash_games(user_id,bet,cashout,bust,win,xp_gain)
                           VALUES(%s,%s,%s,%s,%s,%s)""",
                        (uid, bet, float(cashed), float(bust), win, xp_gain))
            ensure_profile_row(uid)
            cur.execute("UPDATE profiles SET xp=xp+%s WHERE user_id=%s", (xp_gain, uid))
            continue

        if not resolved:
            if goal <= bust:
                # auto cashout at goal
                win = int(math.floor(bet * goal + 1e-9))
                cur.execute("UPDATE balances SET balance=balance+%s WHERE user_id=%s", (win, uid))
                cur.execute("""UPDATE crash_bets SET win=%s, resolved=TRUE WHERE round_id=%s AND user_id=%s""",
                            (int(win), round_id, uid))
                cashed_val = goal
            else:
                # lost
                cur.execute("""UPDATE crash_bets SET resolved=TRUE WHERE round_id=%s AND user_id=%s""",
                            (round_id, uid))
                win = 0
                cashed_val = goal  # record desired goal

            xp_gain = max(1, min(bet, 50))
            cur.execute("""INSERT INTO crash_games(user_id,bet,cashout,bust,win,xp_gain)
                           VALUES(%s,%s,%s,%s,%s,%s)""",
                        (uid, bet, float(cashed_val), float(bust), int(win), xp_gain))
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
    return {"bet": int(r[0]), "cashout": float(r[1]),
            "cashed_out": (float(r[2]) if r[2] is not None else None),
            "resolved": bool(r[3]), "win": int(r[4])}

@with_conn
def your_history(cur, user_id: str, limit: int = 10):
    cur.execute("""SELECT bet, cashout, bust, win, xp_gain, created_at
                   FROM crash_games WHERE user_id=%s
                   ORDER BY id DESC LIMIT %s""", (user_id, limit))
    return [{"bet":int(r[0]),"cashout":float(r[1]),"bust":float(r[2]),"win":int(r[3]),"xp_gain":int(r[4]),"created_at":str(r[5])} for r in cur.fetchall()]

# ---------- Global Chat ----------
@with_conn
def chat_send(cur, user_id: str, username: str, text: str):
    text = (text or "").strip()
    if not text: raise ValueError("Message is empty")
    if len(text) > 300: raise ValueError("Message is too long (max 300)")
    ensure_profile_row(user_id)
    cur.execute("SELECT xp FROM profiles WHERE user_id=%s", (user_id,))
    xp = int(cur.fetchone()[0])
    lvl = 1 + xp // 100
    if lvl < 5: raise PermissionError("You must be level 5 to chat")
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
    if uids:
        cur.execute("SELECT user_id, xp FROM profiles WHERE user_id = ANY(%s)", (uids,))
        for uid, xp in cur.fetchall():
            levels[str(uid)] = 1 + int(xp) // 100
    out = []
    for mid, uid, uname, txt, ts in rows:
        lvl = levels.get(str(uid), 1)
        out.append({"id": int(mid), "user_id": str(uid), "username": uname, "level": int(lvl), "text": txt, "created_at": str(ts)})
    return out

# ---------- Discord bot ----------
def fmt_dl(n: int) -> str: return f"{GEM} {n:,} DL"
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
                       f"**{PREFIX}bal @User** — Show **someone else’s** balance"),
                inline=False)
    owner_line = f"**{PREFIX}addbal @User <amount>** — Add/subtract DL *(owner only)*"
    if is_owner: owner_line += " ✅"
    e.add_field(name="Admin", value=owner_line, inline=False)
    e.set_footer(text="Currency: Growtopia Diamond Locks (DL)")
    await ctx.reply(embed=e, mention_author=False)

@with_conn
def get_user_balance(cur, uid: str) -> int:
    cur.execute("SELECT balance FROM balances WHERE user_id=%s", (uid,))
    r = cur.fetchone(); return int(r[0]) if r else 0

@bot.command(name="bal")
async def bal(ctx: commands.Context, user: discord.User | None = None):
    target = user or ctx.author
    bal_value = get_user_balance(str(target.id))
    e = embed(title="Balance", desc=f"{target.mention}\n**{fmt_dl(bal_value)}**", color=0x34D399)
    e.set_footer(text="Currency: Growtopia Diamond Locks (DL)")
    await ctx.reply(embed=e, mention_author=False)

@bot.command(name="addbal")
async def addbal(ctx: commands.Context, user: discord.User | None = None, amount: int | None = None):
    if ctx.author.id != OWNER_ID:
        return await ctx.reply(embed=embed("Not allowed","Only the owner can adjust balances.",0xEF4444), mention_author=False)
    if user is None or amount is None:
        return await ctx.reply(embed=embed("Usage", f"`{PREFIX}addbal @User <amount>`", 0xF59E0B), mention_author=False)
    if amount == 0:
        return await ctx.reply(embed=embed("Invalid amount","Amount cannot be zero.",0xEF4444), mention_author=False)
    new_balance = adjust_balance(str(ctx.author.id), str(user.id), amount, reason="bot addbal")
    sign = "+" if amount > 0 else ""
    e = embed("Balance Updated", f"**Target:** {user.mention}\n**Change:** `{sign}{amount}` → {fmt_dl(new_balance)}", 0x60A5FA)
    e.set_footer(text="Currency: Growtopia Diamond Locks (DL)")
    await ctx.reply(embed=e, mention_author=False)

# ---------- Web (FastAPI + Frontend) ----------
app = FastAPI()
signer = URLSafeSerializer(SECRET_KEY, salt="session")

def set_session(resp: RedirectResponse, payload: dict):
    resp.set_cookie("session", signer.dumps(payload), httponly=True, samesite="lax", max_age=7*24*3600)
def read_session(request: Request) -> Optional[dict]:
    raw = request.cookies.get("session")
    if not raw: return None
    try: return signer.loads(raw)
    except BadSignature: return None

# ---------- Frontend HTML (plain string) ----------
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
    .header{position:sticky; top:0; z-index:20; backdrop-filter: blur(8px); background:rgba(10,15,30,.7); border-bottom:1px solid var(--border)}
    .header-inner{display:flex; align-items:center; justify-content:space-between; gap:10px; padding:10px 12px}
    .brand{display:flex; align-items:center; gap:10px; font-weight:800; letter-spacing:.2px}
    .brand .logo{width:28px;height:28px;border-radius:8px; background:linear-gradient(135deg,var(--accent),var(--accent2))}
    .tabs{display:flex; gap:8px; align-items:center; overflow:auto; -webkit-overflow-scrolling:touch}
    .tab{padding:8px 12px; border:1px solid var(--border); border-radius:12px; background:linear-gradient(180deg,#0e1833,#0b1326); cursor:pointer; font-weight:600; white-space:nowrap}
    .tab.active{background:linear-gradient(135deg,#3b82f6,#22c1dc); border-color:transparent}
    .right{display:flex; gap:8px; align-items:center}
    .chip{background:#0c1631; border:1px solid var(--border); color:#dce7ff; padding:6px 10px; border-radius:999px; font-size:12px; white-space:nowrap; cursor:pointer}
    .avatar{width:34px;height:34px;border-radius:50%;object-fit:cover;border:1px solid var(--border); cursor:pointer}
    .btn{display:inline-flex; align-items:center; gap:8px; padding:10px 14px; border-radius:12px; border:1px solid var(--border); background:linear-gradient(180deg,#0e1833,#0b1326); cursor:pointer; font-weight:600}
    .btn.primary{background:linear-gradient(135deg,#3b82f6,#22c1dc); border-color:transparent}
    .btn.ghost{background:transparent}
    .grid{display:grid; gap:16px; grid-template-columns:1fr}
    @media(min-width:980px){.grid{grid-template-columns:1fr 1fr}}
    .card{background:linear-gradient(180deg,var(--bg2),#0b1428); border:1px solid var(--border); border-radius:18px; padding:16px; box-shadow: 0 6px 20px rgba(0,0,0,.25)}
    .label{color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.1em}
    .big{font-size:28px; font-weight:800}
    .muted{color:var(--muted)}
    input, .input{
      width:100%; background:#0e1833; color:#e8efff; border:1px solid var(--border); border-radius:12px; padding:12px; outline:none
    }
    input:focus{border-color:#356adf; box-shadow:0 0 0 2px rgba(53,106,223,.25)}
    .game-card{display:flex; flex-direction:column; gap:4px; background:linear-gradient(180deg,#0f1a33,#0c152a); border:1px solid var(--border); border-radius:16px; padding:16px; cursor:pointer; transition:transform .08s ease, box-shadow .12s ease}
    .game-card:hover{transform:translateY(-2px); box-shadow:0 8px 18px rgba(0,0,0,.25)}
    .owner{margin-top:16px; border-top:1px dashed var(--border); padding-top:12px}

    /* Crash page */
    .crash-wrap{display:flex; flex-direction:column; gap:12px}
    .cr-top{display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px}
    .cr-metric{display:flex; align-items:baseline; gap:10px}
    .cr-multi{font-size:34px; font-weight:900}
    .cr-small{font-size:12px; color:var(--muted)}
    .chip2{display:inline-block; padding:6px 8px; background:#0d1a36; color:#cfe6ff; border:1px solid var(--border); border-radius:999px; font-size:12px; margin:2px}
    .bust-bad{color:#ef6a6a}
    .bust-good{color:#4bd3a8}

    /* Vertical graph */
    .cr-graph-wrap{position:relative; height:240px; background:#0e1833; border:1px solid var(--border); border-radius:16px; overflow:hidden}
    canvas{display:block; width:100%; height:100%}

    /* Modal */
    .modal{position:fixed; inset:0; background:rgba(0,0,0,.6); display:none; align-items:center; justify-content:center; padding:20px; z-index:30}
    .modal .box{background:#0f1a33; border:1px solid var(--border); border-radius:16px; padding:16px; max-width:520px; width:100%}

    /* Chat drawer */
    .chat-drawer{position:fixed; top:64px; right:0; width:360px; max-width:100vw; bottom:0; background:linear-gradient(180deg,#0e1936,#0c152a); border-left:1px solid var(--border); transform:translateX(100%); transition:transform .18s ease; z-index:25; display:flex; flex-direction:column}
    .chat-drawer.open{transform:translateX(0)}
    .chat-head{display:flex; align-items:center; justify-content:space-between; padding:10px 12px; border-bottom:1px solid var(--border)}
    .chat-body{flex:1; overflow-y:auto; padding:10px 12px; display:flex; flex-direction:column; gap:8px}
    .chat-input{padding:10px 12px; border-top:1px solid var(--border); display:flex; gap:8px}
    .msg{background:#0e1833; border:1px solid var(--border); border-radius:12px; padding:8px}
    .msghead{display:flex; gap:6px; align-items:center; font-size:13px}
    .lvl{font-size:12px; color:#8aa0c7}
    .time{margin-left:auto; font-size:11px; color:#8aa0c7}
    .txt{margin-top:4px; font-size:14px; white-space:pre-wrap; word-break:break-word}
    .disabled-note{font-size:12px; color:#8aa0c7; padding:0 12px 10px}

    /* Responsive tweaks */
    @media (max-width: 640px){
      .big{font-size:22px}
      .cr-multi{font-size:28px}
      .header-inner{padding:8px}
      .avatar{width:30px;height:30px}
      .chip{padding:6px 8px}
      .cr-graph-wrap{height:200px}
    }
  </style>
</head>
<body>
  <div class="header">
    <div class="header-inner container">
      <a class="brand" href="#" id="homeLink"><span class="logo"></span> 💎 DL Bank</a>
      <div class="tabs">
        <a class="tab active" id="tab-games">Games</a>
        <a class="tab" id="tab-ref">Referral</a>
        <a class="tab" id="tab-promo">Promo Codes</a>
      </div>
      <div class="right" id="authArea"><!-- filled by js --></div>
    </div>
  </div>

  <div class="container" style="padding-top:16px">
    <!-- Games -->
    <div id="page-games">
      <div class="card">
        <div class="label">Games</div>
        <div class="grid">
          <div class="game-card" id="openCrash">
            <div class="big">🚀 Crash</div>
            <div class="muted">Shared rounds • 10s betting • Live cashout</div>
          </div>
          <div class="game-card">
            <div class="big">🎯 Coin Flip</div>
            <div class="muted">Coming soon.</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Crash -->
    <div id="page-crash" style="display:none">
      <div class="card">
        <div class="cr-top">
          <div class="cr-metric">
            <div class="cr-multi" id="crNow">1.00×</div>
            <div class="cr-small" id="crHint">Loading…</div>
          </div>
          <div>
            <button class="chip" id="backToGames">← Games</button>
          </div>
        </div>

        <div class="cr-graph-wrap">
          <canvas id="crCanvas"></canvas>
        </div>

        <div class="crash-wrap">
          <div class="chips" id="lastBusts">Loading last rounds…</div>

          <div class="grid" style="grid-template-columns:1fr 1fr; gap:12px">
            <div>
              <div class="label">Bet (DL)</div>
              <input id="crBet" type="number" min="1" step="1" placeholder="min 1"/>
            </div>
            <div>
              <div class="label">Auto Cashout (×) — optional</div>
              <input id="crCash" type="number" min="1.01" step="0.01" placeholder="e.g. 2.00"/>
            </div>
          </div>

          <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap">
            <button class="btn primary" id="crPlace">Place Bet</button>
            <button class="btn" id="crCashout" style="display:none">💸 Cash Out</button>
            <span id="crMsg" class="muted"></span>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="label">Your recent rounds</div>
        <div id="crLast" class="muted">—</div>
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
        <div class="grid">
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
          <div class="grid" style="grid-template-columns:2fr 1fr 2fr auto; gap:8px">
            <div><div class="label">Discord ID or &lt;@mention&gt;</div><input id="tIdent" placeholder="ID or <@id>"/></div>
            <div><div class="label">Amount (+/- DL)</div><input id="tAmt" type="number" placeholder="10 or -5"/></div>
            <div><div class="label">Reason (optional)</div><input id="tReason" placeholder="promo/correction/prize"/></div>
            <div style="align-self:end"><button class="btn primary" id="tApply">Apply</button></div>
          </div>
          <div id="tMsg" class="muted" style="margin-top:8px"></div>

          <div class="label" style="margin-top:12px">Create Promo Code</div>
          <div class="grid" style="grid-template-columns:1fr 1fr 1fr; gap:8px">
            <div><div class="label">Code (optional)</div><input id="cCode" placeholder="auto-generate if empty"/></div>
            <div><div class="label">Amount (DL)</div><input id="cAmount" type="number" placeholder="e.g. 10"/></div>
            <div><div class="label">Max Uses</div><input id="cMax" type="number" placeholder="e.g. 100"/></div>
          </div>
          <div style="margin-top:8px"><button class="btn primary" id="cMake">Create</button> <span id="cMsg" class="muted"></span></div>
        </div>
      </div>
    </div>

    <!-- Login card -->
    <div id="loginCard" class="card" style="display:none">
      <div class="label">Get Started</div>
      <p>If you’re not logged in, click <b>Login with Discord</b>. If you’re new, click <b>Register</b>:</p>
      <div style="display:flex; gap:8px; flex-wrap:wrap">
        <a class="btn primary" href="/login">Login with Discord</a>
        <button class="btn" id="registerBtn">Register</button>
      </div>
      <div id="registerInfo" class="muted" style="margin-top:10px; display:none">
        <p>1) Join our Discord server.</p>
        <p>2) DM the bot and run <code>.signup &lt;password&gt;</code> to set your site password (optional).</p>
        <p>3) Then use <b>Login with Discord</b> here to see your balance and play.</p>
      </div>
    </div>
  </div>

  <!-- Modal -->
  <div class="modal" id="modal">
    <div class="box">
      <div style="display:flex; justify-content:space-between; align-items:center; gap:10px">
        <div class="big">Balance — How to Deposit / Withdraw</div>
        <button class="btn" id="mClose">Close</button>
      </div>
      <div class="muted" style="margin-top:8px">
        <p><b>Deposit:</b> Join our Discord and type <code>.deposit</code> in the <i>#deposit</i> channel.</p>
        <p><b>Withdraw:</b> In Discord type <code>.withdraw</code>. (Wiring later.)</p>
      </div>
    </div>
  </div>

  <!-- Global Chat Drawer -->
  <div class="chat-drawer" id="chatDrawer">
    <div class="chat-head">
      <div><b>Global Chat</b></div>
      <div style="display:flex; gap:8px; align-items:center">
        <span id="chatNote" class="lvl"></span>
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

  <script>
    function qs(id){return document.getElementById(id)}
    const tabGames = qs('tab-games'), tabRef=qs('tab-ref'), tabPromo=qs('tab-promo');
    const pgGames=qs('page-games'), pgCrash=qs('page-crash'), pgRef=qs('page-ref'), pgPromo=qs('page-promo'), pgProfile=qs('page-profile'), loginCard=qs('loginCard');

    function fmtDL(n){ return `💎 ${Number(n).toLocaleString()} DL`; }
    async function j(u, opt={}){ const r=await fetch(u, Object.assign({credentials:'include'},opt)); if(!r.ok) throw new Error(await r.text()); return r.json(); }

    function setTab(which){
      [tabGames,tabRef,tabPromo].forEach(t=>t.classList.remove('active'));
      [pgGames,pgCrash,pgRef,pgPromo,pgProfile].forEach(p=>p.style.display='none');
      if(which==='games'){ tabGames.classList.add('active'); pgGames.style.display=''; }
      if(which==='crash'){ pgCrash.style.display=''; }
      if(which==='ref'){ tabRef.classList.add('active'); pgRef.style.display=''; }
      if(which==='promo'){ tabPromo.classList.add('active'); pgPromo.style.display=''; }
      if(which==='profile'){ pgProfile.style.display=''; }
      window.scrollTo({top:0, behavior:'smooth'});
    }
    qs('homeLink').onclick=(e)=>{ e.preventDefault(); setTab('games'); };
    qs('openCrash').onclick=()=>setTab('crash');
    qs('backToGames').onclick=()=>setTab('games');

    // modal
    function openModal(){ qs('modal').style.display='flex'; }
    function closeModal(){ qs('modal').style.display='none'; }
    qs('mClose').onclick = closeModal;
    qs('modal').addEventListener('click', (e)=>{ if(e.target.id==='modal') closeModal(); });

    function safeAvatar(me){ return me.avatar_url || ''; }

    // Header/auth
    async function renderHeader(){
      const auth = qs('authArea');
      try{
        const me = await j('/api/me');
        const bal = await j('/api/balance');
        auth.innerHTML = `
          <span class="chip" id="balanceBtn">${fmtDL(bal.balance)}</span>
          <span class="chip" id="chatBtn">Chat</span>
          <img class="avatar" id="avatarBtn" src="${safeAvatar(me)}" title="${me.username}" onerror="this.style.display='none'">
        `;
        loginCard.style.display='none';
        qs('balanceBtn').onclick = openModal;
        qs('avatarBtn').onclick = ()=>{ setTab('profile'); renderProfile(); };
        qs('chatBtn').onclick = toggleChat;
      }catch(e){
        auth.innerHTML = `
          <a class="btn primary" href="/login">Login</a>
          <button class="btn" id="registerBtn2">Register</button>
        `;
        loginCard.style.display='';
        document.getElementById('registerBtn2').onclick = ()=> {
          const info = qs('registerInfo'); info.style.display = 'block';
          window.scrollTo({top: document.body.scrollHeight, behavior:'smooth'});
        };
      }
    }

    // Profile / Referral / Promos
    async function renderProfile(){
      try{
        const me = await j('/api/me'); const prof = await j('/api/profile');
        const lvl=prof.level, pct=Math.max(0,Math.min(100,prof.progress_pct));
        qs('profileBox').innerHTML = `
          <div style="display:flex; gap:12px; align-items:center; flex-wrap:wrap">
            <img src="${safeAvatar(me)}" class="avatar" style="width:56px; height:56px">
            <div><div class="big">${me.username}</div><div class="muted">ID: ${me.id}</div></div>
            <div style="margin-left:auto; display:flex; gap:8px; align-items:center"><a class="btn" href="/logout">Logout</a></div>
          </div>
          <div class="grid" style="margin-top:12px">
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
        if(me.id === '__OWNER_ID__'){ ownerPanel.style.display=''; }
        else ownerPanel.style.display='none';

        // Owner actions
        const tApply = qs('tApply'); if(tApply){
          tApply.onclick = async ()=>{
            const identifier = qs('tIdent').value.trim();
            const amount = parseInt(qs('tAmt').value,10) || 0;
            const reason = qs('tReason').value.trim() || null;
            const msg = qs('tMsg'); msg.textContent = '';
            try{
              const r = await j('/api/admin/adjust', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({identifier, amount, reason})});
              msg.textContent = 'Updated. New balance for ' + identifier + ' = ' + fmtDL(r.new_balance);
            }catch(e){ msg.textContent = 'Error: '+e.message; }
          };
        }
        const cMake = qs('cMake'); if(cMake){
          cMake.onclick = async ()=>{
            const code = qs('cCode').value.trim() || null;
            const amount = parseInt(qs('cAmount').value,10) || 0;
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
      }catch(e){}
    }

    // -------- Crash state + live now polling (no end leak) --------
    const crNowEl = qs('crNow'), crHint = qs('crHint'), crMsg = qs('crMsg');
    const lastBustsEl = qs('lastBusts'), cashBtn = qs('crCashout');

    let crPhase = 'betting';
    let roundId = null;
    let haveActiveBet = false;
    let betResolved = false;

    async function refreshCrash(){
      try{
        const s = await j('/api/crash/state');
        roundId = s.round_id; crPhase = s.phase;
        // bust history
        lastBustsEl.innerHTML = s.last_busts.length
          ? s.last_busts.map(v=>`<span class="chip2 ${v<2?'bust-bad':'bust-good'}">${v.toFixed(2)}×</span>`).join('')
          : 'No history yet.';

        // your bet UI
        haveActiveBet = !!(s.your_bet && !s.your_bet.resolved);
        betResolved = !!(s.your_bet && s.your_bet.resolved);
        cashBtn.style.display = (crPhase==='running' && haveActiveBet) ? '' : 'none';

        if(crPhase==='betting'){
          const left = Math.max(0, Math.round((Date.parse(s.betting_ends_at) - Date.now())/1000));
          crHint.textContent = `Betting open — closes in ${left}s`; crNowEl.textContent = '1.00×';
        } else if(crPhase==='running'){
          crHint.textContent = 'Round running… Tap Cash Out to take profit.';
        } else { // ended
          crHint.textContent = 'Preparing next round…';
          if(s.bust) crNowEl.textContent = s.bust.toFixed(2)+'×';
        }
      }catch(e){
        crHint.textContent = 'Disconnected. Reconnecting…';
      }
    }

    // Live "now" multiplier polling (every 300ms while running)
    let nowTimer=null;
    async function pollNow(){
      if(crPhase!=='running'){ crNowEl.textContent = crNowEl.textContent || '1.00×'; return; }
      try{
        const n = await j('/api/crash/now');
        if(n.phase==='running'){
          crNowEl.textContent = n.multiplier.toFixed(2)+'×';
          drawPoint(n.multiplier);
        }
      }catch(e){ /* ignore */ }
    }

    // Place bet
    qs('crPlace').onclick = async ()=>{
      try{
        const bet = parseInt(document.getElementById('crBet').value,10);
        const cash = parseFloat(document.getElementById('crCash').value);
        if(Number.isNaN(bet) || bet < 1) throw new Error('Enter a bet of at least 1 DL.');
        if(document.getElementById('crCash').value.trim()!=='' && (Number.isNaN(cash) || cash < 1.01)) throw new Error('Auto cashout must be at least 1.01×, or leave empty.');
        await j('/api/crash/bet', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({bet, cashout: (Number.isNaN(cash)? 1000 : cash)})});
        crMsg.textContent = 'Bet placed for this round.';
        await renderHeader(); // update balance
        haveActiveBet = true;
      }catch(e){ crMsg.textContent = 'Error: '+e.message; }
    };

    // Cash Out
    cashBtn.onclick = async ()=>{
      try{
        const r = await j('/api/crash/cashout', {method:'POST'});
        crMsg.textContent = 'Cashed out at '+r.multiplier.toFixed(2)+'× • Won '+fmtDL(r.win);
        await renderHeader(); // update balance
        haveActiveBet = false;
        cashBtn.style.display = 'none';
      }catch(e){
        crMsg.textContent = 'Cashout failed: '+e.message;
      }
    };

    // Graph (vertical curve)
    const canv = qs('crCanvas'); const ctx = canv.getContext('2d');
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
      // grid
      ctx.globalAlpha = 0.25;
      ctx.strokeStyle = '#23304c';
      ctx.lineWidth = 1;
      for(let i=0;i<5;i++){
        const y = h - (i*h/5);
        ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(w,y); ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }

    let lastMult = 1.0;
    function drawPoint(mult){
      const w = canv.clientWidth, h = canv.clientHeight;
      // y increases upward (1.00x near bottom). Map multiplier to y with soft log.
      // y_frac = log(mult)/log(20) capped to [0,1]
      const maxM = 20.0;
      const yf = Math.min(1, Math.log(Math.max(1.0001,mult)) / Math.log(maxM));
      const y = h - (h * yf);
      // x advances slowly with multiplier to create curved "upward" stroke
      const xf = Math.min(1, Math.log(Math.max(1.0001,mult)) / Math.log(maxM));
      const x = (w * (0.1 + 0.8*xf)); // keep some left margin

      // draw curve segment from last point to new point (rounded stroke)
      ctx.lineJoin = 'round'; ctx.lineCap = 'round';
      ctx.strokeStyle = ctx.createLinearGradient(0,h,0,0);
      ctx.strokeStyle.addColorStop(0,'#22c1dc'); ctx.strokeStyle.addColorStop(1,'#3b82f6');
      ctx.lineWidth = 4;
      // Keep a path: we just draw incremental segment; if mult decreased (new round), reset
      if(mult < lastMult){ redrawAxis(); }
      lastMult = mult;

      // Save current point on canvas using a very light approach:
      // We'll sample many small points by just drawing small line segments.
      // Compute previous x,y from a slightly smaller multiplier
      const prevM = Math.max(1.0, mult/1.01);
      const prevYf = Math.min(1, Math.log(Math.max(1.0001,prevM)) / Math.log(maxM));
      const py = h - (h * prevYf);
      const pxf = Math.min(1, Math.log(Math.max(1.0001,prevM)) / Math.log(maxM));
      const px = (w * (0.1 + 0.8*pxf));

      ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(x, y); ctx.stroke();
    }

    // Chat (fixed bug: JS booleans)
    const drawer = qs('chatDrawer'), chatBody=qs('chatBody'), chatText=qs('chatText'), chatSend=qs('chatSend');
    const chatNote = qs('chatNote'), chatDisabled = qs('chatDisabled');
    let chatOpen=false, chatPoll=null, lastChatId=0, myLevel=0, isLogged=false;

    function scrollChatToBottom(){ chatBody.scrollTop = chatBody.scrollHeight; }
    function renderMsg(m){
      const wrap = document.createElement('div'); wrap.className='msg';
      const head = document.createElement('div'); head.className='msghead';
      const name = document.createElement('b'); name.textContent = m.username;
      const lvl = document.createElement('span'); lvl.className='lvl'; lvl.textContent = `[Lv ${m.level}]`;
      const ts = document.createElement('span'); ts.className='time'; ts.textContent = new Date(m.created_at).toLocaleTimeString();
      head.appendChild(name); head.appendChild(lvl); head.appendChild(ts);
      const txt = document.createElement('div'); txt.className='txt'; txt.textContent = m.text;
      wrap.appendChild(head); wrap.appendChild(txt);
      chatBody.appendChild(wrap);
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
      try{ await j('/api/me'); isLogged = true; }catch(e){ isLogged = false; }
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
      await updateChatGate(); await fetchChat(true);
      if(chatPoll) clearInterval(chatPoll);
      chatPoll = setInterval(()=>{ if(chatOpen) fetchChat(false); }, 2000);
    }
    function closeChat(){ drawer.classList.remove('open'); chatOpen=false; if(chatPoll){clearInterval(chatPoll); chatPoll=null;} }
    function toggleChat(){ if(chatOpen) closeChat(); else openChat(); }
    qs('chatClose').onclick = closeChat;
    chatSend.onclick = async ()=>{
      const t = chatText.value.trim();
      if(!t) return;
      try{
        await j('/api/chat/send', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({text:t})});
        chatText.value=''; await fetchChat(false);
      }catch(e){ alert(e.message); await updateChatGate(); }
    };
    chatText.addEventListener('keydown', (e)=>{ if(e.key==='Enter'){ e.preventDefault(); chatSend.click(); } });

    // Periodic
    setInterval(()=>{ if(pgCrash.style.display!=='none') refreshCrash(); }, 1000);
    resizeCanvas();
    setInterval(()=>{ if(pgCrash.style.display!=='none') pollNow(); }, 300);

    // Other tabs & data
    tabGames.onclick=()=>setTab('games');
    tabRef.onclick=()=>{ setTab('ref'); renderReferral(); };
    tabPromo.onclick=()=>{ setTab('promo'); renderPromos(); };
    document.getElementById('openCrash').onclick=()=>{ setTab('crash'); };

    async function renderOther(){
      await renderReferral();
      await renderPromos();
      await renderProfile();
      try{
        const h = await j('/api/game/crash/history');
        document.getElementById('crLast').innerHTML = h.rows.length
          ? `<table><thead><tr><th>When</th><th>Bet</th><th>Goal</th><th>Bust</th><th>Win</th><th>XP</th></tr></thead>
              <tbody>${
                h.rows.map(r=>`
                  <tr>
                    <td>${new Date(r.created_at).toLocaleString()}</td>
                    <td>${fmtDL(r.bet)}</td>
                    <td>${r.cashout.toFixed(2)}×</td>
                    <td>${r.bust.toFixed(2)}×</td>
                    <td style="color:${r.win>0?'#34d399':'#ef4444'}">${r.win>0?fmtDL(r.win):'-'}</td>
                    <td>${r.xp_gain}</td>
                  </tr>`).join('')
              }</tbody></table>`
          : 'No recent rounds.';
      }catch(e){}
    }

    // Init
    renderHeader();
    renderOther();
    refreshCrash();
  </script>
</body>
</html>
"""

# Build final HTML (inject owner id)
INDEX_HTML = HTML_TEMPLATE.replace("__OWNER_ID__", str(OWNER_ID))

# ----- Routes -----
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(INDEX_HTML)

@app.get("/login")
async def login():
    if not (CLIENT_ID and OAUTH_REDIRECT and CLIENT_SECRET):
        raise HTTPException(500, "OAuth not configured")
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "scope": "identify",
        "redirect_uri": OAUTH_REDIRECT,
        "prompt": "none"
    }
    return RedirectResponse(f"{DISCORD_API}/oauth2/authorize?{urlencode(params)}")

@app.get("/callback")
async def callback(code: str | None = None):
    if not code:
        raise HTTPException(400, "Missing code")
    async with httpx.AsyncClient() as client:
        token = (await client.post(f"{DISCORD_API}/oauth2/token", data={
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": OAUTH_REDIRECT
        })).json()
        if "access_token" not in token:
            raise HTTPException(400, f"OAuth token error: {token}")
        me = (await client.get(f"{DISCORD_API}/users/@me",
                               headers={"Authorization": f"{token['token_type']} {token['access_token']}"}
                              )).json()
    resp = RedirectResponse(url="/")
    payload = {"id": str(me["id"]), "username": me.get("username", "#"), "avatar": me.get("avatar")}
    set_session(resp, payload)
    return resp

@app.get("/logout")
async def logout():
    resp = RedirectResponse(url="/")
    resp.delete_cookie("session")
    return resp

@app.get("/api/me")
async def api_me(request: Request):
    user = read_session(request)
    if not user: raise HTTPException(401, "Not logged in")
    return {"id": user["id"], "username": user.get("username", ""), "avatar_url": avatar_url_from(user["id"], user.get("avatar"))}

@app.get("/api/balance")
async def api_balance(request: Request):
    user = read_session(request)
    if not user: raise HTTPException(401, "Not logged in")
    return {"balance": get_balance(str(user["id"]))}

# Profiles / referrals
class SetNameBody(BaseModel): name: str

@app.get("/api/referral/state")
async def api_ref_state(request: Request):
    user = read_session(request)
    if not user: raise HTTPException(401, "Not logged in")
    return {"name": get_profile_name(str(user["id"]))}

@app.post("/api/profile/set_name")
async def api_set_name(request: Request, body: SetNameBody):
    user = read_session(request)
    if not user: raise HTTPException(401, "Not logged in")
    try: return set_profile_name(str(user["id"]), body.name)
    except ValueError as e: raise HTTPException(400, str(e))

@app.get("/api/profile")
async def api_profile(request: Request):
    user = read_session(request)
    if not user: raise HTTPException(401, "Not logged in")
    return {"user_id": user["id"], **profile_info(user["id"])}

# Promos
class RedeemBody(BaseModel): code: str

@app.get("/api/promo/my")
async def api_promo_my(request: Request):
    user = read_session(request)
    if not user: raise HTTPException(401, "Not logged in")
    @with_conn
    def _my(cur, uid):
        cur.execute("SELECT code, redeemed_at FROM promo_redemptions WHERE user_id=%s ORDER BY redeemed_at DESC LIMIT 50", (uid,))
        return [{"code":r[0],"redeemed_at":str(r[1])} for r in cur.fetchall()]
    return {"rows": _my(user["id"])}

@app.post("/api/promo/redeem")
async def api_promo_redeem(request: Request, body: RedeemBody):
    user = read_session(request)
    if not user: raise HTTPException(401, "Not logged in")
    try:
        new_bal = redeem_promo(str(user["id"]), body.code)
        return {"new_balance": new_bal}
    except PromoAlreadyRedeemed as e: raise HTTPException(400, str(e))
    except PromoInvalid as e: raise HTTPException(400, str(e))
    except PromoExpired as e: raise HTTPException(400, str(e))
    except PromoExhausted as e: raise HTTPException(400, str(e))

# Owner admin
class AdjustBody(BaseModel):
    identifier: str
    amount: int
    reason: Optional[str] = None

class CreatePromoBody(BaseModel):
    code: Optional[str] = None
    amount: int
    max_uses: int = 1
    expires_at: Optional[str] = None

def parse_user_identifier(identifier: str) -> Optional[str]:
    if not identifier: return None
    cleaned = identifier.strip().replace("<@!", "").replace("<@", "").replace(">", "")
    return cleaned if cleaned.isdigit() and len(cleaned)>=17 else None

def require_owner(request: Request):
    user = read_session(request)
    if not user: raise HTTPException(401, "Not logged in")
    if str(user["id"]) != str(OWNER_ID): raise HTTPException(403, "Owner only")
    return user

@app.post("/api/admin/adjust")
async def api_admin_adjust(request: Request, body: AdjustBody):
    actor = require_owner(request)
    uid = parse_user_identifier(body.identifier)
    if not uid: raise HTTPException(400, "Invalid identifier (use raw ID or <@mention>)")
    if body.amount == 0: raise HTTPException(400, "Amount cannot be zero")
    new_balance = adjust_balance(str(actor["id"]), uid, int(body.amount), body.reason)
    return {"user_id": uid, "new_balance": new_balance}

@app.post("/api/admin/promo/create")
async def api_admin_promo_create(request: Request, body: CreatePromoBody):
    require_owner(request)
    if body.amount == 0: raise HTTPException(400, "Amount cannot be zero")
    if body.max_uses < 1: raise HTTPException(400, "Max uses must be >= 1")
    return create_promo(str(OWNER_ID), body.code, int(body.amount), int(body.max_uses), body.expires_at)

# Crash API (no end leak)
class BetBody(BaseModel):
    bet: int
    cashout: float

@app.get("/api/crash/state")
async def api_crash_state(request: Request):
    user = read_session(request)
    if not user: raise HTTPException(401, "Not logged in")

    r = load_round()
    rid, phase = (r["id"], r["status"]) if r else (None, "betting")
    yb = your_bet(rid, user["id"]) if rid else None
    return {
        "phase": phase,
        "round_id": rid,
        "betting_opens_at": iso(r["betting_opens_at"]) if r else None,
        "betting_ends_at":  iso(r["betting_ends_at"])  if r else None,
        # intentionally NOT sending expected_end_at to avoid revealing crash timing
        "started_at":       iso(r["started_at"])       if r else None,
        "bust": r["bust"] if (r and phase=='ended') else None,   # reveal bust only after end
        "your_bet": yb,
        "min_bet": MIN_BET,
        "last_busts": last_busts(15)
    }

@app.get("/api/crash/now")
async def api_crash_now():
    # returns current multiplier only while running (no bust / no end time)
    r = load_round()
    if not r or r["status"] != "running":
        return {"phase": r["status"] if r else "betting", "multiplier": 1.0}
    m = current_multiplier(r["started_at"], r["expected_end_at"], r["bust"], now_utc())
    return {"phase": "running", "multiplier": m}

@app.post("/api/crash/bet")
async def api_crash_bet(request: Request, body: BetBody):
    user = read_session(request)
    if not user: raise HTTPException(401, "Not logged in")
    bet = int(body.bet); cash = float(body.cashout)
    if bet < MIN_BET: raise HTTPException(400, f"Min bet is {MIN_BET} DL")
    if bet > MAX_BET: raise HTTPException(400, f"Max bet is {MAX_BET} DL")
    if cash < 1.01: cash = 1000.0  # treat empty/invalid as very high (no auto cash)
    try:
        res = place_bet(user["id"], bet, cash)
        return {"ok": True, "round_id": res["round_id"]}
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.post("/api/crash/cashout")
async def api_crash_cashout(request: Request):
    user = read_session(request)
    if not user: raise HTTPException(401, "Not logged in")
    try:
        res = cashout_now(user["id"])
        return {"ok": True, **res}
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.get("/api/game/crash/history")
async def api_game_crash_history(request: Request, limit: int = Query(10, ge=1, le=50)):
    user = read_session(request)
    if not user: raise HTTPException(401, "Not logged in")
    return {"rows": your_history(user["id"], limit)}

# Global Chat API
class ChatSendBody(BaseModel):
    text: str

@app.get("/api/chat/fetch")
async def api_chat_fetch(request: Request, since_id: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)):
    user = read_session(request)
    if not user: raise HTTPException(401, "Not logged in")
    msgs = chat_fetch(since_id, min(limit, 200))
    return {"messages": msgs}

@app.post("/api/chat/send")
async def api_chat_send(request: Request, body: ChatSendBody):
    user = read_session(request)
    if not user: raise HTTPException(401, "Not logged in")
    try:
        res = chat_send(str(user["id"]), user.get("username","user"), body.text)
        return {"ok": True, "id": res["id"], "created_at": res["created_at"]}
    except PermissionError as e:
        raise HTTPException(403, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.get("/health")
async def health():
    return {"ok": True}

# ---------- Crash loop ----------
async def crash_loop():
    while True:
        rid, r = ensure_betting_round()
        now = now_utc()

        if r["status"] == "betting":
            wait = (r["betting_ends_at"] - now).total_seconds()
            if wait > 0:
                await asyncio.sleep(min(wait, 0.5))
            else:
                begin = begin_running(rid)
                # do not resolve here; allow live cashouts; we'll resolve at end
        elif r["status"] == "running":
            if r["expected_end_at"]:
                wait = (r["expected_end_at"] - now).total_seconds()
                if wait > 0:
                    await asyncio.sleep(min(wait, 0.3))
                else:
                    finish_round(rid)
                    create_next_betting()
                    await asyncio.sleep(0.3)
            else:
                finish_round(rid)
                create_next_betting()
                await asyncio.sleep(0.3)
        else:  # ended
            create_next_betting()
            await asyncio.sleep(0.3)

# ---------- Runner ----------
async def main():
    import traceback, sys
    init_db()

    config = uvicorn.Config(app, host="0.0.0.0", port=PORT, log_level="info")
    server = uvicorn.Server(config)

    async def run_bot_forever():
        while True:
            try:
                if not BOT_TOKEN: raise RuntimeError("DISCORD_TOKEN env var not set.")
                await bot.start(BOT_TOKEN)
            except discord.errors.LoginFailure:
                print("[bot] LoginFailure: bad token. Fix DISCORD_TOKEN.", file=sys.stderr)
                await asyncio.sleep(3600)
            except Exception:
                traceback.print_exc()
                await asyncio.sleep(10)

    await asyncio.gather(server.serve(), run_bot_forever(), crash_loop())

if __name__ == "__main__":
    asyncio.run(main())
