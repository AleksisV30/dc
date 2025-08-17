import os, json, asyncio, re, random, string, math, secrets, datetime
from urllib.parse import urlencode
from typing import Optional, Tuple

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

    # Multiplayer Crash: rounds + bets
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
            cashout NUMERIC(8,2) NOT NULL,
            win INTEGER NOT NULL DEFAULT 0,
            resolved BOOLEAN NOT NULL DEFAULT FALSE,
            PRIMARY KEY(round_id, user_id)
        )
    """)

    # Per-player game log (reuses from earlier single-player crash)
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

# ---- Basic helpers / balances / profiles ----
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
def get_profile_name(cur, user_id: str):
    cur.execute("SELECT display_name FROM profiles WHERE user_id = %s", (user_id,))
    r = cur.fetchone(); return r[0] if r else None

@with_conn
def ensure_profile_row(cur, user_id: str):
    cur.execute("INSERT INTO profiles(user_id, display_name, name_lower) VALUES (%s,%s,%s) ON CONFLICT (user_id) DO NOTHING",
                (user_id, f"user_{user_id[-4:]}", f"user_{user_id[-4:]}"))

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

@with_conn
def add_xp(cur, user_id: str, amount: int):
    ensure_profile_row(user_id)
    cur.execute("UPDATE profiles SET xp = xp + %s WHERE user_id=%s", (amount, user_id))

# ---- Promos ----
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
    cur.execute("INSERT INTO balances(user_id,balance) VALUES (%s,0) ON CONFLICT (user_id) DO NOTHING", (user_id,))
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
    # ~1.2s .. 7s depending on bust, for client animation
    return min(7.0, 1.2 + math.log(bust+1.0)*1.6)

@with_conn
def ensure_betting_round(cur) -> Tuple[int, dict]:
    # Return current round (betting/running) or create betting if none
    cur.execute("SELECT id,status,betting_opens_at,betting_ends_at,started_at,expected_end_at,bust FROM crash_rounds ORDER BY id DESC LIMIT 1")
    r = cur.fetchone()
    now = datetime.datetime.utcnow()
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
    # validate current betting round
    cur.execute("""SELECT id, betting_ends_at FROM crash_rounds
                   WHERE status='betting'
                   ORDER BY id DESC LIMIT 1""")
    row = cur.fetchone()
    if not row: raise ValueError("Betting is closed")
    round_id, ends_at = int(row[0]), row[1]
    cur.execute("SELECT NOW() < %s", (ends_at,))
    if not cur.fetchone()[0]:
        raise ValueError("Betting just closed")

    # ensure balance & debit
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
        # refund if duplicate
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
    # if already running/ended, no-op
    cur.execute("SELECT status FROM crash_rounds WHERE id=%s FOR UPDATE", (round_id,))
    st = cur.fetchone()
    if not st: return None
    if st[0] != 'betting': return None

    bust = gen_bust(HOUSE_EDGE)
    dur = run_duration_for(bust)
    now = datetime.datetime.utcnow()
    exp_end = now + datetime.timedelta(seconds=dur)
    cur.execute("""UPDATE crash_rounds
                   SET status='running', bust=%s, started_at=%s, expected_end_at=%s
                   WHERE id=%s""",
                (float(bust), now, exp_end, round_id))
    return {"bust": bust, "expected_end_at": exp_end}

@with_conn
def resolve_bets(cur, round_id: int, bust: float):
    # credit winners, log results, add XP
    cur.execute("""SELECT user_id, bet, cashout FROM crash_bets
                   WHERE round_id=%s AND resolved=FALSE""", (round_id,))
    bets = cur.fetchall()
    for user_id, bet, cash in bets:
        win = 0
        if float(cash) <= bust:
            win = int(math.floor(int(bet) * float(cash) + 1e-9))
            cur.execute("UPDATE balances SET balance=balance+%s WHERE user_id=%s", (win, user_id))
        xp_gain = max(1, min(int(bet), 50))
        ensure_profile_row(user_id)
        cur.execute("UPDATE profiles SET xp=xp+%s WHERE user_id=%s", (xp_gain, user_id))
        cur.execute("""INSERT INTO crash_games(user_id, bet, cashout, bust, win, xp_gain)
                       VALUES(%s,%s,%s,%s,%s,%s)""",
                    (user_id, int(bet), float(cash), float(bust), int(win), xp_gain))
        cur.execute("""UPDATE crash_bets SET win=%s, resolved=TRUE
                       WHERE round_id=%s AND user_id=%s""", (int(win), round_id, user_id))

@with_conn
def finish_round(cur, round_id: int):
    now = datetime.datetime.utcnow()
    cur.execute("""UPDATE crash_rounds
                   SET status='ended', ended_at=%s
                   WHERE id=%s AND status='running'""", (now, round_id))

@with_conn
def create_next_betting(cur):
    now = datetime.datetime.utcnow()
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
    cur.execute("SELECT bet, cashout FROM crash_bets WHERE round_id=%s AND user_id=%s", (round_id, user_id))
    r = cur.fetchone()
    return {"bet": int(r[0]), "cashout": float(r[1])} if r else None

@with_conn
def your_history(cur, user_id: str, limit: int = 10):
    cur.execute("""SELECT bet, cashout, bust, win, xp_gain, created_at
                   FROM crash_games WHERE user_id=%s
                   ORDER BY id DESC LIMIT %s""", (user_id, limit))
    return [{"bet":int(r[0]),"cashout":float(r[1]),"bust":float(r[2]),"win":int(r[3]),"xp_gain":int(r[4]),"created_at":str(r[5])} for r in cur.fetchall()]

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

# ---------- Frontend HTML ----------
INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>💎 DL Bank</title>
  <style>
    :root{ --bg:#0b1220; --card:#121a2b; --muted:#8aa0c7; --text:#e8efff; --accent:#60a5fa; --ok:#34d399; --warn:#f59e0b; --err:#ef4444; --border:#23304c; }
    *{box-sizing:border-box}
    body{background:linear-gradient(180deg,#0a1020,#0e1530); color:var(--text); font-family:system-ui,Segoe UI,Roboto,Arial; margin:0; min-height:100vh;}
    a{color:inherit; text-decoration:none}
    .container{max-width:1100px; margin:0 auto; padding:16px}
    .header{position:sticky; top:0; z-index:10; background:linear-gradient(135deg,#0e1630 0%,#0a1124 100%); border-bottom:1px solid var(--border);}
    .header-inner{display:flex; align-items:center; justify-content:space-between; gap:12px; padding:12px 16px;}
    .brand{display:flex; align-items:center; gap:10px; font-weight:800; letter-spacing:.2px}
    .nav{display:flex; gap:10px; align-items:center}
    .tab{padding:8px 12px; border:1px solid var(--border); border-radius:10px; background:#0f1a33; cursor:pointer}
    .tab.active{background:linear-gradient(135deg,#3b82f6,#22c1dc); border-color:transparent}
    .right{display:flex; gap:10px; align-items:center}
    .chip{background:#0c1631; border:1px solid var(--border); color:#dce7ff; padding:6px 10px; border-radius:999px; font-size:12px; white-space:nowrap; cursor:pointer}
    .avatar{width:28px;height:28px;border-radius:50%;object-fit:cover;border:1px solid var(--border); cursor:pointer}
    .btn{display:inline-flex; align-items:center; gap:8px; padding:8px 12px; border-radius:10px; border:1px solid var(--border); background:#0f1a33; cursor:pointer}
    .btn.primary{background:linear-gradient(135deg,#3b82f6,#22c1dc); border-color:transparent}
    .grid{display:grid; gap:16px; grid-template-columns:1fr}
    @media(min-width:1000px){.grid{grid-template-columns:1fr 1fr}}
    .card{background:var(--card); border:1px solid var(--border); border-radius:16px; padding:16px}
    .label{color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.1em}
    .big{font-size:28px; font-weight:800}
    .muted{color:var(--muted)}
    input{width:100%; background:#0e1833; color:#e8efff; border:1px solid var(--border); border-radius:10px; padding:10px}
    .game-card{background:#0f1a33;border:1px solid var(--border);border-radius:16px;padding:14px;cursor:pointer;transition:transform .08s ease}
    .game-card:hover{transform:translateY(-2px)}
    .banner{font-weight:900; font-size:18px}
    .owner{margin-top:16px; border-top:1px dashed var(--border); padding-top:12px}

    /* Crash page */
    .crash-wrap{display:flex; flex-direction:column; gap:10px}
    .cr-row{display:grid; grid-template-columns:1fr 1fr; gap:12px}
    .cr-line{position:relative; height:18px; background:#0e1833; border:1px solid var(--border); border-radius:999px; overflow:hidden}
    .cr-fill{position:absolute; left:0; top:0; bottom:0; width:0%; background:linear-gradient(90deg,#22c1dc,#3b82f6)}
    .cr-marker{position:absolute; top:-6px; width:2px; height:30px; background:#f59e0b; opacity:.9}
    .cr-marker.bust{background:#ef4444}
    .cr-head{display:flex; align-items:center; gap:10px}
    .cr-multi{font-size:28px; font-weight:900}
    .cr-small{font-size:12px; color:var(--muted)}
    table{width:100%; border-collapse:collapse; margin-top:10px}
    th,td{border-bottom:1px solid var(--border); padding:8px 6px; text-align:left}
    .win{color:#34d399}
    .lose{color:#ef4444}
    .bust-bad{color:#ef4444; font-weight:700}
    .bust-good{color:#34d399; font-weight:700}

    /* Modal */
    .modal{position:fixed; inset:0; background:rgba(0,0,0,.6); display:none; align-items:center; justify-content:center; padding:20px}
    .modal .box{background:#0f1a33; border:1px solid var(--border); border-radius:16px; padding:16px; max-width:520px; width:100%}
    .chips{display:flex; flex-wrap:wrap; gap:6px; margin-top:8px}
    .chip2{padding:4px 8px; border-radius:999px; border:1px solid var(--border); background:#0c1631}
  </style>
</head>
<body>
  <div class="header">
    <div class="header-inner container">
      <a class="brand" href="#" id="homeLink"><span class="banner">💎 DL Bank</span></a>
      <div class="nav">
        <a class="tab active" id="tab-games">Games</a>
        <a class="tab" id="tab-ref">Referral</a>
        <a class="tab" id="tab-promo">Promo Codes</a>
      </div>
      <div class="right" id="authArea"><!-- balance + avatar OR login/register --></div>
    </div>
  </div>

  <div class="container" style="padding-top:16px">
    <!-- Games list -->
    <div id="page-games">
      <div class="card">
        <div class="label">Games</div>
        <div class="grid">
          <div class="game-card" id="openCrash">
            <div class="big">🚀 Crash</div>
            <div class="muted">Click to enter the lobby. Shared rounds with 10s betting.</div>
          </div>
          <div class="game-card">
            <div class="big">🎯 Coin Flip</div>
            <div class="muted">Coming soon.</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Crash page -->
    <div id="page-crash" style="display:none">
      <div class="card">
        <div class="cr-head">
          <div class="big">🚀 Crash — Shared rounds</div>
          <div class="cr-small">House edge 1% • Min bet 1 DL</div>
        </div>

        <div class="crash-wrap">
          <div class="chips" id="lastBusts">Loading last rounds…</div>

          <div class="cr-row" style="margin-top:6px">
            <div>
              <div class="label">Bet (DL)</div>
              <input id="crBet" type="number" min="1" step="1" placeholder="min 1"/>
            </div>
            <div>
              <div class="label">Cashout goal (×)</div>
              <input id="crCash" type="number" min="1.01" step="0.01" value="2.00"/>
            </div>
          </div>

          <div class="cr-head" style="margin-top:4px">
            <div class="cr-multi" id="crNow">—</div>
            <div class="cr-small" id="crHint">Connecting…</div>
          </div>

          <div class="cr-line">
            <div class="cr-fill" id="crFill"></div>
            <div class="cr-marker" id="crCashMarker" style="display:none"></div>
            <div class="cr-marker bust" id="crBustMarker" style="display:none"></div>
          </div>

          <div style="display:flex; gap:8px; align-items:center">
            <button class="btn primary" id="crPlace">Place Bet</button>
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
      <p>If you’re not logged in, click <b>Login with Discord</b>. If you’re new, click <b>Register</b> to see how to join:</p>
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

  <!-- Modal (deposit/withdraw info) -->
  <div class="modal" id="modal">
    <div class="box">
      <div style="display:flex; justify-content:space-between; align-items:center; gap:10px">
        <div class="big">Balance — How to Deposit / Withdraw</div>
        <button class="btn" id="mClose">Close</button>
      </div>
      <div class="muted" style="margin-top:8px">
        <p><b>Deposit:</b> Join our Discord and type <code>.deposit</code> in the <i>#deposit</i> channel.</p>
        <p><b>Withdraw:</b> In Discord type <code>.withdraw</code>. (We’ll wire this up later.)</p>
      </div>
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

    // modal
    function openModal(){ qs('modal').style.display='flex'; }
    function closeModal(){ qs('modal').style.display='none'; }
    qs('mClose').onclick = closeModal;
    qs('modal').addEventListener('click', (e)=>{ if(e.target.id==='modal') closeModal(); });

    function safeAvatar(me){ return me.avatar_url || ''; }

    // Global header/auth render
    async function renderHeader(){
      const auth = qs('authArea');
      try{
        const me = await j('/api/me');
        const bal = await j('/api/balance');
        auth.innerHTML = `
          <span class="chip" id="balanceBtn">${fmtDL(bal.balance)}</span>
          <img class="avatar" id="avatarBtn" src="${safeAvatar(me)}" title="${me.username}" onerror="this.style.display='none'">
        `;
        loginCard.style.display='none';
        qs('balanceBtn').onclick = openModal;
        qs('avatarBtn').onclick = ()=>setTab('profile');
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

    // Profile / Referral / Promo helpers
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
        if(me.id === '${OWNER_ID}'){ ownerPanel.style.display=''; }
        else ownerPanel.style.display='none';
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

    // Crash LIVE state polling + UI
    const crNow = ()=>document.getElementById('crNow');
    const crHint = ()=>document.getElementById('crHint');
    const crFill = ()=>document.getElementById('crFill');
    const cashMarker = ()=>document.getElementById('crCashMarker');
    const bustMarker = ()=>document.getElementById('crBustMarker');
    const lastBustsEl = ()=>document.getElementById('lastBusts');
    const crMsgEl = ()=>document.getElementById('crMsg');

    let animRAF = null;
    function resetLine(){ if(animRAF) cancelAnimationFrame(animRAF); crFill().style.width='0%'; crNow().textContent='—'; cashMarker().style.display='none'; bustMarker().style.display='none'; }

    function setMarkers(target, bust){
      if(!bust || bust<=1){ cashMarker().style.display='none'; bustMarker().style.display='none'; return; }
      const lnB = Math.log(bust);
      const tFrac = Math.min(1, Math.log(Math.max(1.0001,target)) / lnB);
      cashMarker().style.left = (tFrac*100)+'%'; cashMarker().style.display='block';
      bustMarker().style.left = '100%'; bustMarker().style.display='block';
    }

    function animateRunning(startISO, endISO, bust){
      const start = new Date(startISO).getTime(), end = new Date(endISO).getTime(); const lnB = Math.log(bust);
      function frame(){
        const now = Date.now(); const t = Math.min(1, (now-start)/(end-start));
        const m = Math.exp(lnB * t); crNow().textContent=m.toFixed(2)+'×';
        crFill().style.width = (Math.log(m)/lnB*100) + '%';
        if(t<1) animRAF=requestAnimationFrame(frame);
        else { crNow().textContent=bust.toFixed(2)+'×'; }
      }
      frame();
    }

    async function refreshCrash(){
      try{
        const s = await j('/api/crash/state');
        // last 15 busts
        lastBustsEl().innerHTML = s.last_busts.length
          ? s.last_busts.map(v=>`<span class="chip2 ${v<2?'bust-bad':'bust-good'}">${v.toFixed(2)}×</span>`).join('')
          : 'No history yet.';
        // your bet info
        if(s.your_bet){ crMsgEl().textContent = `Your bet: ${fmtDL(s.your_bet.bet)} @ ${s.your_bet.cashout.toFixed(2)}×`; }
        else crMsgEl().textContent = '';

        const now = Date.parse(s.now);
        if(s.phase==='betting'){
          resetLine();
          crNow().textContent = '1.00×';
          const left = Math.max(0, Math.round((Date.parse(s.betting_ends_at) - now)/1000));
          crHint().textContent = `Betting open — closes in ${left}s`;
          setMarkers(parseFloat(document.getElementById('crCash').value||'2'), 2); // temp markers
        } else if(s.phase==='running'){
          resetLine();
          crHint().textContent = 'Round running…';
          setMarkers(parseFloat(document.getElementById('crCash').value||'2'), s.bust);
          animateRunning(s.started_at, s.expected_end_at, s.bust);
        } else { // ended -> next betting begins immediately
          resetLine();
          crNow().textContent = s.bust ? s.bust.toFixed(2)+'×' : '—';
          crHint().textContent = 'Preparing next round…';
        }
      }catch(e){
        crHint().textContent = 'Disconnected. Reconnecting…';
      }
    }

    // Place bet
    document.getElementById('crPlace').onclick = async ()=>{
      try{
        const bet = parseInt(document.getElementById('crBet').value,10);
        const cash = parseFloat(document.getElementById('crCash').value);
        if(Number.isNaN(bet) || bet < 1) throw new Error('Enter a bet of at least 1 DL.');
        if(Number.isNaN(cash) || cash < 1.01) throw new Error('Cashout goal must be at least 1.01×.');
        const res = await j('/api/crash/bet', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({bet, cashout: cash})});
        crMsgEl().textContent = 'Bet placed for this round.';
        renderHeader(); // update header balance
      }catch(e){ crMsgEl().textContent = 'Error: '+e.message; }
    };

    // Render other tabs
    async function renderOther(){
      await renderReferral();
      await renderPromos();
      await renderProfile();
      // My crash history
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
                    <td class="${r.win>0?'win':'lose'}">${r.win>0?fmtDL(r.win):'-'}</td>
                    <td>${r.xp_gain}</td>
                  </tr>`).join('')
              }</tbody></table>`
          : 'No recent rounds.';
      }catch(e){}
    }

    // Tab handlers
    tabGames.onclick=()=>setTab('games');
    tabRef.onclick=()=>{ setTab('ref'); renderReferral(); };
    tabPromo.onclick=()=>{ setTab('promo'); renderPromos(); };
    document.getElementById('openCrash').onclick=()=>{ setTab('crash'); };

    // Owner panel actions
    document.addEventListener('click', (e)=>{
      if(e.target && e.target.id==='tApply'){
        (async ()=>{
          const ident = document.getElementById('tIdent').value.trim();
          const amt = parseInt(document.getElementById('tAmt').value, 10);
          const reason = document.getElementById('tReason').value.trim();
          const msg = document.getElementById('tMsg'); msg.textContent='';
          try{
            if(!ident) throw new Error('Enter a user id or <@mention>');
            if(Number.isNaN(amt) || amt === 0) throw new Error('Amount must be non-zero');
            const res = await j('/api/admin/adjust', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({identifier: ident, amount: amt, reason})});
            msg.textContent = 'OK — new balance: ' + fmtDL(res.new_balance);
            renderHeader();
          }catch(err){ msg.textContent = 'Error: ' + err.message; }
        })();
      }
      if(e.target && e.target.id==='cMake'){
        (async ()=>{
          const c = document.getElementById('cCode').value.trim();
          const a = parseInt(document.getElementById('cAmount').value, 10);
          const m = parseInt(document.getElementById('cMax').value, 10);
          const msg = document.getElementById('cMsg'); msg.textContent='';
          try{
            if(Number.isNaN(a) || a==0) throw new Error('Amount must be non-zero');
            if(Number.isNaN(m) || m<1) throw new Error('Max uses must be >= 1');
            const res = await j('/api/admin/promo/create', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({code:c||null, amount:a, max_uses:m})});
            msg.textContent = 'Created code: ' + res.code;
          }catch(err){ msg.textContent = 'Error: ' + err.message; }
        })();
      }
    });

    // Promo redeem button
    document.getElementById('redeemBtn').onclick = async ()=>{
      const code = document.getElementById('promoInput').value.trim();
      const msg = document.getElementById('promoMsg'); msg.textContent='';
      if(!code){ msg.textContent='Enter a code.'; return; }
      try{
        const res = await j('/api/promo/redeem', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({code})});
        msg.textContent = 'Success! New balance: ' + fmtDL(res.new_balance);
        renderHeader();
      }catch(e){ msg.textContent = 'Error: ' + e.message; }
    };

    // Poll Crash state every 1s while on crash tab
    setInterval(()=>{ if(pgCrash.style.display!=='none') refreshCrash(); }, 1000);

    // Initial setup
    renderHeader();
    renderOther();
    refreshCrash();
  </script>
</body>
</html>
""".replace("${OWNER_ID}", str(OWNER_ID))

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

# ----- Profiles & Referrals -----
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

# ----- Promos -----
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

# ----- Owner admin -----
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

# ----- Multiplayer Crash API -----
class BetBody(BaseModel):
    bet: int
    cashout: float

@app.get("/api/crash/state")
async def api_crash_state(request: Request):
    user = read_session(request)
    if not user: raise HTTPException(401, "Not logged in")

    r = load_round()
    rid, phase = None, "betting"
    now = datetime.datetime.utcnow().isoformat() + "Z"
    if r:
        rid = r["id"]; phase = r["status"]
    yb = your_bet(rid, user["id"]) if rid else None
    return {
        "phase": phase,
        "round_id": rid,
        "betting_opens_at": r["betting_opens_at"].isoformat()+"Z" if r and r["betting_opens_at"] else None,
        "betting_ends_at": r["betting_ends_at"].isoformat()+"Z" if r and r["betting_ends_at"] else None,
        "started_at": r["started_at"].isoformat()+"Z" if r and r["started_at"] else None,
        "expected_end_at": r["expected_end_at"].isoformat()+"Z" if r and r["expected_end_at"] else None,
        "bust": r["bust"] if r and r["bust"] is not None else None,
        "now": now,
        "your_bet": yb,
        "min_bet": MIN_BET,
        "last_busts": last_busts(15)
    }

@app.post("/api/crash/bet")
async def api_crash_bet(request: Request, body: BetBody):
    user = read_session(request)
    if not user: raise HTTPException(401, "Not logged in")
    bet = int(body.bet); cash = float(body.cashout)
    if bet < MIN_BET: raise HTTPException(400, f"Min bet is {MIN_BET} DL")
    if bet > MAX_BET: raise HTTPException(400, f"Max bet is {MAX_BET} DL")
    if cash < 1.01: raise HTTPException(400, "Cashout must be at least 1.01×")
    try:
        res = place_bet(user["id"], bet, cash)
        return {"ok": True, "round_id": res["round_id"]}
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.get("/api/game/crash/history")
async def api_game_crash_history(request: Request, limit: int = Query(10, ge=1, le=50)):
    user = read_session(request)
    if not user: raise HTTPException(401, "Not logged in")
    return {"rows": your_history(user["id"], limit)}

@app.get("/health")
async def health():
    return {"ok": True}

# ---------- Crash round loop ----------
async def crash_loop():
    # Keep shared rounds going forever
    while True:
        # ensure a round exists
        rid, r = ensure_betting_round()
        now = datetime.datetime.utcnow()

        if r["status"] == "betting":
            # wait until betting end or trigger start if past
            wait = (r["betting_ends_at"] - now).total_seconds()
            if wait > 0:
                await asyncio.sleep(min(wait, 1.0))  # small ticks to re-check
            else:
                # move to running
                begin = begin_running(rid)
                if begin:
                    # resolve immediately (credit winners), but keep UI "running" until expected_end_at
                    resolve_bets(rid, begin["bust"])
        elif r["status"] == "running":
            # sleep until expected_end_at, then mark ended and create next round
            if r["expected_end_at"]:
                wait = (r["expected_end_at"] - now).total_seconds()
                if wait > 0:
                    await asyncio.sleep(min(wait, 1.0))
                else:
                    finish_round(rid)
                    create_next_betting()
                    await asyncio.sleep(0.5)
            else:
                # safety: if expected_end_at missing, end immediately
                finish_round(rid)
                create_next_betting()
                await asyncio.sleep(0.5)
        else:  # ended
            create_next_betting()
            await asyncio.sleep(0.5)

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
