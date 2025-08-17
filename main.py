import os, json, asyncio, re, random, string, math, secrets
from urllib.parse import urlencode
from typing import Optional

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
CLIENT_SECRET = os.getenv("CLIENT_SECRET") or os.getenv("DISCORD_CLIENT_SECRET")  # allow either name
OAUTH_REDIRECT = os.getenv("OAUTH_REDIRECT")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")
PORT = int(os.getenv("PORT", "8080"))
DISCORD_API = "https://discord.com/api"
OWNER_ID = 1128658280546320426
GUILD_ID = int(os.getenv("GUILD_ID", "0"))
DATABASE_URL = os.getenv("DATABASE_URL")

GEM = "💎"
HOUSE_EDGE = 0.05  # 1% house edge for Crash
MIN_BET = 1       # DL
MAX_BET = 1_000_000

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

@with_conn
def get_balance(cur, user_id: str) -> int:
    cur.execute("SELECT balance FROM balances WHERE user_id = %s", (user_id,))
    row = cur.fetchone()
    return int(row[0]) if row else 0

@with_conn
def add_balance(cur, user_id: str, amount: int) -> int:
    cur.execute("INSERT INTO balances (user_id, balance) VALUES (%s, 0) ON CONFLICT (user_id) DO NOTHING", (user_id,))
    cur.execute("UPDATE balances SET balance = balance + %s WHERE user_id = %s", (amount, user_id))
    cur.execute("SELECT balance FROM balances WHERE user_id = %s", (user_id,))
    return int(cur.fetchone()[0])

@with_conn
def adjust_balance(cur, actor_id: str, target_id: str, amount: int, reason: Optional[str]) -> int:
    cur.execute("INSERT INTO balances (user_id, balance) VALUES (%s, 0) ON CONFLICT (user_id) DO NOTHING", (target_id,))
    cur.execute("UPDATE balances SET balance = balance + %s WHERE user_id = %s", (amount, target_id))
    cur.execute("INSERT INTO balance_log(actor_id, target_id, amount, reason) VALUES (%s, %s, %s, %s)",
                (actor_id, target_id, amount, reason))
    cur.execute("SELECT balance FROM balances WHERE user_id = %s", (target_id,))
    return int(cur.fetchone()[0])

@with_conn
def get_profile(cur, user_id: str):
    cur.execute("SELECT display_name, xp FROM profiles WHERE user_id = %s", (user_id,))
    row = cur.fetchone()
    return {"display_name": row[0], "xp": int(row[1])} if row else None

@with_conn
def get_profile_name(cur, user_id: str):
    cur.execute("SELECT display_name FROM profiles WHERE user_id = %s", (user_id,))
    row = cur.fetchone()
    return row[0] if row else None

@with_conn
def ensure_profile_row(cur, user_id: str):
    cur.execute("INSERT INTO profiles(user_id, display_name, name_lower) VALUES (%s, %s, %s) ON CONFLICT (user_id) DO NOTHING",
                (user_id, f"user_{user_id[-4:]}", f"user_{user_id[-4:]}"))
    cur.execute("SELECT display_name, xp FROM profiles WHERE user_id = %s", (user_id,))
    r = cur.fetchone()
    return {"display_name": r[0], "xp": int(r[1])}

@with_conn
def add_xp(cur, user_id: str, amount: int):
    ensure_profile_row(user_id)
    cur.execute("UPDATE profiles SET xp = xp + %s WHERE user_id = %s", (amount, user_id))

@with_conn
def my_redemptions(cur, user_id: str):
    cur.execute("SELECT code, redeemed_at FROM promo_redemptions WHERE user_id=%s ORDER BY redeemed_at DESC LIMIT 50", (user_id,))
    return [{"code": r[0], "redeemed_at": str(r[1])} for r in cur.fetchall()]

def _rand_code(n=8):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=n))

@with_conn
def create_promo(cur, actor_id: str, code: Optional[str], amount: int, max_uses: int = 1, expires_at: Optional[str] = None):
    code = (code.strip().upper() if code else _rand_code())
    cur.execute("""
        INSERT INTO promo_codes(code, amount, max_uses, expires_at, created_by)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (code) DO UPDATE SET amount=EXCLUDED.amount, max_uses=EXCLUDED.max_uses, expires_at=EXCLUDED.expires_at
    """, (code, amount, max_uses, expires_at, actor_id))
    return {"ok": True, "code": code}

# ---- Promo helpers ----
class PromoError(Exception): ...
class PromoAlreadyRedeemed(PromoError): ...
class PromoInvalid(PromoError): ...
class PromoExpired(PromoError): ...
class PromoExhausted(PromoError): ...

@with_conn
def redeem_promo(cur, user_id: str, code: str) -> int:
    code = code.strip().upper()
    cur.execute("SELECT code, amount, max_uses, uses, expires_at FROM promo_codes WHERE code = %s", (code,))
    row = cur.fetchone()
    if not row:
        raise PromoInvalid("Invalid code")
    _, amount, max_uses, uses, expires_at = row
    if expires_at is not None:
        cur.execute("SELECT NOW() > %s", (expires_at,))
        if cur.fetchone()[0]:
            raise PromoExpired("Code expired")
    if uses >= max_uses:
        raise PromoExhausted("Code maxed out")
    cur.execute("SELECT 1 FROM promo_redemptions WHERE user_id=%s AND code=%s", (user_id, code))
    if cur.fetchone():
        raise PromoAlreadyRedeemed("You already redeemed this code")
    cur.execute("INSERT INTO balances (user_id, balance) VALUES (%s, 0) ON CONFLICT (user_id) DO NOTHING", (user_id,))
    cur.execute("UPDATE balances SET balance = balance + %s WHERE user_id = %s", (amount, user_id))
    cur.execute("UPDATE promo_codes SET uses = uses + 1 WHERE code = %s", (code,))
    cur.execute("INSERT INTO promo_redemptions(user_id, code) VALUES (%s, %s)", (user_id, code))
    cur.execute("INSERT INTO balance_log(actor_id, target_id, amount, reason) VALUES (%s, %s, %s, %s)",
                ("promo", user_id, amount, f"promo:{code}"))
    cur.execute("SELECT balance FROM balances WHERE user_id = %s", (user_id,))
    return int(cur.fetchone()[0])

# ---- Crash game logic ----
def _rand_unit():
    n = secrets.randbelow(1_000_000_000) + 1  # 1..1e9
    return n / 1_000_000_001.0

def gen_crash_bust(edge: float = HOUSE_EDGE) -> float:
    # P(B >= x) = (1 - edge) / x, x >= 1
    u = _rand_unit()
    B = max(1.0, (1.0 - edge) / u)
    return math.floor(B * 100.0) / 100.0

@with_conn
def crash_play_db(cur, user_id: str, bet: int, cashout: float, edge: float = HOUSE_EDGE):
    cur.execute("INSERT INTO balances (user_id, balance) VALUES (%s, 0) ON CONFLICT (user_id) DO NOTHING", (user_id,))
    cur.execute("SELECT balance FROM balances WHERE user_id=%s FOR UPDATE", (user_id,))
    row = cur.fetchone()
    bal = int(row[0]) if row else 0
    if bet < MIN_BET:
        raise ValueError(f"Min bet is {MIN_BET} DL")
    if bet > MAX_BET:
        raise ValueError(f"Max bet is {MAX_BET} DL")
    if bal < bet:
        raise ValueError("Insufficient balance")

    # take bet
    cur.execute("UPDATE balances SET balance = balance - %s WHERE user_id = %s", (bet, user_id))

    bust = gen_crash_bust(edge)
    win = 0
    if cashout <= bust:
        win = int(math.floor(bet * cashout + 1e-9))
        cur.execute("UPDATE balances SET balance = balance + %s WHERE user_id = %s", (win, user_id))

    xp_gain = max(1, min(bet, 50))
    ensure_profile_row(user_id)
    cur.execute("UPDATE profiles SET xp = xp + %s WHERE user_id = %s", (xp_gain, user_id))

    cur.execute("""
        INSERT INTO crash_games(user_id, bet, cashout, bust, win, xp_gain)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id
    """, (user_id, bet, float(cashout), float(bust), win, xp_gain))
    game_id = int(cur.fetchone()[0])

    cur.execute("SELECT balance FROM balances WHERE user_id=%s", (user_id,))
    new_bal = int(cur.fetchone()[0])

    return {"id": game_id, "bust": float(bust), "win": int(win), "net": int(win - bet), "new_balance": new_bal, "xp_gain": int(xp_gain)}

@with_conn
def crash_history(cur, user_id: str, limit: int = 10):
    cur.execute("""
        SELECT id, bet, cashout, bust, win, xp_gain, created_at
        FROM crash_games
        WHERE user_id=%s
        ORDER BY id DESC
        LIMIT %s
    """, (user_id, limit))
    rows = cur.fetchall()
    return [{
        "id": int(r[0]),
        "bet": int(r[1]),
        "cashout": float(r[2]),
        "bust": float(r[3]),
        "win": int(r[4]),
        "xp_gain": int(r[5]),
        "created_at": str(r[6]),
    } for r in rows]

# ---------- Helpers ----------
def parse_user_identifier(identifier: str) -> Optional[str]:
    if not identifier: return None
    cleaned = identifier.strip().replace("<@!", "").replace("<@", "").replace(">", "")
    if cleaned.isdigit() and len(cleaned) >= 17:
        return cleaned
    return None

def fmt_dl(n: int) -> str:
    return f"{GEM} {n:,} DL"

def embed(title: str, desc: Optional[str] = None, color: int = 0x00C2FF) -> discord.Embed:
    return discord.Embed(title=title, description=desc or "", color=color)

def avatar_url_from(id_str: str, avatar_hash: Optional[str]) -> str:
    if avatar_hash:
        return f"https://cdn.discordapp.com/avatars/{id_str}/{avatar_hash}.png?size=64"
    idx = int(id_str) % 6
    return f"https://cdn.discordapp.com/embed/avatars/{idx}.png?size=64"

# ---------- Discord bot ----------
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

@bot.command(name="bal")
async def bal(ctx: commands.Context, user: discord.User | None = None):
    target = user or ctx.author
    bal_value = get_balance(str(target.id))
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

def read_session(request: Request) -> dict | None:
    raw = request.cookies.get("session")
    if not raw: return None
    try: return signer.loads(raw)
    except BadSignature: return None

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

    /* Modal */
    .modal{position:fixed; inset:0; background:rgba(0,0,0,.6); display:none; align-items:center; justify-content:center; padding:20px}
    .modal .box{background:#0f1a33; border:1px solid var(--border); border-radius:16px; padding:16px; max-width:520px; width:100%}
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
            <div class="muted">Click to play. Set a cashout goal, watch the multiplier climb… don’t get busted!</div>
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
          <div class="big">🚀 Crash</div>
          <div class="cr-small">House edge 1% • Min bet 1 DL</div>
        </div>
        <div class="crash-wrap">
          <div class="cr-row">
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
            <div class="cr-multi" id="crNow">1.00×</div>
            <div class="cr-small" id="crHint">Set your bet and goal, then press Play.</div>
          </div>

          <div class="cr-line">
            <div class="cr-fill" id="crFill"></div>
            <div class="cr-marker" id="crCashMarker" style="display:none"></div>
            <div class="cr-marker bust" id="crBustMarker" style="display:none"></div>
          </div>

          <div style="display:flex; gap:8px; align-items:center">
            <button class="btn primary" id="crPlay">Play</button>
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

  <!-- Modal -->
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
    tabGames.onclick=()=>setTab('games');
    tabRef.onclick=()=>setTab('ref');
    tabPromo.onclick=()=>setTab('promo');
    qs('homeLink').onclick=(e)=>{ e.preventDefault(); setTab('games'); };
    qs('openCrash').onclick=()=>setTab('crash');

    qs('registerBtn')?.addEventListener('click', ()=> {
      const info = qs('registerInfo');
      info.style.display = info.style.display ? '' : 'block';
    });

    // modal
    function openModal(){ qs('modal').style.display='flex'; }
    function closeModal(){ qs('modal').style.display='none'; }
    qs('mClose').onclick = closeModal;
    qs('modal').addEventListener('click', (e)=>{ if(e.target.id==='modal') closeModal(); });

    function safeAvatar(me){ return me.avatar_url || ''; }

    async function render(){
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

        // Referral content
        const refState = await j('/api/referral/state');
        if(refState.name){
          const link = location.origin + '/?ref=' + encodeURIComponent(refState.name);
          qs('refContent').innerHTML = `
            <p>Your referral name: <b>${refState.name}</b></p>
            <p>Share this link:</p>
            <p><code>${link}</code></p>
          `;
        } else {
          qs('refContent').innerHTML = `
            <p>Claim a referral name to get your link.</p>
            <div style="display:flex; gap:8px; align-items:center; max-width:420px">
              <input id="refName" placeholder="3-20 letters/numbers/_-" />
              <button class="btn primary" id="claimBtn">Claim</button>
            </div>
            <div id="refMsg" class="muted" style="margin-top:8px"></div>
          `;
          qs('claimBtn').onclick = async ()=>{
            const name = qs('refName').value.trim();
            const msg = qs('refMsg'); msg.textContent='';
            try{
              const r = await j('/api/profile/set_name', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({name})});
              msg.textContent = 'Saved. Your link: ' + location.origin + '/?ref=' + encodeURIComponent(r.name);
            }catch(e){ msg.textContent = 'Error: ' + e.message; }
          };
        }

        // Promo redemptions
        const mine = await j('/api/promo/my');
        qs('myCodes').innerHTML = mine.rows.length
          ? '<ul>' + mine.rows.map(r=>`<li><code>${r.code}</code> — ${new Date(r.redeemed_at).toLocaleString()}</li>`).join('') + '</ul>'
          : 'No redemptions yet.';

        // Profile box
        const prof = await j('/api/profile');
        const lvl = prof.level, pct = Math.max(0, Math.min(100, prof.progress_pct));
        qs('profileBox').innerHTML = `
          <div style="display:flex; gap:12px; align-items:center; flex-wrap:wrap">
            <img src="${safeAvatar(me)}" class="avatar" style="width:56px; height:56px">
            <div>
              <div class="big">${me.username}</div>
              <div class="muted">ID: ${me.id}</div>
            </div>
            <div style="margin-left:auto; display:flex; gap:8px; align-items:center">
              <a class="btn" href="/logout">Logout</a>
            </div>
          </div>
          <div class="grid" style="margin-top:12px">
            <div class="card" style="padding:12px">
              <div class="label">Balance</div>
              <div class="big">${fmtDL(prof.balance)}</div>
              <div class="muted" style="margin-top:6px">Click your balance in the header for Deposit/Withdraw instructions.</div>
            </div>
            <div class="card" style="padding:12px">
              <div class="label">Level</div>
              <div><b>Level ${lvl}</b> — XP ${prof.xp} / ${((lvl-1)*100)+100}</div>
              <div style="height:10px; background:#0e1833; border:1px solid var(--border); border-radius:999px; overflow:hidden; margin-top:8px">
                <div style="height:100%; width:${pct}%; background:linear-gradient(90deg,#22c1dc,#3b82f6)"></div>
              </div>
              <div class="muted" style="margin-top:6px">${prof.progress}/${prof.next_needed} XP to next level</div>
            </div>
          </div>
        `;

        // Owner panel
        if (me.id === 'OWNER_PLACEHOLDER') {
          qs('ownerPanel').style.display='';
          qs('tApply').onclick = async ()=>{
            const ident = qs('tIdent').value.trim();
            const amt = parseInt(qs('tAmt').value, 10);
            const reason = qs('tReason').value.trim();
            const msg = qs('tMsg'); msg.textContent='';
            try{
              if(!ident) throw new Error('Enter a user id or <@mention>');
              if(Number.isNaN(amt) || amt === 0) throw new Error('Amount must be non-zero');
              const res = await j('/api/admin/adjust', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({identifier: ident, amount: amt, reason})});
              msg.textContent = 'OK — new balance: ' + fmtDL(res.new_balance);
              qs('tAmt').value=''; qs('tReason').value='';
              try { render(); } catch(e){}
            }catch(e){ msg.textContent = 'Error: ' + e.message; }
          };
          qs('cMake').onclick = async ()=>{
            const c = qs('cCode').value.trim();
            const a = parseInt(qs('cAmount').value, 10);
            const m = parseInt(qs('cMax').value, 10);
            const msg = qs('cMsg'); msg.textContent='';
            try{
              if(Number.isNaN(a) || a == 0) throw new Error('Amount must be non-zero');
              if(Number.isNaN(m) || m < 1) throw new Error('Max uses must be >= 1');
              const res = await j('/api/admin/promo/create', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({code:c||null, amount:a, max_uses:m})});
              msg.textContent = 'Created code: ' + res.code;
              qs('cCode').value=''; qs('cAmount').value=''; qs('cMax').value='';
            }catch(e){ msg.textContent = 'Error: ' + e.message; }
          };
        } else {
          qs('ownerPanel').style.display='none';
        }

        // Crash history on enter crash page
        async function loadCrashHistory(){
          try{
            const h = await j('/api/game/crash/history');
            qs('crLast').innerHTML = h.rows.length
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
          }catch(e){ qs('crLast').textContent = 'Unable to load history.'; }
        }

        // Crash play with animated multiplier line
        const crPlayBtn = qs('crPlay'), crMsg = qs('crMsg');
        const crNow = qs('crNow'), crHint = qs('crHint');
        const crFill = qs('crFill'), cashMarker = qs('crCashMarker'), bustMarker = qs('crBustMarker');

        function resetLine(){
          crFill.style.width = '0%';
          crNow.textContent = '1.00×';
          crHint.textContent = 'Set your bet and goal, then press Play.';
          cashMarker.style.display='none';
          bustMarker.style.display='none';
        }
        resetLine();

        crPlayBtn.onclick = async ()=>{
          crMsg.textContent='';
          const bet = parseInt(qs('crBet').value, 10);
          const cash = parseFloat(qs('crCash').value);
          if(Number.isNaN(bet) || bet < 1){ crMsg.textContent = 'Enter a bet of at least 1 DL.'; return; }
          if(Number.isNaN(cash) || cash < 1.01){ crMsg.textContent = 'Cashout goal must be at least 1.01×.'; return; }

          crPlayBtn.disabled = true;
          crHint.textContent = 'Round in progress…';
          try{
            const res = await j('/api/game/crash/play', {
              method:'POST', headers:{'Content-Type':'application/json'},
              body: JSON.stringify({bet, cashout: cash})
            });

            // Animate 1.00× -> bust (accelerating)
            const bust = res.bust;
            const dur = Math.min(7.0, 1.2 + Math.log(bust + 1) * 1.6); //  ~1.2s..7s
            const start = performance.now();
            const lnBust = Math.log(bust);
            // markers
            cashMarker.style.display = 'block';
            bustMarker.style.display = 'block';
            const cashFrac = Math.min(1, Math.log(Math.max(1.0001, cash)) / lnBust);
            cashMarker.style.left = (cashFrac * 100) + '%';
            bustMarker.style.left = '100%';

            function frame(now){
              const t = Math.min(1, (now - start)/ (dur*1000));
              // exponential growth to exactly hit bust at t=1
              const m = Math.exp(lnBust * t); // 1 -> bust
              crNow.textContent = m.toFixed(2) + '×';
              const fillPct = (Math.log(m) / lnBust) * 100;
              crFill.style.width = fillPct + '%';
              if(t < 1) requestAnimationFrame(frame);
              else {
                // final outcome
                const won = res.win > 0;
                crHint.textContent = won ? 'You cashed out before bust!' : 'Busted!';
                crMsg.innerHTML = (won
                  ? `<span class="win">You won ${fmtDL(res.win)}!</span>`
                  : `<span class="lose">Busted at ${bust.toFixed(2)}× — you lost.</span>`
                ) + ` New balance: <b>${fmtDL(res.new_balance)}</b> (+${res.xp_gain} XP)`;
                // refresh header/profile and history
                try { render(); } catch (e) {}
                loadCrashHistory();
                crPlayBtn.disabled = false;
              }
            }
            requestAnimationFrame(frame);

          }catch(e){
            crPlayBtn.disabled = false;
            crMsg.textContent = 'Error: ' + e.message;
            resetLine();
          }
        };

        // load crash history once when visiting crash
        loadCrashHistory();

      }catch(e){
        // not logged in
        const auth = qs('authArea');
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

    // Initial render
    render();

    // Click avatar -> profile, balance -> modal handled inside render
  </script>
</body>
</html>
""".replace("OWNER_PLACEHOLDER", str(OWNER_ID))

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

# ----- Referral / Profiles -----
class SetNameBody(BaseModel):
    name: str

@app.get("/api/referral/state")
async def api_ref_state(request: Request):
    user = read_session(request)
    if not user: raise HTTPException(401, "Not logged in")
    name = get_profile_name(str(user["id"]))
    return {"name": name}

@app.post("/api/profile/set_name")
async def api_set_name(request: Request, body: SetNameBody):
    user = read_session(request)
    if not user: raise HTTPException(401, "Not logged in")
    try:
        res = set_profile_name(str(user["id"]), body.name)
        return res
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.get("/api/profile")
async def api_profile(request: Request):
    user = read_session(request)
    if not user: raise HTTPException(401, "Not logged in")
    uid = str(user["id"])
    info = ensure_profile_row(uid)
    xp = info["xp"]
    level = 1 + (xp // 100)
    base = (level - 1) * 100
    next_at = level * 100
    progress = xp - base
    pct = 0 if next_at == base else int(progress * 100 / (next_at - base))
    return {
        "user_id": uid,
        "display_name": info["display_name"],
        "xp": xp,
        "level": level,
        "progress": progress,
        "next_needed": (next_at - base),
        "progress_pct": pct,
        "balance": get_balance(uid),
    }

# ----- Promo endpoints -----
class RedeemBody(BaseModel):
    code: str

@app.get("/api/promo/my")
async def api_promo_my(request: Request):
    user = read_session(request)
    if not user: raise HTTPException(401, "Not logged in")
    return {"rows": my_redemptions(str(user["id"]))}

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

# ----- Owner: adjust balance & create promo -----
class AdjustBody(BaseModel):
    identifier: str
    amount: int
    reason: Optional[str] = None

class CreatePromoBody(BaseModel):
    code: Optional[str] = None
    amount: int
    max_uses: int = 1
    expires_at: Optional[str] = None

def require_owner(request: Request):
    user = read_session(request)
    if not user: raise HTTPException(401, "Not logged in")
    if str(user["id"]) != str(OWNER_ID): raise HTTPException(403, "Owner only")
    return user

@app.post("/api/admin/adjust")
async def api_admin_adjust(request: Request, body: AdjustBody):
    actor = require_owner(request)
    user_id = parse_user_identifier(body.identifier)
    if not user_id: raise HTTPException(400, "Invalid identifier (use raw ID or <@mention>)")
    if body.amount == 0: raise HTTPException(400, "Amount cannot be zero")
    new_balance = adjust_balance(str(actor["id"]), user_id, int(body.amount), body.reason)
    return {"user_id": user_id, "new_balance": new_balance}

@app.post("/api/admin/promo/create")
async def api_admin_promo_create(request: Request, body: CreatePromoBody):
    require_owner(request)
    if body.amount == 0: raise HTTPException(400, "Amount cannot be zero")
    if body.max_uses < 1: raise HTTPException(400, "Max uses must be >= 1")
    return create_promo(str(OWNER_ID), body.code, int(body.amount), int(body.max_uses), body.expires_at)

# ----- Crash endpoints -----
class CrashPlayBody(BaseModel):
    bet: int
    cashout: float

@app.post("/api/game/crash/play")
async def api_game_crash_play(request: Request, body: CrashPlayBody):
    user = read_session(request)
    if not user: raise HTTPException(401, "Not logged in")
    bet = int(body.bet)
    cash = float(body.cashout)
    if bet < MIN_BET: raise HTTPException(400, f"Min bet is {MIN_BET} DL")
    if bet > MAX_BET: raise HTTPException(400, f"Max bet is {MAX_BET} DL")
    if cash < 1.01: raise HTTPException(400, "Cashout must be at least 1.01×")
    try:
        return crash_play_db(str(user["id"]), bet, cash, HOUSE_EDGE)
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.get("/api/game/crash/history")
async def api_game_crash_history(request: Request, limit: int = Query(10, ge=1, le=50)):
    user = read_session(request)
    if not user: raise HTTPException(401, "Not logged in")
    return {"rows": crash_history(str(user["id"]), limit)}

@app.get("/health")
async def health():
    return {"ok": True}

# ---------- Runner ----------
async def main():
    import traceback, sys
    init_db()

    config = uvicorn.Config(app, host="0.0.0.0", port=PORT, log_level="info")
    server = uvicorn.Server(config)

    async def run_bot_forever():
        while True:
            try:
                if not BOT_TOKEN:
                    raise RuntimeError("DISCORD_TOKEN env var not set.")
                await bot.start(BOT_TOKEN)
            except discord.errors.LoginFailure:
                print("[bot] LoginFailure: bad token. Fix DISCORD_TOKEN.", file=sys.stderr)
                await asyncio.sleep(3600)
            except Exception:
                traceback.print_exc()
                await asyncio.sleep(10)

    await asyncio.gather(server.serve(), run_bot_forever())

if __name__ == "__main__":
    asyncio.run(main())
