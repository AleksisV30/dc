import os, json, asyncio, re, random, string
from urllib.parse import urlencode
from typing import Optional

import httpx
import psycopg
import discord
from discord.ext import commands
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from itsdangerous import URLSafeSerializer, BadSignature
import uvicorn
from pydantic import BaseModel, Field

# ---------- Config ----------
PREFIX = "."
BOT_TOKEN = os.getenv("DISCORD_TOKEN")
CLIENT_ID = os.getenv("DISCORD_CLIENT_ID")
CLIENT_SECRET = os.getenv("DISCORD_CLIENT_SECRET")
OAUTH_REDIRECT = os.getenv("OAUTH_REDIRECT")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")
PORT = int(os.getenv("PORT", "8080"))
DISCORD_API = "https://discord.com/api"
OWNER_ID = 1128658280546320426
GUILD_ID = int(os.getenv("GUILD_ID", "0"))
DATABASE_URL = os.getenv("DATABASE_URL")

GEM = "💎"

# ---------- DB (Postgres helpers) ----------
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
    # logs
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
    # promo codes
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
    # redemptions (1 per user per code)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS promo_redemptions (
            user_id TEXT NOT NULL,
            code TEXT NOT NULL,
            redeemed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(user_id, code)
        )
    """)
    # profiles (unique referral name)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS profiles (
            user_id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            name_lower TEXT NOT NULL UNIQUE,
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
def get_top_balances(cur, limit: int = 20):
    cur.execute("SELECT user_id, balance FROM balances ORDER BY balance DESC LIMIT %s", (limit,))
    return [{"user_id": r[0], "balance": int(r[1])} for r in cur.fetchall()]

@with_conn
def get_recent_logs(cur, limit: int = 20):
    cur.execute("""
        SELECT actor_id, target_id, amount, reason, created_at
        FROM balance_log
        ORDER BY id DESC
        LIMIT %s
    """, (limit,))
    rows = cur.fetchall()
    return [{"actor_id":r[0], "target_id":r[1], "amount":int(r[2]), "reason":r[3], "created_at":str(r[4])} for r in rows]

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
    # apply
    cur.execute("INSERT INTO balances (user_id, balance) VALUES (%s, 0) ON CONFLICT (user_id) DO NOTHING", (user_id,))
    cur.execute("UPDATE balances SET balance = balance + %s WHERE user_id = %s", (amount, user_id))
    cur.execute("UPDATE promo_codes SET uses = uses + 1 WHERE code = %s", (code,))
    cur.execute("INSERT INTO promo_redemptions(user_id, code) VALUES (%s, %s)", (user_id, code))
    cur.execute("INSERT INTO balance_log(actor_id, target_id, amount, reason) VALUES (%s, %s, %s, %s)",
                ("promo", user_id, amount, f"promo:{code}"))
    cur.execute("SELECT balance FROM balances WHERE user_id = %s", (user_id,))
    return int(cur.fetchone()[0])

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

# ---- Profiles (referral name) ----
NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{3,20}$")

@with_conn
def get_profile(cur, user_id: str):
    cur.execute("SELECT display_name FROM profiles WHERE user_id = %s", (user_id,))
    row = cur.fetchone()
    return row[0] if row else None

@with_conn
def set_profile_name(cur, user_id: str, name: str):
    if not NAME_RE.match(name):
        raise ValueError("Name must be 3-20 chars [a-zA-Z0-9_-]")
    name_lower = name.lower()
    # unique (case-insensitive)
    cur.execute("SELECT user_id FROM profiles WHERE name_lower = %s AND user_id <> %s", (name_lower, user_id))
    if cur.fetchone():
        raise ValueError("Name is already taken")
    cur.execute("""
        INSERT INTO profiles(user_id, display_name, name_lower)
        VALUES (%s, %s, %s)
        ON CONFLICT (user_id)
        DO UPDATE SET display_name = EXCLUDED.display_name, name_lower = EXCLUDED.name_lower
    """, (user_id, name, name_lower))
    return {"ok": True, "name": name}

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
    # Return a working Discord CDN avatar URL
    if avatar_hash:
        return f"https://cdn.discordapp.com/avatars/{id_str}/{avatar_hash}.png?size=64"
    # default avatar sprite 0..5
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

# ---------- Web (FastAPI) ----------
app = FastAPI()
signer = URLSafeSerializer(SECRET_KEY, salt="session")

def set_session(resp: RedirectResponse, payload: dict):
    resp.set_cookie("session", signer.dumps(payload), httponly=True, samesite="lax", max_age=7*24*3600)

def read_session(request: Request) -> dict | None:
    raw = request.cookies.get("session")
    if not raw: return None
    try: return signer.loads(raw)
    except BadSignature: return None

# ---- HTML (same dark style; header with tabs; owner promo creator) ----
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
    .chip{background:#0c1631; border:1px solid var(--border); color:#dce7ff; padding:6px 10px; border-radius:999px; font-size:12px; white-space:nowrap}
    .avatar{width:28px;height:28px;border-radius:50%;object-fit:cover;border:1px solid var(--border)}
    .btn{display:inline-flex; align-items:center; gap:8px; padding:8px 12px; border-radius:10px; border:1px solid var(--border); background:#0f1a33; cursor:pointer}
    .btn.primary{background:linear-gradient(135deg,#3b82f6,#22c1dc); border-color:transparent}
    .grid{display:grid; gap:16px; grid-template-columns:1fr}
    @media(min-width:900px){.grid{grid-template-columns:1fr 1fr}}
    .card{background:var(--card); border:1px solid var(--border); border-radius:16px; padding:16px}
    .label{color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.1em}
    .big{font-size:28px; font-weight:800}
    table{width:100%; border-collapse:collapse; margin-top:10px}
    th,td{border-bottom:1px solid var(--border); padding:8px 6px; text-align:left}
    .muted{color:var(--muted)}
    input{width:100%; background:#0e1833; color:var(--text); border:1px solid var(--border); border-radius:10px; padding:10px}
    .games-grid{display:grid; gap:12px; grid-template-columns:1fr; margin-top:16px}
    @media(min-width:800px){.games-grid{grid-template-columns:repeat(3,1fr)}}
    .game{background:#0f1a33;border:1px solid var(--border);border-radius:16px;padding:14px}
    .banner{font-weight:900; font-size:18px}
    .owner{margin-top:16px; border-top:1px dashed var(--border); padding-top:12px}
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
    <div id="page-games">
      <div class="card">
        <div class="label">Games</div>
        <div class="games-grid">
          <div class="game"><div class="big">🎰 Lucky Spin</div><div class="muted">Coming soon.</div></div>
          <div class="game"><div class="big">🎯 Coin Flip</div><div class="muted">Coming soon.</div></div>
          <div class="game"><div class="big">🃏 Blackjack</div><div class="muted">Coming soon.</div></div>
        </div>
      </div>
    </div>

    <div id="page-ref" style="display:none">
      <div class="card">
        <div class="label">Referral</div>
        <div id="refContent">Loading…</div>
      </div>
    </div>

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

        <div id="ownerPromoBox" class="owner" style="display:none">
          <div class="label">Owner — Create Promo Code</div>
          <div class="grid" style="grid-template-columns:1fr 1fr 1fr; gap:8px">
            <div><div class="label">Code (optional)</div><input id="cCode" placeholder="auto-generate if empty"/></div>
            <div><div class="label">Amount (DL)</div><input id="cAmount" type="number" placeholder="e.g. 10"/></div>
            <div><div class="label">Max Uses</div><input id="cMax" type="number" placeholder="e.g. 100"/></div>
          </div>
          <div style="margin-top:8px"><button class="btn primary" id="cMake">Create</button> <span id="cMsg" class="muted"></span></div>
        </div>
      </div>
    </div>

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

  <script>
    function qs(id){return document.getElementById(id)}
    const tabGames = qs('tab-games'), tabRef=qs('tab-ref'), tabPromo=qs('tab-promo');
    const pgGames=qs('page-games'), pgRef=qs('page-ref'), pgPromo=qs('page-promo'), loginCard=qs('loginCard');

    function fmtDL(n){ return `💎 ${Number(n).toLocaleString()} DL`; }
    async function j(u, opt={}){ const r=await fetch(u, Object.assign({credentials:'include'},opt)); if(!r.ok) throw new Error(await r.text()); return r.json(); }

    function setTab(which){
      [tabGames,tabRef,tabPromo].forEach(t=>t.classList.remove('active'));
      [pgGames,pgRef,pgPromo].forEach(p=>p.style.display='none');
      if(which==='games'){ tabGames.classList.add('active'); pgGames.style.display=''; }
      if(which==='ref'){ tabRef.classList.add('active'); pgRef.style.display=''; }
      if(which==='promo'){ tabPromo.classList.add('active'); pgPromo.style.display=''; }
      window.scrollTo({top:0, behavior:'smooth'});
    }
    tabGames.onclick=()=>setTab('games');
    tabRef.onclick=()=>setTab('ref');
    tabPromo.onclick=()=>setTab('promo');
    qs('homeLink').onclick=(e)=>{ e.preventDefault(); setTab('games'); };

    qs('registerBtn')?.addEventListener('click', ()=> {
      const info = qs('registerInfo');
      info.style.display = info.style.display ? '' : 'block';
    });

    function safeAvatar(me){
      return me.avatar_url || '';
    }

    async function render(){
      const auth = qs('authArea');
      try{
        const me = await j('/api/me');          // now includes avatar_url
        const bal = await j('/api/balance');
        auth.innerHTML = `
          <span class="chip">${fmtDL(bal.balance)}</span>
          <img class="avatar" src="${safeAvatar(me)}" title="${me.username}" onerror="this.style.display='none'">
        `;
        loginCard.style.display='none';

        // REFERRAL
        const ref = await j('/api/referral/state');
        if(ref.name){
          const base = location.origin;
          const link = base + '/?ref=' + encodeURIComponent(ref.name);
          qs('refContent').innerHTML = \`
            <p>Your referral name: <b>\${ref.name}</b></p>
            <p>Share this link:</p>
            <p><code>\${link}</code></p>
          \`;
        } else {
          qs('refContent').innerHTML = \`
            <p>Claim a referral name to get your link.</p>
            <div style="display:flex; gap:8px; align-items:center; max-width:420px">
              <input id="refName" placeholder="3-20 letters/numbers/_-" />
              <button class="btn primary" id="claimBtn">Claim</button>
            </div>
            <div id="refMsg" class="muted" style="margin-top:8px"></div>
          \`;
          qs('claimBtn').onclick = async ()=>{
            const name = qs('refName').value.trim();
            const msg = qs('refMsg');
            msg.textContent='';
            try{
              const r = await j('/api/profile/set_name', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({name})});
              msg.textContent = 'Saved. Your link: ' + location.origin + '/?ref=' + encodeURIComponent(r.name);
              render();
            }catch(e){ msg.textContent = 'Error: ' + e.message; }
          };
        }

        // PROMO: my redemptions
        const mine = await j('/api/promo/my');
        qs('myCodes').innerHTML = mine.rows.length
          ? '<ul>' + mine.rows.map(r=>\`<li><code>\${r.code}</code> — \${new Date(r.redeemed_at).toLocaleString()}</li>\`).join('') + '</ul>'
          : 'No redemptions yet.';

        // Owner promo creator
        if (me.id === 'OWNER_PLACEHOLDER') {
          const box = qs('ownerPromoBox'); box.style.display='';
          qs('cMake').onclick = async ()=>{
            const c = qs('cCode').value.trim();
            const a = parseInt(qs('cAmount').value, 10);
            const m = parseInt(qs('cMax').value, 10);
            const msg = qs('cMsg'); msg.textContent='';
            try{
              if(Number.isNaN(a) || a === 0) throw new Error('Amount must be non-zero');
              if(Number.isNaN(m) || m < 1) throw new Error('Max uses must be >= 1');
              const res = await j('/api/admin/promo/create', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({code:c||null, amount:a, max_uses:m})});
              msg.textContent = 'Created code: ' + res.code;
              qs('cCode').value=''; qs('cAmount').value=''; qs('cMax').value='';
            }catch(e){ msg.textContent = 'Error: ' + e.message; }
          };
        } else {
          qs('ownerPromoBox').style.display='none';
        }

      }catch(e){
        // not logged in
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

    // redeem promo
    qs('redeemBtn').onclick = async ()=>{
      const code = qs('promoInput').value.trim();
      const msg = qs('promoMsg');
      msg.textContent = '';
      if(!code){ msg.textContent = 'Enter a code.'; return; }
      try{
        const res = await j('/api/promo/redeem', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({code})});
        msg.textContent = 'Success! New balance: ' + fmtDL(res.new_balance);
        qs('promoInput').value='';
        render();
      }catch(e){
        msg.textContent = 'Error: ' + e.message;
      }
    };

    render();
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
    if not (CLIENT_ID and OAUTH_REDIRECT):
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
    # store only essentials
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
    return {
        "id": user["id"],
        "username": user.get("username", ""),
        "avatar_url": avatar_url_from(user["id"], user.get("avatar"))
    }

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
    name = get_profile(str(user["id"]))
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

# ----- Owner: create promo -----
class CreatePromoBody(BaseModel):
    code: Optional[str] = None
    amount: int
    max_uses: int = 1
    expires_at: Optional[str] = None  # optional ISO string

def require_owner(request: Request):
    user = read_session(request)
    if not user: raise HTTPException(401, "Not logged in")
    if str(user["id"]) != str(OWNER_ID): raise HTTPException(403, "Owner only")
    return user

@app.post("/api/admin/promo/create")
async def api_admin_promo_create(request: Request, body: CreatePromoBody):
    require_owner(request)
    if body.amount == 0: raise HTTPException(400, "Amount cannot be zero")
    if body.max_uses < 1: raise HTTPException(400, "Max uses must be >= 1")
    return create_promo(str(OWNER_ID), body.code, int(body.amount), int(body.max_uses), body.expires_at)

# (member search endpoint kept from earlier, if you still use it elsewhere)
@app.get("/api/admin/members")
async def api_admin_members(request: Request, q: str = Query("", description="Search after @ (optional)")):
    require_owner(request)
    if not GUILD_ID: raise HTTPException(400, "GUILD_ID not set")
    guild = bot.get_guild(GUILD_ID)
    if not guild: raise HTTPException(400, "Guild not found or bot not in guild")
    members = []
    try:
        if q:
            try:
                found = await guild.search_members(q, limit=10)
            except Exception:
                found = []
        else:
            found = list(guild.members)[:10]
        for m in found:
            try: avatar = m.display_avatar.url
            except Exception: avatar = None
            members.append({"id": str(m.id), "username": m.name, "display_name": m.display_name, "avatar": avatar})
    except Exception:
        members = []
    return {"members": members}

@app.get("/health")
async def health():
    return {"ok": True}

# ---------- Runner (resilient) ----------
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
