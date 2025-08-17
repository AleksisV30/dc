import os, json, asyncio
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
OAUTH_REDIRECT = os.getenv("OAUTH_REDIRECT")   # e.g. https://your-app.up.railway.app/callback
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
    # admin change log
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
    # promo redemptions
    cur.execute("""
        CREATE TABLE IF NOT EXISTS promo_redemptions (
            user_id TEXT NOT NULL,
            code TEXT NOT NULL,
            redeemed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(user_id, code)
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

@with_conn
def create_promo(cur, actor_id: str, code: str, amount: int, max_uses: int = 1, expires_at: Optional[str] = None):
    code = code.strip().upper()
    cur.execute("""
        INSERT INTO promo_codes(code, amount, max_uses, expires_at, created_by)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (code) DO UPDATE SET amount=EXCLUDED.amount, max_uses=EXCLUDED.max_uses, expires_at=EXCLUDED.expires_at
    """, (code, amount, max_uses, expires_at, actor_id))
    return {"ok": True, "code": code}

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

# ---- HTML (same vibe, added tabs) ----
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
    .suggest{margin-top:6px; border:1px solid var(--border); border-radius:10px; background:#0f1a33; display:none;}
    .sitem{padding:6px 8px; border-bottom:1px solid var(--border); cursor:pointer; display:flex; align-items:center; gap:8px}
    .sitem:last-child{border-bottom:none}
    .sitem img{width:20px; height:20px; border-radius:50%; object-fit:cover}
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
      <div class="right" id="authArea">
        <!-- balance + avatar OR login/register -->
      </div>
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
        <p>Your referral code is your Discord ID. Share this link:</p>
        <p><code id="refLink">—</code></p>
        <p class="muted">When friends join via your link and play, you can reward them (details coming soon).</p>
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

    async function render(){
      const auth = qs('authArea');
      try{
        const me = await j('/api/me');
        const bal = await j('/api/balance');
        const avatar = `https://cdn.discordapp.com/avatars/${me.id}/${me.avatar || ''}.png?size=64`;
        auth.innerHTML = `
          <span class="chip">${fmtDL(bal.balance)}</span>
          <img class="avatar" src="${avatar}" onerror="this.style.display='none'" title="${me.username}">
        `;
        loginCard.style.display='none';

        // referral link
        const link = await j('/api/referral/link');
        qs('refLink').textContent = link.link;

        // load my promo redemptions
        const mine = await j('/api/promo/my');
        qs('myCodes').innerHTML = mine.rows.length
          ? '<ul>' + mine.rows.map(r=>`<li><code>${r.code}</code> — ${new Date(r.redeemed_at).toLocaleString()}</li>`).join('') + '</ul>'
          : 'No redemptions yet.';
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
        render(); // refresh balance + my codes
      }catch(e){
        msg.textContent = 'Error: ' + e.message;
      }
    };

    render();
  </script>
</body>
</html>
"""

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
    set_session(resp, {"id": str(me["id"]), "username": me.get("username", "#"), "avatar": me.get("avatar")})
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
    return JSONResponse(user)

@app.get("/api/balance")
async def api_balance(request: Request):
    user = read_session(request)
    if not user: raise HTTPException(401, "Not logged in")
    return JSONResponse({"balance": get_balance(str(user["id"]))})

# ----- Referral -----
@app.get("/api/referral/link")
async def api_ref_link(request: Request):
    user = read_session(request)
    if not user: raise HTTPException(401, "Not logged in")
    base = str(request.base_url).rstrip("/")
    link = f"{base}/?ref={user['id']}"
    return {"code": str(user["id"]), "link": link}

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
    except PromoAlreadyRedeemed as e:
        raise HTTPException(400, str(e))
    except PromoInvalid as e:
        raise HTTPException(400, str(e))
    except PromoExpired as e:
        raise HTTPException(400, str(e))
    except PromoExhausted as e:
        raise HTTPException(400, str(e))

# (Optional) Owner create promo via API
class CreatePromoBody(BaseModel):
    code: str
    amount: int
    max_uses: int = 1
    expires_at: Optional[str] = None  # ISO timestamp string, optional

def require_owner(request: Request):
    user = read_session(request)
    if not user: raise HTTPException(401, "Not logged in")
    if str(user["id"]) != str(OWNER_ID): raise HTTPException(403, "Owner only")
    return user

@app.post("/api/admin/promo/create")
async def api_admin_promo_create(request: Request, body: CreatePromoBody):
    require_owner(request)
    if body.amount == 0: raise HTTPException(400, "Amount cannot be zero")
    return create_promo(str(OWNER_ID), body.code, int(body.amount), int(body.max_uses), body.expires_at)

# ----- Member search for owner autocomplete (kept) -----
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
