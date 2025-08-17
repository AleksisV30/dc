import os, sqlite3, json, asyncio
from urllib.parse import urlencode
from typing import Optional

import httpx
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
OAUTH_REDIRECT = os.getenv("OAUTH_REDIRECT")   # e.g. https://your-app.pella.dev/callback
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")
PORT = int(os.getenv("PORT", "8080"))
DISCORD_API = "https://discord.com/api"
DB_PATH = "balances.db"
OWNER_ID = 1128658280546320426  # owner

GEM = "💎"

# ---------- DB ----------
def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
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
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            actor_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            amount INTEGER NOT NULL,
            reason TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    con.commit(); con.close()

def get_balance(user_id: str) -> int:
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("SELECT balance FROM balances WHERE user_id = ?", (user_id,))
    row = cur.fetchone(); con.close()
    return int(row[0]) if row else 0

def add_balance(user_id: str, amount: int) -> int:
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("INSERT INTO balances (user_id, balance) VALUES (?, 0) "
                "ON CONFLICT(user_id) DO NOTHING", (user_id,))
    cur.execute("UPDATE balances SET balance = balance + ? WHERE user_id = ?", (amount, user_id))
    con.commit()
    cur.execute("SELECT balance FROM balances WHERE user_id = ?", (user_id,))
    new_bal = int(cur.fetchone()[0]); con.close()
    return new_bal

def adjust_balance(actor_id: str, target_id: str, amount: int, reason: Optional[str]) -> int:
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("INSERT INTO balances (user_id, balance) VALUES (?, 0) "
                "ON CONFLICT(user_id) DO NOTHING", (target_id,))
    cur.execute("UPDATE balances SET balance = balance + ? WHERE user_id = ?", (amount, target_id))
    cur.execute("INSERT INTO balance_log(actor_id, target_id, amount, reason) VALUES (?, ?, ?, ?)",
                (actor_id, target_id, amount, reason))
    con.commit()
    cur.execute("SELECT balance FROM balances WHERE user_id = ?", (target_id,))
    new_bal = int(cur.fetchone()[0]); con.close()
    return new_bal

def get_top_balances(limit: int = 20):
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("SELECT user_id, balance FROM balances ORDER BY balance DESC LIMIT ?", (limit,))
    rows = [{"user_id": r[0], "balance": int(r[1])} for r in cur.fetchall()]
    con.close()
    return rows

def get_recent_logs(limit: int = 20):
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("""
        SELECT actor_id, target_id, amount, reason, created_at
        FROM balance_log
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = [{"actor_id":r[0], "target_id":r[1], "amount":int(r[2]), "reason":r[3], "created_at":r[4]} for r in cur.fetchall()]
    con.close()
    return rows

# ---------- Helpers ----------
def parse_user_identifier(identifier: str) -> Optional[str]:
    """Accepts raw ID or mention <@...>/<@!...>; returns raw numeric ID or None."""
    if not identifier: return None
    cleaned = identifier.strip().replace("<@!", "").replace("<@", "").replace(">", "")
    if cleaned.isdigit() and len(cleaned) >= 17:
        return cleaned
    return None

def fmt_dl(n: int) -> str:
    return f"{GEM} {n:,} DL"

def embed(title: str, desc: Optional[str] = None, color: int = 0x00C2FF) -> discord.Embed:
    e = discord.Embed(title=title, description=desc or "", color=color)
    return e

# ---------- Discord bot ----------
intents = discord.Intents.default()
intents.message_content = True  # prefix commands
bot = commands.Bot(command_prefix=PREFIX, intents=intents, help_command=None)  # disable default help

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} (id={bot.user.id})")

@bot.command(name="bal")
async def bal(ctx: commands.Context, user: discord.User | None = None):
    target = user or ctx.author
    bal_value = get_balance(str(target.id))
    e = embed(
        title="Balance",
        desc=f"{target.mention}\n**{fmt_dl(bal_value)}**",
        color=0x34D399
    )
    e.set_footer(text="Currency: Growtopia Diamond Locks (DL)")
    await ctx.reply(embed=e, mention_author=False)

@bot.command(name="addbal")
async def addbal(ctx: commands.Context, user: discord.User | None = None, amount: int | None = None):
    if ctx.author.id != OWNER_ID:
        e = embed("Not allowed", "Only the owner can adjust balances.", color=0xEF4444)
        return await ctx.reply(embed=e, mention_author=False)

    if user is None or amount is None:
        e = embed("Usage", f"`{PREFIX}addbal @User <amount>`", color=0xF59E0B)
        return await ctx.reply(embed=e, mention_author=False)

    if amount == 0:
        e = embed("Invalid amount", "Amount cannot be zero.", color=0xEF4444)
        return await ctx.reply(embed=e, mention_author=False)

    new_balance = adjust_balance(str(ctx.author.id), str(user.id), amount, reason="bot addbal")
    sign = "+" if amount > 0 else ""
    e = embed(
        title="Balance Updated",
        desc=(
            f"**Target:** {user.mention}\n"
            f"**Change:** `{sign}{amount}` → {fmt_dl(new_balance)}"
        ),
        color=0x60A5FA
    )
    e.set_footer(text="Currency: Growtopia Diamond Locks (DL)")
    await ctx.reply(embed=e, mention_author=False)

@bot.command(name="help")
async def help_command(ctx: commands.Context):
    """Show a pretty embed with all commands."""
    is_owner = (ctx.author.id == OWNER_ID)
    e = embed(
        title="💎 DL Bank — Help",
        desc="Manage and view DL balances. Prefix: `{}`".format(PREFIX),
        color=0x60A5FA
    )
    e.add_field(
        name="General",
        value=(
            f"**{PREFIX}help** — Show this help\n"
            f"**{PREFIX}bal** — Show **your** balance\n"
            f"**{PREFIX}bal @User** — Show **someone else’s** balance"
        ),
        inline=False
    )
    owner_line = f"**{PREFIX}addbal @User <amount>** — Add/subtract DL *(owner only)*"
    if is_owner:
        owner_line += " ✅"
    e.add_field(name="Admin", value=owner_line, inline=False)
    e.add_field(
        name="Website",
        value="Use **Login with Discord** to see your balance and (owner) adjust others.",
        inline=False
    )
    e.set_footer(text="Currency: Growtopia Diamond Locks (DL)")
    await ctx.reply(embed=e, mention_author=False)

# ---------- Web (FastAPI + OAuth) ----------
app = FastAPI()
signer = URLSafeSerializer(SECRET_KEY, salt="session")

def set_session(resp: RedirectResponse, payload: dict):
    resp.set_cookie(
        "session",
        signer.dumps(payload),
        httponly=True,
        samesite="lax",
        max_age=7*24*3600,
    )

def read_session(request: Request) -> dict | None:
    raw = request.cookies.get("session")
    if not raw: return None
    try:
        return signer.loads(raw)
    except BadSignature:
        return None

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>💎 DL Bank</title>
  <style>
    :root{
      --bg:#0b1220; --card:#121a2b; --muted:#8aa0c7; --text:#e8efff;
      --accent:#60a5fa; --ok:#34d399; --warn:#f59e0b; --err:#ef4444; --border:#23304c;
    }
    *{box-sizing:border-box}
    body{background:linear-gradient(180deg,#0a1020,#0e1530); color:var(--text); font-family:system-ui,Segoe UI,Roboto,Arial;
         margin:0; min-height:100vh; display:flex; align-items:center; justify-content:center; padding:24px}
    .wrap{width:100%; max-width:860px}
    .hero{padding:22px 24px; background:linear-gradient(135deg,#0e1630 0%,#0a1124 100%); border:1px solid var(--border);
          border-radius:18px; box-shadow:0 20px 60px rgba(0,0,0,.35)}
    .row{display:flex; gap:16px; align-items:center; justify-content:space-between; flex-wrap:wrap}
    .chip{background:#0c1631; border:1px solid var(--border); color:var(--muted); padding:6px 10px; border-radius:999px; font-size:12px}
    .title{font-size:22px; font-weight:700; letter-spacing:.2px}
    .btn{display:inline-flex; align-items:center; gap:8px; padding:10px 14px; border-radius:10px; text-decoration:none;
         border:1px solid var(--border); color:var(--text); background:#0f1a33; cursor:pointer}
    .btn.primary{background:linear-gradient(135deg,#3b82f6,#22c1dc); border-color:transparent}
    .grid{display:grid; gap:16px; grid-template-columns:1fr}
    @media(min-width:900px){.grid{grid-template-columns:1fr 1fr}}
    .card{background:var(--card); border:1px solid var(--border); border-radius:16px; padding:16px}
    .label{color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.1em}
    .big{font-size:28px; font-weight:800}
    table{width:100%; border-collapse:collapse; margin-top:10px}
    th,td{border-bottom:1px solid var(--border); padding:8px 6px; text-align:left}
    .muted{color:var(--muted)}
    input,select{width:100%; background:#0e1833; color:var(--text); border:1px solid var(--border); border-radius:10px; padding:10px}
    .actions{display:flex; gap:8px; flex-wrap:wrap}
    .pill{display:inline-flex; align-items:center; gap:6px; padding:6px 10px; border-radius:999px; background:#0f1a33; border:1px solid var(--border)}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <div class="row">
        <div class="title">💎 DL Bank</div>
        <div id="auth" class="actions"></div>
      </div>
    </div>

    <div class="grid" style="margin-top:16px">
      <div class="card">
        <div class="label">My Balance</div>
        <div id="bal" class="big">—</div>
        <div class="muted" style="margin-top:6px">Currency: Growtopia Diamond Locks (DL)</div>
      </div>

      <div id="ownerBox" class="card" style="display:none">
        <div class="label">Owner Panel</div>
        <div class="pill" style="margin:8px 0">Adjust balances, view top holders & logs.</div>
        <div class="row" style="gap:8px; align-items:flex-end">
          <div style="flex:2">
            <div class="label">Discord ID or &lt;@mention&gt;</div>
            <input id="target" placeholder="e.g. 1128658280546320426 or <@1128...>" />
          </div>
          <div style="flex:1">
            <div class="label">Amount (+/-)</div>
            <input id="amount" type="number" placeholder="e.g. 10 or -5" />
          </div>
          <div style="flex:2">
            <div class="label">Reason (optional)</div>
            <input id="reason" placeholder="Promo / correction / prize ..." />
          </div>
          <div>
            <button id="doAdjust" class="btn primary">Apply</button>
          </div>
        </div>
        <div id="msg" class="muted" style="margin-top:8px"></div>
      </div>
    </div>

    <div id="ownerTables" class="grid" style="margin-top:16px; display:none">
      <div class="card">
        <div class="label">Top Balances</div>
        <div id="top"></div>
      </div>
      <div class="card">
        <div class="label">Recent Changes</div>
        <div id="logs"></div>
      </div>
    </div>
  </div>

  <script>
    async function j(u, opt={}) {
      const r = await fetch(u, Object.assign({credentials:'include'}, opt));
      if(!r.ok) throw new Error(await r.text());
      return r.json();
    }

    function fmtDL(n){ return `💎 ${Number(n).toLocaleString()} DL`; }

    async function render(){
      const auth = document.getElementById('auth');
      const balEl = document.getElementById('bal');
      const ownerBox = document.getElementById('ownerBox');
      const ownerTables = document.getElementById('ownerTables');
      try{
        const me = await j('/api/me');
        auth.innerHTML = `
          <span class="pill">Logged in as <b>${me.username}</b> (${me.id})</span>
          <a class="btn" href="/logout">Logout</a>
        `;
        const bal = await j('/api/balance');
        balEl.textContent = fmtDL(bal.balance);

        if (me.id === 'OWNER_PLACEHOLDER') {
          ownerBox.style.display = '';
          ownerTables.style.display = 'grid';

          const top = await j('/api/admin/top');
          const logs = await j('/api/admin/logs');

          document.getElementById('top').innerHTML = `
            <table>
              <thead><tr><th>User</th><th>Balance</th></tr></thead>
              <tbody>${top.rows.map(r=>`<tr><td>&lt;@${r.user_id}&gt;</td><td>${fmtDL(r.balance)}</td></tr>`).join('')}</tbody>
            </table>
          `;
          document.getElementById('logs').innerHTML = `
            <table>
              <thead><tr><th>When</th><th>Actor</th><th>Target</th><th>Change</th><th>Reason</th></tr></thead>
              <tbody>${logs.rows.map(r=>`<tr>
                <td>${r.created_at}</td>
                <td>&lt;@${r.actor_id}&gt;</td>
                <td>&lt;@${r.target_id}&gt;</td>
                <td>${r.amount>0?'+':''}${r.amount}</td>
                <td>${r.reason ?? ''}</td>
              </tr>`).join('')}</tbody>
            </table>
          `;

          document.getElementById('doAdjust').onclick = async ()=>{
            const target = document.getElementById('target').value.trim();
            const amount = parseInt(document.getElementById('amount').value, 10);
            const reason = document.getElementById('reason').value.trim();
            const msg = document.getElementById('msg');
            msg.textContent = '';
            try{
              if(!target || Number.isNaN(amount)) throw new Error('Fill target and amount.');
              const res = await j('/api/admin/adjust', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({identifier: target, amount, reason})});
              msg.textContent = 'OK. New balance: ' + fmtDL(res.new_balance);
              // refresh cards
              const bal = await j('/api/balance');
              balEl.textContent = fmtDL(bal.balance);
              const top = await j('/api/admin/top');
              const logs = await j('/api/admin/logs');
              document.getElementById('top').innerHTML = `
                <table><thead><tr><th>User</th><th>Balance</th></tr></thead>
                <tbody>${top.rows.map(r=>`<tr><td>&lt;@${r.user_id}&gt;</td><td>${fmtDL(r.balance)}</td></tr>`).join('')}</tbody></table>`;
              document.getElementById('logs').innerHTML = `
                <table><thead><tr><th>When</th><th>Actor</th><th>Target</th><th>Change</th><th>Reason</th></tr></thead>
                <tbody>${logs.rows.map(r=>`<tr><td>${r.created_at}</td><td>&lt;@${r.actor_id}&gt;</td><td>&lt;@${r.target_id}&gt;</td><td>${r.amount>0?'+':''}${r.amount}</td><td>${r.reason ?? ''}</td></tr>`).join('')}</tbody></table>`;
            }catch(e){
              msg.textContent = 'Error: ' + e.message;
            }
          };
        } else {
          ownerBox.style.display = 'none';
          ownerTables.style.display = 'none';
        }
      }catch(e){
        auth.innerHTML = `<a class="btn primary" href="/login">Login with Discord</a>`;
        balEl.textContent = '—';
        ownerBox.style.display = 'none';
        ownerTables.style.display = 'none';
      }
    }
    render();
  </script>
</body>
</html>
""".replace("OWNER_PLACEHOLDER", str(OWNER_ID))

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
    if not (CLIENT_ID and CLIENT_SECRET and OAUTH_REDIRECT):
        raise HTTPException(500, "OAuth not configured")

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

        me = (await client.get(
            f"{DISCORD_API}/users/@me",
            headers={"Authorization": f"{token['token_type']} {token['access_token']}"}
        )).json()

    resp = RedirectResponse(url="/")
    set_session(resp, {"id": str(me["id"]), "username": me.get("username", "#")})
    return resp

@app.get("/logout")
async def logout():
    resp = RedirectResponse(url="/")
    resp.delete_cookie("session")
    return resp

@app.get("/api/me")
async def api_me(request: Request):
    user = read_session(request)
    if not user:
        raise HTTPException(401, "Not logged in")
    return JSONResponse(user)

@app.get("/api/balance")
async def api_balance(request: Request):
    user = read_session(request)
    if not user:
        raise HTTPException(401, "Not logged in")
    return JSONResponse({"balance": get_balance(str(user["id"]))})

# -------- Admin APIs (owner-only) --------
class AdjustBody(BaseModel):
    identifier: str = Field(..., description="Discord ID or <@mention>")
    amount: int = Field(..., description="Positive to add, negative to subtract", ge=-1_000_000, le=1_000_000)
    reason: Optional[str] = Field(default=None, max_length=200)

def require_owner(request: Request):
    user = read_session(request)
    if not user:
        raise HTTPException(401, "Not logged in")
    if str(user["id"]) != str(OWNER_ID):
        raise HTTPException(403, "Owner only")
    return user

@app.get("/api/admin/top")
async def api_admin_top(request: Request, limit: int = Query(20, ge=1, le=100)):
    require_owner(request)
    return {"rows": get_top_balances(limit)}

@app.get("/api/admin/logs")
async def api_admin_logs(request: Request, limit: int = Query(20, ge=1, le=100)):
    require_owner(request)
    return {"rows": get_recent_logs(limit)}

@app.get("/api/admin/balance")
async def api_admin_balance(request: Request, identifier: str = Query(...)):
    require_owner(request)
    user_id = parse_user_identifier(identifier)
    if not user_id:
        raise HTTPException(400, "Invalid identifier (use raw ID or <@mention>)")
    return {"user_id": user_id, "balance": get_balance(user_id)}

@app.post("/api/admin/adjust")
async def api_admin_adjust(request: Request, body: AdjustBody):
    actor = require_owner(request)
    user_id = parse_user_identifier(body.identifier)
    if not user_id:
        raise HTTPException(400, "Invalid identifier (use raw ID or <@mention>)")
    if body.amount == 0:
        raise HTTPException(400, "Amount cannot be zero")
    new_balance = adjust_balance(str(actor["id"]), user_id, int(body.amount), body.reason)
    return {"user_id": user_id, "new_balance": new_balance}

@app.get("/health")
async def health():
    return {"ok": True}

# ---------- Runner ----------
async def main():
    init_db()
    if not BOT_TOKEN:
        raise RuntimeError("DISCORD_TOKEN env var not set.")
    config = uvicorn.Config(app, host="0.0.0.0", port=PORT, log_level="info")
    server = uvicorn.Server(config)
    await asyncio.gather(
        bot.start(BOT_TOKEN),
        server.serve()
    )

if __name__ == "__main__":
    asyncio.run(main())
