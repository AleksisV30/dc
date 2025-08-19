# app/main.py — single-file site with games + chat + profile + balance
import os, json, asyncio, re, random, string, datetime, base64
from urllib.parse import urlencode
from typing import Optional, Dict
from decimal import Decimal, ROUND_DOWN, getcontext
from contextlib import asynccontextmanager

import httpx
import psycopg
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from itsdangerous import URLSafeSerializer, BadSignature
from pydantic import BaseModel

# ---------- Import games ----------
from crash import (
    ensure_betting_round, place_bet, load_round, begin_running,
    finish_round, create_next_betting, last_busts, your_bet,
    your_history, cashout_now, current_multiplier
)
from mines import (
    mines_start, mines_pick, mines_cashout, mines_state, mines_history
)

# ---------- Config ----------
getcontext().prec = 28
CLIENT_ID = os.getenv("DISCORD_CLIENT_ID", "")
CLIENT_SECRET = os.getenv("DISCORD_CLIENT_SECRET", "")
OAUTH_REDIRECT = os.getenv("OAUTH_REDIRECT", "")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")
PORT = int(os.getenv("PORT", "8080"))
DISCORD_API = "https://discord.com/api"
OWNER_ID = int(os.getenv("OWNER_ID", "0"))
DATABASE_URL = os.getenv("DATABASE_URL")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
GUILD_ID = int(os.getenv("GUILD_ID", "0"))
DISCORD_INVITE = os.getenv("DISCORD_INVITE", "")

TWO = Decimal("0.01")
def D(x) -> Decimal: return x if isinstance(x, Decimal) else Decimal(str(x))
def q2(x: Decimal) -> Decimal: return D(x).quantize(TWO, rounding=ROUND_DOWN)

UTC = datetime.timezone.utc
def now_utc() -> datetime.datetime: return datetime.datetime.now(UTC)
def iso(dt: Optional[datetime.datetime]) -> Optional[str]:
    return None if dt is None else dt.astimezone(UTC).isoformat()

# ---------- App ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    apply_migrations()
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------- Sessions ----------
SER = URLSafeSerializer(SECRET_KEY, salt="session-v1")
def _set_session(resp, data: dict):
    resp.set_cookie("session", SER.dumps(data), max_age=30*86400, httponly=True, samesite="lax")
def _clear_session(resp): resp.delete_cookie("session")
def _require_session(request: Request) -> dict:
    raw = request.cookies.get("session")
    if not raw: raise HTTPException(401, "Not logged in")
    try:
        sess = SER.loads(raw)
        if not sess.get("id"): raise BadSignature("no id")
        return sess
    except BadSignature:
        raise HTTPException(401, "Invalid session")

# ---------- DB ----------
def with_conn(fn):
    def wrapper(*args, **kwargs):
        if not DATABASE_URL: raise RuntimeError("DATABASE_URL not set")
        with psycopg.connect(DATABASE_URL) as con:
            with con.cursor() as cur:
                res = fn(cur, *args, **kwargs)
                con.commit()
                return res
    return wrapper

@with_conn
def init_db(cur):
    cur.execute("""CREATE TABLE IF NOT EXISTS balances (
        user_id TEXT PRIMARY KEY, balance NUMERIC(18,2) NOT NULL DEFAULT 0)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS profiles (
        user_id TEXT PRIMARY KEY, display_name TEXT NOT NULL, name_lower TEXT NOT NULL UNIQUE,
        xp INTEGER NOT NULL DEFAULT 0, role TEXT NOT NULL DEFAULT 'member', is_anon BOOLEAN NOT NULL DEFAULT FALSE,
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS balance_log (
        id BIGSERIAL PRIMARY KEY, actor_id TEXT NOT NULL, target_id TEXT NOT NULL,
        amount NUMERIC(18,2) NOT NULL, reason TEXT, created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS oauth_tokens (
        user_id TEXT PRIMARY KEY, access_token TEXT NOT NULL, refresh_token TEXT, expires_at TIMESTAMPTZ)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS chat_messages (
        id BIGSERIAL PRIMARY KEY, user_id TEXT NOT NULL, username TEXT NOT NULL, text TEXT NOT NULL,
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP)""")

# ---------- Balance/Profile ----------
@with_conn
def get_balance(cur, user_id: str) -> Decimal:
    cur.execute("SELECT balance FROM balances WHERE user_id=%s", (user_id,))
    r = cur.fetchone()
    return q2(r[0]) if r else Decimal("0.00")

@with_conn
def adjust_balance(cur, actor_id: str, target_id: str, amount, reason: Optional[str]) -> Decimal:
    amount = q2(D(amount))
    cur.execute("INSERT INTO balances(user_id,balance) VALUES (%s,0) ON CONFLICT(user_id) DO NOTHING", (target_id,))
    cur.execute("UPDATE balances SET balance=balance+%s WHERE user_id=%s", (amount, target_id))
    cur.execute("INSERT INTO balance_log(actor_id,target_id,amount,reason) VALUES (%s,%s,%s,%s)",
                (actor_id, target_id, amount, reason))
    cur.execute("SELECT balance FROM balances WHERE user_id=%s", (target_id,))
    return q2(cur.fetchone()[0])

NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{3,20}$")

@with_conn
def ensure_profile_row(cur, user_id: str):
    defname = f"user_{str(user_id)[-4:]}"
    cur.execute("""
        INSERT INTO profiles(user_id, display_name, name_lower)
        VALUES (%s,%s,%s)
        ON CONFLICT(user_id) DO NOTHING
    """, (user_id, defname, defname))

@with_conn
def profile_info(cur, user_id: str):
    ensure_profile_row(user_id)
    cur.execute("SELECT display_name, xp, role, is_anon, created_at FROM profiles WHERE user_id=%s", (user_id,))
    row = cur.fetchone()
    display_name, xp, role, is_anon, created_at = row
    return {
        "id": str(user_id), "display_name": display_name,
        "balance": float(get_balance(user_id)), "role": role,
        "xp": int(xp), "is_anon": bool(is_anon),
        "created_at": str(created_at)
    }

# ---------- Promo ----------
class PromoInvalid(Exception): pass
class PromoExpired(Exception): pass
class PromoExhausted(Exception): pass
class PromoAlreadyRedeemed(Exception): pass

@with_conn
def redeem_promo(cur, user_id: str, code: str) -> Decimal:
    code = (code or "").strip().lower()
    cur.execute("SELECT code, amount, max_uses, uses, expires_at FROM promo_codes WHERE code=%s", (code,))
    r = cur.fetchone()
    if not r: raise PromoInvalid("Invalid code")
    _, amount, max_uses, uses, expires_at = r
    if expires_at and now_utc() > expires_at: raise PromoExpired("Code expired")
    if uses >= max_uses: raise PromoExhausted("Code fully used")
    cur.execute("SELECT 1 FROM promo_redemptions WHERE user_id=%s AND code=%s", (user_id, code))
    if cur.fetchone(): raise PromoAlreadyRedeemed("You already redeemed this code")
    cur.execute("UPDATE promo_codes SET uses=uses+1 WHERE code=%s", (code,))
    cur.execute("INSERT INTO promo_redemptions(user_id,code) VALUES(%s,%s)", (user_id, code))
    newbal = adjust_balance(user_id, user_id, amount, f"promo:{code}")
    return newbal
@with_conn
def create_promo(cur, creator_id: str, code: Optional[str], amount: str, max_uses: int, expires_at: Optional[str]):
    if not code:
        code = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(8))
    code = code.lower()
    amt = q2(D(amount))
    exp_dt = None
    if expires_at:
        try:
            exp_dt = datetime.datetime.fromisoformat(expires_at).astimezone(UTC)
        except Exception:
            raise ValueError("Invalid expires_at format")
    cur.execute("""CREATE TABLE IF NOT EXISTS promo_codes (
                     code TEXT PRIMARY KEY, amount NUMERIC(18,2) NOT NULL,
                     max_uses INTEGER NOT NULL DEFAULT 1, uses INTEGER NOT NULL DEFAULT 0,
                     expires_at TIMESTAMPTZ, created_by TEXT, created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS promo_redemptions (
                     user_id TEXT NOT NULL, code TEXT NOT NULL, redeemed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                     PRIMARY KEY(user_id, code))""")
    cur.execute("""INSERT INTO promo_codes(code,amount,max_uses,expires_at,created_by)
                   VALUES(%s,%s,%s,%s,%s)""",
                (code, amt, max_uses, exp_dt, creator_id))
    return {"ok": True, "code": code, "amount": float(amt), "max_uses": max_uses,
            "expires_at": (exp_dt.isoformat() if exp_dt else None)}

# ---------- OAuth / Auth ----------
@app.get("/login")
async def login():
    if not (CLIENT_ID and OAUTH_REDIRECT):
        return HTMLResponse("OAuth not configured")
    params = {
        "client_id": CLIENT_ID, "redirect_uri": OAUTH_REDIRECT, "response_type": "code",
        "scope": "identify guilds.join", "prompt": "consent"
    }
    return RedirectResponse(f"{DISCORD_API}/oauth2/authorize?{urlencode(params)}")

@app.get("/callback")
async def callback(code: str):
    if not (CLIENT_ID and CLIENT_SECRET and OAUTH_REDIRECT):
        return HTMLResponse("OAuth not configured")
    async with httpx.AsyncClient(timeout=15) as cx:
        data = {"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET,
                "grant_type": "authorization_code", "code": code, "redirect_uri": OAUTH_REDIRECT}
        r = await cx.post(f"{DISCORD_API}/oauth2/token", data=data,
                          headers={"Content-Type":"application/x-www-form-urlencoded"})
        if r.status_code != 200:
            return HTMLResponse(f"OAuth failed: {r.text}", status_code=400)
        tok = r.json()
        access = tok.get("access_token")
        u = await cx.get(f"{DISCORD_API}/users/@me", headers={"Authorization": f"Bearer {access}"})
        if u.status_code != 200:
            return HTMLResponse(f"User fetch failed: {u.text}", status_code=400)
        me = u.json()
    user_id = str(me["id"])
    username = f'{me.get("username","user")}#{me.get("discriminator","0")}'.replace("#0","")
    avatar_hash = me.get("avatar")
    avatar_url = (f"https://cdn.discordapp.com/avatars/{user_id}/{avatar_hash}.png?size=64"
                  if avatar_hash else "https://cdn.discordapp.com/embed/avatars/0.png")
    ensure_profile_row(user_id)
    save_tokens(user_id, tok.get("access_token",""), tok.get("refresh_token"), tok.get("expires_in"))
    resp = RedirectResponse("/")
    _set_session(resp, {"id": user_id, "username": username, "avatar_url": avatar_url})
    return resp

@app.get("/logout")
async def logout():
    resp = RedirectResponse("/")
    _clear_session(resp)
    return resp

# ---------- Tokens ----------
@with_conn
def get_role(cur, user_id: str) -> str:
    cur.execute("SELECT role FROM profiles WHERE user_id=%s", (user_id,))
    r = cur.fetchone()
    return r[0] if r else "member"

@with_conn
def get_profile_name(cur, user_id: str):
    cur.execute("SELECT display_name FROM profiles WHERE user_id=%s", (user_id,))
    r = cur.fetchone(); return r[0] if r else None

@with_conn
def set_role(cur, target_id: str, role: str):
    if role not in ("member","admin","owner"): raise ValueError("Invalid role")
    cur.execute("UPDATE profiles SET role=%s WHERE user_id=%s", (role, target_id))
    return {"ok": True, "role": role}

@with_conn
def set_profile_is_anon(cur, user_id: str, is_anon: bool):
    ensure_profile_row(user_id)
    cur.execute("ALTER TABLE profiles ADD COLUMN IF NOT EXISTS is_anon BOOLEAN NOT NULL DEFAULT FALSE")
    cur.execute("UPDATE profiles SET is_anon=%s WHERE user_id=%s", (bool(is_anon), user_id))
    return {"ok": True, "is_anon": bool(is_anon)}

@with_conn
def public_profile(cur, user_id: str) -> Optional[dict]:
    cur.execute("SELECT display_name, xp, role, created_at, is_anon FROM profiles WHERE user_id=%s", (user_id,))
    r = cur.fetchone()
    if not r: return None
    display_name, xp, role, created_at, is_anon = r
    level = 1 + int(xp)//100
    return {"id": str(user_id), "name": ("Anonymous" if is_anon else display_name),
            "role": role, "is_anon": bool(is_anon), "xp": int(xp), "level": level,
            "created_at": str(created_at)}

# ---------- Leaderboard helpers ----------
def _start_of_utc_day(dt: datetime.datetime) -> datetime.datetime:
    return dt.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
def _start_of_utc_month(dt: datetime.datetime) -> datetime.datetime:
    return dt.astimezone(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)

@with_conn
def get_leaderboard_rows_db(cur, period: str, limit: int = 50):
    period = (period or "daily").lower()
    now = now_utc()
    params = []
    where_crash = ""
    where_mines = "WHERE status<>'active'"
    if period in ("daily","monthly"):
        start = _start_of_utc_day(now) if period=="daily" else _start_of_utc_month(now)
        where_crash = "WHERE created_at >= %s"
        where_mines += " AND started_at >= %s"
        params.extend([start, start])
    sql = f"""
        WITH wagers AS (
          SELECT user_id, COALESCE(SUM(bet),0)::numeric(18,2) AS total
          FROM crash_games {where_crash} GROUP BY user_id
          UNION ALL
          SELECT user_id, COALESCE(SUM(bet),0)::numeric(18,2) AS total
          FROM mines_games {where_mines} GROUP BY user_id
        ),
        by_user AS (
          SELECT user_id, SUM(total)::numeric(18,2) AS total_wagered
          FROM wagers GROUP BY user_id
        )
        SELECT bu.user_id, COALESCE(p.display_name, 'user_'||RIGHT(bu.user_id,4)) AS name,
               COALESCE(p.is_anon, FALSE) AS is_anon, bu.total_wagered::numeric(18,2)
        FROM by_user bu LEFT JOIN profiles p ON p.user_id=bu.user_id
        ORDER BY bu.total_wagered DESC, bu.user_id
        LIMIT %s
    """
    params.append(limit)
    cur.execute(sql, tuple(params))
    rows = cur.fetchall()
    return [{"user_id": str(uid), "display_name": name, "is_anon": bool(is_anon),
             "total_wagered": float(q2(D(total)))} for uid, name, is_anon, total in rows]

# ---------- Referral helpers ----------
NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{3,20}$")

@with_conn
def set_ref_name(cur, user_id: str, name: str):
    if not NAME_RE.match(name): raise ValueError("3–20 chars: letters, numbers, _ or -")
    lower = name.lower()
    cur.execute("""CREATE TABLE IF NOT EXISTS ref_names (
                     user_id TEXT PRIMARY KEY, name_lower TEXT UNIQUE NOT NULL,
                     created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS ref_visits (
                     id BIGSERIAL PRIMARY KEY, referrer_id TEXT NOT NULL,
                     joined_user_id TEXT, created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP)""")
    cur.execute("SELECT user_id FROM ref_names WHERE name_lower=%s AND user_id<>%s", (lower, user_id))
    if cur.fetchone(): raise ValueError("Name is already taken")
    cur.execute("""INSERT INTO ref_names(user_id, name_lower) VALUES (%s,%s)
                   ON CONFLICT(user_id) DO UPDATE SET name_lower=EXCLUDED.name_lower""", (user_id, lower))
    return {"ok": True, "name": lower}

@with_conn
def get_ref_state(cur, user_id: str):
    cur.execute("SELECT name_lower FROM ref_names WHERE user_id=%s", (user_id,))
    r = cur.fetchone(); name = r[0] if r else None
    cur.execute("SELECT COUNT(*) FROM ref_visits WHERE referrer_id=%s AND joined_user_id IS NOT NULL", (user_id,))
    joined = int(cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM ref_visits WHERE referrer_id=%s", (user_id,))
    clicks = int(cur.fetchone()[0])
    return {"name": name, "joined": joined, "clicks": clicks}

# ---------- Mines/Crash imports ----------
from crash import (
    ensure_betting_round, place_bet, load_round, begin_running,
    finish_round, create_next_betting, last_busts, your_bet,
    your_history, cashout_now, current_multiplier
)
from mines import (
    mines_start, mines_pick, mines_cashout, mines_state, mines_history
)

# ---------- API: Me/Profile ----------
@app.get("/api/me")
async def api_me(request: Request):
    s = _require_session(request)
    in_guild = False
    if DISCORD_BOT_TOKEN and GUILD_ID:
        try:
            async with httpx.AsyncClient(timeout=8) as cx:
                r = await cx.get(f"{DISCORD_API}/guilds/{GUILD_ID}/members/{s['id']}",
                                 headers={"Authorization": f"Bot {DISCORD_BOT_TOKEN}"})
                in_guild = (r.status_code == 200)
        except:
            in_guild = False
    return {"id": s["id"], "username": s["username"], "avatar_url": s.get("avatar_url"), "in_guild": in_guild}

@app.get("/api/balance")
async def api_balance(request: Request):
    s = _require_session(request)
    return {"balance": float(get_balance(s["id"]))}

@app.get("/api/profile")
async def api_profile(request: Request):
    s = _require_session(request)
    return profile_info(s["id"])

@app.get("/api/public/{user_id}")
async def api_public(user_id: str):
    info = public_profile(user_id)
    if not info: raise HTTPException(404, "Not found")
    info.pop("balance", None)
    return info

# ---------- Settings ----------
class AnonIn(BaseModel): is_anon: bool

@app.get("/api/settings/get")
async def api_settings_get(request: Request):
    s = _require_session(request)
    info = profile_info(s["id"])
    return {"is_anon": bool(info["is_anon"])}

@app.post("/api/settings/set_anon")
async def api_settings_set_anon(request: Request, body: AnonIn):
    s = _require_session(request)
    return set_profile_is_anon(s["id"], bool(body.is_anon))

# ---------- Leaderboard ----------
@app.get("/api/leaderboard")
async def api_leaderboard(period: str = Query("daily"), limit: int = Query(50, ge=1, le=200)):
    rows = get_leaderboard_rows_db(period, limit)
    return {"rows": rows}

# ---------- Referral ----------
class RefIn(BaseModel): name: str

@app.get("/api/referral/state")
async def api_ref_state(request: Request):
    s = _require_session(request)
    return get_ref_state(s["id"])

@app.post("/api/referral/set")
async def api_ref_set(request: Request, body: RefIn):
    s = _require_session(request)
    return set_ref_name(s["id"], body.name)

@app.get("/r/{refname}")
async def referral_landing(refname: str, request: Request):
    rn = (refname or "").lower()
    with psycopg.connect(DATABASE_URL) as con, con.cursor() as cur:
        cur.execute("SELECT user_id FROM ref_names WHERE name_lower=%s", (rn,))
        r = cur.fetchone()
        if r:
            referrer = str(r[0])
            cur.execute("INSERT INTO ref_visits(referrer_id) VALUES (%s)", (referrer,))
            con.commit()
    html = f"""<script>
      document.cookie = "refname={rn}; path=/; max-age=1209600; samesite=lax"; location.href="/";
    </script>"""
    return HTMLResponse(html)

@app.get("/api/referral/attach")
async def api_ref_attach(request: Request, refname: str = ""):
    s = _require_session(request)
    rn = (refname or "").lower()
    if not rn: return {"ok": True}
    with psycopg.connect(DATABASE_URL) as con, con.cursor() as cur:
        cur.execute("SELECT user_id FROM ref_names WHERE name_lower=%s", (rn,))
        r = cur.fetchone()
        if not r: return {"ok": True}
        referrer = str(r[0])
        if referrer == s["id"]: return {"ok": True}
        cur.execute("SELECT referred_by FROM profiles WHERE user_id=%s", (s["id"],))
        already = cur.fetchone()
        if already and already[0]: return {"ok": True}
        cur.execute("ALTER TABLE profiles ADD COLUMN IF NOT EXISTS referred_by TEXT")
        cur.execute("UPDATE profiles SET referred_by=%s WHERE user_id=%s", (referrer, s["id"]))
        cur.execute("""INSERT INTO ref_visits(referrer_id, joined_user_id) VALUES (%s,%s)""",
                    (referrer, s["id"]))
        con.commit()
    return {"ok": True}

# ---------- Promo endpoints ----------
class PromoIn(BaseModel): code: str
class PromoCreateIn(BaseModel):
    code: Optional[str] = None
    amount: str
    max_uses: int = 1
    expires_at: Optional[str] = None

@app.get("/api/promo/my")
async def api_promo_my(request: Request):
    s = _require_session(request)
    with psycopg.connect(DATABASE_URL) as con, con.cursor() as cur:
        cur.execute("""CREATE TABLE IF NOT EXISTS promo_redemptions (
                         user_id TEXT NOT NULL, code TEXT NOT NULL, redeemed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                         PRIMARY KEY(user_id, code))""")
        cur.execute("SELECT code, redeemed_at FROM promo_redemptions WHERE user_id=%s ORDER BY redeemed_at DESC LIMIT 50", (s["id"],))
        rows = [{"code": r[0], "redeemed_at": str(r[1])} for r in cur.fetchall()]
    return {"rows": rows}

@app.post("/api/promo/redeem")
async def api_promo_redeem(request: Request, body: PromoIn):
    s = _require_session(request)
    try:
        bal = redeem_promo(s["id"], body.code)
        return {"ok": True, "new_balance": float(bal)}
    except (PromoInvalid, PromoExpired, PromoExhausted, PromoAlreadyRedeemed) as e:
        raise HTTPException(400, str(e))

@app.post("/api/admin/promo/create")
async def api_admin_promo_create(request: Request, body: PromoCreateIn):
    s = _require_session(request)
    role = get_role(s["id"])
    if role not in ("admin","owner"): raise HTTPException(403, "No permission")
    return create_promo(s["id"], body.code, body.amount, int(body.max_uses or 1), body.expires_at)

# ---------- Chat ----------
class ChatIn(BaseModel): text: str

@with_conn
def chat_insert(cur, user_id: str, username: str, text: str):
    text = (text or "").strip()
    if not text: raise ValueError("Message is empty")
    if len(text) > 300: raise ValueError("Message too long")
    cur.execute("""INSERT INTO chat_messages(user_id,username,text) VALUES (%s,%s,%s)
                   RETURNING id,created_at""", (user_id, username, text))
    row = cur.fetchone()
    return {"id": int(row[0]), "created_at": str(row[1])}

@with_conn
def chat_fetch(cur, since_id: int, limit: int):
    if since_id <= 0:
        cur.execute("""SELECT id,user_id,username,text,created_at
                       FROM chat_messages ORDER BY id DESC LIMIT %s""", (limit,))
        rows = list(reversed(cur.fetchall()))
    else:
        cur.execute("""SELECT id,user_id,username,text,created_at
                       FROM chat_messages WHERE id>%s ORDER BY id ASC LIMIT %s""", (since_id, limit))
        rows = cur.fetchall()
    out = []
    uids = list({str(r[1]) for r in rows})
    levels: Dict[str,int] = {}; roles: Dict[str,str] = {}
    if uids:
        cur.execute("SELECT user_id, xp, role FROM profiles WHERE user_id = ANY(%s)", (uids,))
        for uid, xp, role in cur.fetchall():
            levels[str(uid)] = 1 + int(xp)//100
            roles[str(uid)] = role or "member"
    for mid, uid, uname, txt, ts in rows:
        uid = str(uid)
        out.append({"id": int(mid), "user_id": uid, "username": uname,
                    "level": int(levels.get(uid,1)), "role": roles.get(uid,"member"),
                    "text": txt, "created_at": str(ts)})
    return out

@app.get("/api/chat/fetch")
async def api_chat_fetch(since: int = 0, limit: int = 30):
    return {"rows": chat_fetch(since, limit)}

@app.post("/api/chat/send")
async def api_chat_send(request: Request, body: ChatIn):
    s = _require_session(request)
    return chat_insert(s["id"], s["username"], body.text)

# ---------- Crash ----------
class CrashBetIn(BaseModel):
    bet: str
    cashout: Optional[float] = None

@app.get("/api/crash/state")
async def api_crash_state(request: Request):
    try:
        s = _require_session(request); uid = s["id"]
    except:
        uid = None
    rid, info = ensure_betting_round()
    now = now_utc()
    if info["status"]=="betting" and now>=info["betting_ends_at"]:
        begin_running(rid); info = load_round()
    if info and info["status"]=="running" and info["expected_end_at"] and now>=info["expected_end_at"]:
        finish_round(rid); create_next_betting(); info = load_round()
    out = {"phase": info["status"], "bust": info["bust"],
           "betting_opens_at": iso(info["betting_opens_at"]),
           "betting_ends_at": iso(info["betting_ends_at"]),
           "started_at": iso(info["started_at"]),
           "expected_end_at": iso(info["expected_end_at"]),
           "last_busts": last_busts()}
    if info["status"] == "running":
        out["current_multiplier"] = current_multiplier(
            info["started_at"], info["expected_end_at"], info["bust"], now
        )
    if uid:
        y = your_bet(rid, uid)
        if y: out["your_bet"] = y
    return out

@app.post("/api/crash/place")
async def api_crash_place(request: Request, body: CrashBetIn):
    s = _require_session(request)
    bet = q2(D(body.bet or "0"))
    cashout = float(body.cashout or 2.0)
    return place_bet(s["id"], bet, max(1.01, cashout))

@app.post("/api/crash/cashout")
async def api_crash_cashout(request: Request):
    s = _require_session(request)
    cur = load_round()
    if not cur or cur["status"] != "running":
        raise HTTPException(400, "No running round")
    return cashout_now(s["id"])

@app.get("/api/crash/history")
async def api_crash_history(request: Request):
    s = _require_session(request)
    return {"rows": your_history(s["id"], 10)}

# ---------- Mines ----------
class MinesStartIn(BaseModel):
    bet: str
    mines: int

@app.post("/api/mines/start")
async def api_mines_start(request: Request, body: MinesStartIn):
    s = _require_session(request)
    return mines_start(s["id"], q2(D(body.bet or "0")), int(body.mines))

@app.post("/api/mines/pick")
async def api_mines_pick(request: Request, cell: int = Query(..., ge=0, le=24)):
    s = _require_session(request)
    return mines_pick(s["id"], cell)

@app.post("/api/mines/cashout")
async def api_mines_cashout(request: Request):
    s = _require_session(request)
    return mines_cashout(s["id"])

@app.get("/api/mines/state")
async def api_mines_state(request: Request):
    s = _require_session(request)
    return mines_state(s["id"]) or {}

@app.get("/api/mines/history")
async def api_mines_history(request: Request):
    s = _require_session(request)
    return {"rows": mines_history(s["id"], 15)}

# ---------- Discord Join ----------
async def discord_refresh_token(user_id: str):
    rec = get_tokens(user_id)
    if not rec or not rec.get("refresh_token"): return None
    if not CLIENT_ID or not CLIENT_SECRET: return None
    async with httpx.AsyncClient(timeout=15) as cx:
        data = {"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET, "grant_type": "refresh_token",
                "refresh_token": rec["refresh_token"]}
        r = await cx.post(f"{DISCORD_API}/oauth2/token", data=data,
                          headers={"Content-Type":"application/x-www-form-urlencoded"})
        if r.status_code != 200: return None
        js = r.json()
        save_tokens(user_id, js.get("access_token",""), js.get("refresh_token"), js.get("expires_in"))
        return js.get("access_token")

@with_conn
def get_tokens(cur, user_id: str):
    cur.execute("SELECT access_token, refresh_token, expires_at FROM oauth_tokens WHERE user_id=%s", (user_id,))
    r = cur.fetchone()
    return {"access_token": r[0], "refresh_token": r[1], "expires_at": r[2]} if r else None

async def discord_get_valid_access_token(user_id: str):
    rec = get_tokens(user_id)
    if not rec: return None
    exp = rec.get("expires_at")
    if exp and isinstance(exp, datetime.datetime) and exp.tzinfo is None:
        exp = exp.replace(tzinfo=UTC)
    if (not exp) or (exp - now_utc() < datetime.timedelta(seconds=30)):
        tok = await discord_refresh_token(user_id)
        if tok: return tok
    return rec.get("access_token")

async def guild_add_member(user_id: str, nickname: Optional[str] = None):
    if not (DISCORD_BOT_TOKEN and GUILD_ID):
        raise HTTPException(500, "Discord bot or guild not configured")
    access = await discord_get_valid_access_token(user_id)
    if not access: raise HTTPException(400, "Missing OAuth token. Re-login needed.")
    payload = {"access_token": access}
    if nickname: payload["nick"] = nickname
    async with httpx.AsyncClient(timeout=15) as cx:
        url = f"{DISCORD_API}/guilds/{GUILD_ID}/members/{user_id}"
        r = await cx.put(url, json=payload, headers={"Authorization": f"Bot {DISCORD_BOT_TOKEN}"})
        if r.status_code in (201, 204, 409): return {"ok": True}
        raise HTTPException(r.status_code, f"Discord join failed: {r.text}")

@app.post("/api/discord/join")
async def api_discord_join(request: Request):
    s = _require_session(request)
    nick = get_profile_name(s["id"]) or s["username"]
    return await guild_add_member(s["id"], nickname=nick)

# ---------- Inline UI (About Us last tab) ----------
HTML_TEMPLATE = """
<!doctype html><html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"/>
<title>GROWCB</title>
<style>
:root{--bg:#0b0f14;--bg2:#111825;--card:#121a2a;--text:#e7eefc;--muted:#9db0ce;--pri:#52a7ff;--good:#27d17f;--bad:#ff5c7a}
*{box-sizing:border-box}html,body{height:100%}
body{margin:0;background:linear-gradient(180deg,#0b0f14,#0b0f14 60%,#0b1826);color:var(--text);font:14px/1.35 system-ui,Segoe UI,Roboto,Ubuntu,sans-serif}
a{color:var(--pri);text-decoration:none}
header{position:sticky;top:0;z-index:50;background:rgba(11,15,20,.7);backdrop-filter:blur(8px);border-bottom:1px solid #1d2740}
.container{max-width:1024px;margin:0 auto;padding:12px}
.row{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.brand{font-weight:700;letter-spacing:.5px}
.tabs{display:flex;gap:8px;flex-wrap:wrap}
.tab{padding:8px 10px;border-radius:8px;background:#121a2a;color:#cfe1ff;cursor:pointer;border:1px solid #1c2840}
.tab.active{background:#1a2640;color:#fff;border-color:#29406e}
.right{margin-left:auto;display:flex;gap:8px;align-items:center}
.btn{padding:8px 10px;border-radius:8px;border:1px solid #2a3a5e;background:#162034;color:#cfe1ff;cursor:pointer}
.btn.primary{background:#1b2d4f;border-color:#2e4a80}
.btn.good{background:#123524;border-color:#1f6d45;color:#bdf6d6}
.btn.bad{background:#3a0f1a;border-color:#6e2340;color:#ffd0db}
.card{background:var(--card);border:1px solid #1d2740;border-radius:12px;padding:12px}
.grid{display:grid;gap:12px}
.grid.cols-2{grid-template-columns:1fr 1fr}
.grid.cols-3{grid-template-columns:1fr 1fr 1fr}
section{display:none;padding:16px 0}
section.active{display:block}
.small{color:var(--muted);font-size:12px}
input,select{background:#0f1523;border:1px solid #2a3a5e;color:#e7eefc;border-radius:8px;padding:8px}
input[type=number]{width:120px}
table{width:100%;border-collapse:collapse}
th,td{padding:6px;border-bottom:1px solid #20304e;font-size:13px}
ul{margin:0;padding-left:18px}
#chatList{max-height:360px;overflow:auto}
.chat-row{padding:6px 0;border-bottom:1px solid #1b2a45}
.chat-name{color:#abd0ff;cursor:pointer}
.badge{display:inline-block;padding:2px 6px;border-radius:6px;background:#20314f;color:#cfe1ff;font-size:11px;margin-left:6px}
.mono{font-family:ui-monospace, Menlo, Consolas, monospace}
footer{color:#6c84a7;text-align:center;padding:20px 0}
.kv{display:inline-grid;grid-template-columns:auto auto;gap:4px 10px}
.kv b{text-align:right}
</style>
</head>
<body>
<header>
  <div class="container row">
    <div class="brand">💎 GROWCB</div>
    <nav class="tabs" id="tabs">
      <div class="tab active" data-tab="games">Games</div>
      <div class="tab" data-tab="promo">Promo</div>
      <div class="tab" data-tab="referral">Referral</div>
      <div class="tab" data-tab="settings">Settings</div>
      <div class="tab" data-tab="chat">Chat</div>
      <div class="tab" data-tab="leaderboard">Leaderboard</div>
      <div class="tab" data-tab="about">About Us</div>
    </nav>
    <div class="right">
      <a id="joinGuild" class="btn" style="display:none" target="_blank" rel="noopener">Join Discord</a>
      <div id="userBox" class="small"></div>
    </div>
  </div>
</header>

<main class="container">
  <section id="games" class="active">
    <div class="grid cols-2">
      <div class="card">
        <h3>Account</h3>
        <div id="acctBox" class="small">Checking session…</div>
        <div id="balBox" class="small"></div>
      </div>
      <div class="card">
        <h3>Quick Links</h3>
        <div class="small">Use the tabs to explore games, chat, promos, and more.</div>
      </div>
    </div>
    <div class="grid cols-2" style="margin-top:12px">
      <div class="card">
        <h3>Crash</h3>
        <div class="small" id="crashState">Loading…</div>
        <div class="row" style="margin-top:8px">
          <input id="crBet" type="number" step="0.01" min="0.01" placeholder="Bet"/>
          <input id="crCash" type="number" step="0.01" min="1.01" value="2.0" />
          <button class="btn primary" id="crPlace">Place</button>
          <button class="btn good" id="crCashout">Cashout</button>
        </div>
        <div class="small" style="margin-top:8px">Last busts: <span id="crBusts">—</span></div>
      </div>
      <div class="card">
        <h3>Mines</h3>
        <div class="row">
          <input id="mnBet" type="number" step="0.01" min="0.01" placeholder="Bet"/>
          <select id="mnMines">
            <option value="3">3 mines</option><option value="5">5</option><option value="8">8</option>
          </select>
          <button class="btn primary" id="mnStart">Start</button>
          <button class="btn good" id="mnCashout">Cashout</button>
        </div>
        <div class="grid cols-3" style="margin-top:8px" id="mnGrid"></div>
        <div class="small" id="mnInfo"></div>
      </div>
    </div>
  </section>

  <section id="promo">
    <div class="card">
      <h3>Redeem Promo</h3>
      <div class="row">
        <input id="promoCode" placeholder="enter code"/>
        <button id="promoBtn" class="btn primary">Redeem</button>
      </div>
      <div id="promoMsg" class="small"></div>
      <h4 style="margin-top:14px">Your redemptions</h4>
      <table id="promoTable"><thead><tr><th>Code</th><th>Redeemed at</th></tr></thead><tbody></tbody></table>
    </div>
  </section>

  <section id="referral">
    <div class="card">
      <h3>Your referral</h3>
      <div class="row">
        <input id="refName" placeholder="set your name"/>
        <button id="refSet" class="btn primary">Save</button>
      </div>
      <div id="refInfo" class="small"></div>
    </div>
  </section>

  <section id="settings">
    <div class="card">
      <h3>Settings</h3>
      <div class="row">
        <label class="small"><input type="checkbox" id="anonChk"/> Anonymous public profile</label>
        <button id="saveAnon" class="btn primary">Save</button>
        <a id="logoutBtn" class="btn bad" href="/logout">Logout</a>
      </div>
      <div class="small" id="setMsg"></div>
    </div>
  </section>

  <section id="chat">
    <div class="card">
      <h3>Chat</h3>
      <div id="chatList"></div>
      <div class="row">
        <input id="chatMsg" placeholder="Type a message…" style="flex:1"/>
        <button id="chatSend" class="btn primary">Send</button>
      </div>
    </div>
  </section>

  <section id="leaderboard">
    <div class="card">
      <h3>Leaderboard</h3>
      <table id="lbTable"><thead><tr><th>User</th><th>Total Wagered</th></tr></thead><tbody></tbody></table>
    </div>
  </section>

  <section id="about">
    <div class="card">
      <h3>About Us</h3>
      <p class="small">
        Welcome to GROWCB.  
        This section is for your own custom text.  
        You can edit it later in <code>main.py</code> under the About tab.
      </p>
    </div>
  </section>
</main>

<footer><div class="container">© 2025 GROWCB</div></footer>

<script>
const tabs = document.querySelectorAll(".tab");
tabs.forEach(t => t.onclick = () => {
  tabs.forEach(x => x.classList.remove("active"));
  document.querySelectorAll("main section").forEach(s => s.classList.remove("active"));
  t.classList.add("active");
  document.getElementById(t.dataset.tab).classList.add("active");
});

async function fetchJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}

// --- Session & Balance ---
async function loadMe() {
  try {
    const me = await fetchJSON("/api/me");
    document.getElementById("userBox").innerHTML =
      `<img src="${me.avatar_url}" style="width:20px;border-radius:50%;vertical-align:middle;margin-right:6px"/> ${me.username}`;
    if (me.in_guild) document.getElementById("joinGuild").style.display = "none";
    else {
      document.getElementById("joinGuild").style.display = "";
      document.getElementById("joinGuild").href = "${DISCORD_INVITE}";
    }
    const bal = await fetchJSON("/api/balance");
    document.getElementById("acctBox").textContent = "Logged in as " + me.username;
    document.getElementById("balBox").textContent = "Balance: " + bal.balance.toFixed(2) + " DL";
  } catch {
    document.getElementById("acctBox").innerHTML = '<a href="/login" class="btn primary">Login with Discord</a>';
  }
}
loadMe();

// --- Crash ---
async function loadCrash() {
  try {
    const st = await fetchJSON("/api/crash/state");
    document.getElementById("crashState").textContent = "Phase: " + st.phase + (st.current_multiplier ? (" ×" + st.current_multiplier) : "");
    document.getElementById("crBusts").textContent = (st.last_busts||[]).join(", ");
  } catch(e) {
    document.getElementById("crashState").textContent = "Error loading crash";
  }
}
document.getElementById("crPlace").onclick = async () => {
  try {
    const b = document.getElementById("crBet").value;
    const c = document.getElementById("crCash").value;
    await fetchJSON("/api/crash/place", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({bet:b,cashout:c})});
    loadCrash();
  } catch(e){ alert(e); }
};
document.getElementById("crCashout").onclick = async () => {
  try { await fetchJSON("/api/crash/cashout",{method:"POST"}); loadCrash(); } catch(e){ alert(e); }
};
setInterval(loadCrash, 5000);

// --- Mines ---
let mnState=null;
document.getElementById("mnStart").onclick = async () => {
  try {
    const b=document.getElementById("mnBet").value;
    const m=document.getElementById("mnMines").value;
    mnState=await fetchJSON("/api/mines/start",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({bet:b,mines:m})});
    renderMines();
  } catch(e){ alert(e); }
};
document.getElementById("mnCashout").onclick = async () => {
  try{ await fetchJSON("/api/mines/cashout",{method:"POST"}); mnState=null; renderMines(); }catch(e){ alert(e); }
};
function renderMines(){
  const g=document.getElementById("mnGrid"); g.innerHTML="";
  if(!mnState||!mnState.cells) return;
  mnState.cells.forEach((c,i)=>{
    const d=document.createElement("div");
    d.className="card"; d.style.padding="20px"; d.style.textAlign="center"; d.textContent=c==="?"?"?":c;
    d.onclick=async()=>{ try{ mnState=await fetchJSON("/api/mines/pick?cell="+i,{method:"POST"}); renderMines(); }catch(e){ alert(e);} };
    g.appendChild(d);
  });
  document.getElementById("mnInfo").textContent="Bet: "+mnState.bet+" | Mines: "+mnState.mines;
}

// --- Chat ---
let lastChatId=0;
async function loadChat(){
  try {
    const data=await fetchJSON("/api/chat/fetch?since="+lastChatId);
    const list=document.getElementById("chatList");
    data.rows.forEach(r=>{
      lastChatId=r.id;
      const div=document.createElement("div");
      div.className="chat-row";
      div.innerHTML=`<span class="chat-name" data-uid="${r.user_id}">${r.username}</span> <span class="badge">Lv${r.level}</span>: ${r.text}`;
      list.appendChild(div);
    });
    list.scrollTop=list.scrollHeight;
  }catch(e){}
}
document.getElementById("chatSend").onclick=async()=>{
  try{
    const t=document.getElementById("chatMsg").value;
    if(!t) return;
    await fetchJSON("/api/chat/send",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text:t})});
    document.getElementById("chatMsg").value="";
    loadChat();
  }catch(e){alert(e);}
};
setInterval(loadChat,4000);

// --- Leaderboard ---
async function loadLb(){
  const d=await fetchJSON("/api/leaderboard");
  const tb=document.querySelector("#lbTable tbody"); tb.innerHTML="";
  d.rows.forEach(r=>{
    const tr=document.createElement("tr");
    tr.innerHTML="<td>"+r.display_name+"</td><td>"+r.total_wagered.toFixed(2)+"</td>";
    tb.appendChild(tr);
  });
}
setInterval(loadLb,10000); loadLb();
</script>
</body></html>
"""

@app.get("/")
async def index():
    return HTMLResponse(HTML_TEMPLATE)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
