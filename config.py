import os, datetime
from decimal import Decimal, ROUND_DOWN, getcontext

# Precision for Decimal math
getcontext().prec = 28
TWO = Decimal("0.01")

def D(x) -> Decimal:
    if isinstance(x, Decimal): return x
    return Decimal(str(x))

def q2(x: Decimal) -> Decimal:
    return D(x).quantize(TWO, rounding=ROUND_DOWN)

UTC = datetime.timezone.utc
def now_utc() -> datetime.datetime:
    return datetime.datetime.now(UTC)

def iso(dt):
    if dt is None: return None
    return dt.astimezone(UTC).isoformat()

# ---------- ENV / Config ----------
CLIENT_ID = os.getenv("DISCORD_CLIENT_ID", "")
CLIENT_SECRET = os.getenv("DISCORD_CLIENT_SECRET") or os.getenv("CLIENT_SECRET", "")
OAUTH_REDIRECT = os.getenv("OAUTH_REDIRECT", "")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")
PORT = int(os.getenv("PORT", "8080"))
DISCORD_API = "https://discord.com/api"
OWNER_ID = int(os.getenv("OWNER_ID", "1128658280546320426"))
DATABASE_URL = os.getenv("DATABASE_URL")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
GUILD_ID = int(os.getenv("GUILD_ID", "0"))
DISCORD_INVITE = os.getenv("DISCORD_INVITE", "")

# Site/business constants
GEM = "💎"
MIN_BET = Decimal("1.00")
MAX_BET = Decimal("1000000.00")
BETTING_SECONDS = 10

HOUSE_EDGE_CRASH = Decimal(os.getenv("HOUSE_EDGE_CRASH", "0.06"))
HOUSE_EDGE_MINES = Decimal(os.getenv("HOUSE_EDGE_MINES", "0.03"))

# Chat gate (fixed so chat works by default)
CHAT_MIN_LEVEL = int(os.getenv("CHAT_MIN_LEVEL", "1"))
