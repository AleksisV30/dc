# db_utils.py
import os, datetime
from decimal import Decimal, ROUND_DOWN, getcontext
import psycopg

getcontext().prec = 28

# DB connection string
DATABASE_URL = os.getenv("DATABASE_URL")

# Decimal helpers
TWO = Decimal("0.01")
def D(x):
    if isinstance(x, Decimal): return x
    return Decimal(str(x))
def q2(x: Decimal) -> Decimal:
    return D(x).quantize(TWO)

# Time helpers
UTC = datetime.timezone.utc
def now_utc() -> datetime.datetime:
    return datetime.datetime.now(UTC)
def iso(dt):
    if dt is None: return None
    return dt.astimezone(UTC).isoformat()

# DB connection decorator
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

# Used by games to make sure profile row exists inside their transactions
def ensure_profile_row_cur(cur, user_id: str, owner_id_env: str = None):
    owner_id_env = str(owner_id_env or os.getenv("OWNER_ID", ""))
    role = 'owner' if str(user_id) == owner_id_env else 'member'
    default_name = f"user_{str(user_id)[-4:]}"
    cur.execute("""
        INSERT INTO profiles(user_id, display_name, name_lower, role, is_anon)
        VALUES (%s,%s,%s,%s,FALSE)
        ON CONFLICT (user_id) DO NOTHING
    """, (str(user_id), default_name, default_name, role))
