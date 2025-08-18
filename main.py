# mines.py
import os, random, secrets, hashlib
from decimal import Decimal
from typing import Optional, Dict, List

from db_utils import with_conn, D, q2

HOUSE_EDGE_MINES = Decimal(os.getenv("HOUSE_EDGE_MINES", "0.03"))
MIN_BET = Decimal("1.00")
MAX_BET = Decimal("1000000.00")

def mines_random_board(mines: int) -> str:
    idxs = list(range(25))
    random.shuffle(idxs)
    mines_set = set(idxs[:mines])
    return ''.join('1' if i in mines_set else '0' for i in range(25))

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def picks_count_from_bitmask(mask: int) -> int:
    try:
        return mask.bit_count()
    except AttributeError:
        # Python < 3.8 fallback
        return bin(mask).count("1")

def mines_multiplier(mines: int, picks_count: int) -> float:
    if picks_count <= 0: return 1.0
    total = Decimal("1.0")
    for i in range(picks_count):
        total *= D(25 - i) / D(max(1, 25 - mines - i))
        total *= (Decimal("1.0") - HOUSE_EDGE_MINES)
    return float(total)

@with_conn
def mines_start(cur, user_id: str, bet: Decimal, mines: int):
    if bet < MIN_BET or bet > MAX_BET: raise ValueError(f"Bet must be between {MIN_BET:.2f} and {MAX_BET:.2f}")
    if mines < 1 or mines > 24: raise ValueError("Mines must be between 1 and 24")

    # only one active game
    cur.execute("SELECT 1 FROM mines_games WHERE user_id=%s AND status='active'", (user_id,))
    if cur.fetchone(): raise ValueError("You already have an active Mines game")

    # funds
    cur.execute("INSERT INTO balances(user_id,balance) VALUES (%s,0) ON CONFLICT(user_id) DO NOTHING", (user_id,))
    cur.execute("SELECT balance FROM balances WHERE user_id=%s FOR UPDATE", (user_id,))
    bal = D(cur.fetchone()[0])
    if bal < bet: raise ValueError("Insufficient balance")
    cur.execute("UPDATE balances SET balance=balance-%s WHERE user_id=%s", (q2(bet), user_id))

    # commit-reveal
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
