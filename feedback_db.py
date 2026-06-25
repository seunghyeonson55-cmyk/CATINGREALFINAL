"""
피드백(정답 신호) 저장소 — SQLite
================================================
■ 가장 중요한 설계 원칙(절대 어기지 말 것)
  - '얼굴 분석(인상 점수)'은 고정된 자(ruler)다. 이 파일은 그 점수를 **절대 건드리지 않는다.**
  - 여기 쌓는 것은 오직 "검색 결과를 어떤 순서로 보여줄지"(=랭킹)를 나중에 학습할 신호다.
    → 학습은 '랭킹'에만 적용. '인상 점수'에는 적용 안 함. (둘을 완전히 분리)

■ 지금 단계
  - '기록만' 한다. 학습(재랭킹)은 데이터가 충분히 쌓인 뒤 별도로 켠다.
  - 이 파일은 신호를 차곡차곡 쌓는 '창고'일 뿐, 검색 순위를 바꾸지 않는다.

■ 신호의 종류
  - 명시적(explicit): 👍 up / 👎 down  — 감독이 직접 누른 평가
  - 암묵적(implicit): click(자세히 봄) / shortlist(찜) / contact(컨택)
    → 말로 안 해도 '관심의 세기'를 드러내는 행동. 보통 contact > shortlist > click 순으로 강한 신호.

■ 테이블 (CLAUDE.md 스키마 초안과 동일한 방향)
  searches : 어떤 '검색어'로 검색했는가
  events   : 그 검색에서 '누구'에게 '무슨 신호'를 줬는가 (+ 그때의 순위·점수)
"""
from __future__ import annotations
import os
import sqlite3
import datetime
import secrets
import hashlib
import hmac
import numpy as np

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "feedback.db")

# ■ 학습 스위치 — 지금은 '기록만' 한다(꺼짐). 데이터가 충분히 쌓인 뒤에만 True로 켠다.
#   이 값이 False인 동안 아래 재랭킹 함수들은 항상 '보정 없음(0)'을 돌려준다.
LEARNING_ENABLED = False

# 신호 종류와 '관심 세기'(나중에 재랭킹 가중치로 쓸 참고값. 지금은 저장만, 학습엔 미사용)
SIGNAL_WEIGHTS = {
    "down": -1.0,      # 명시적 부정
    "click": 0.3,      # 자세히 봄(약한 관심)
    "shortlist": 0.7,  # 찜(중간 관심)
    "up": 1.0,         # 명시적 긍정
    "contact": 1.5,    # 컨택(가장 강한 관심 — 실제 섭외 의사)
}


# ---------- 벡터(의미 좌표) ↔ 저장용 바이트 변환 ----------
# 검색어의 임베딩 벡터를 DB에 그대로(BLOB) 넣고 빼기 위한 도우미.
# float32로 통일해 용량을 아낀다(검색 엔진 벡터는 정규화돼 있어 코사인=내적).

def _vec_to_blob(vec) -> bytes | None:
    if vec is None:
        return None
    return np.asarray(vec, dtype=np.float32).tobytes()


def _blob_to_vec(blob) -> np.ndarray | None:
    if not blob:
        return None
    return np.frombuffer(blob, dtype=np.float32)


def _unit(v: np.ndarray) -> np.ndarray:
    """방향만 남기려고 길이를 1로 맞춘다(0 벡터는 그대로)."""
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


# ======================================================================
#  저장 백엔드 선택 — Supabase(Postgres)가 설정돼 있으면 그것, 아니면 로컬 SQLite.
#  · 서버(Streamlit Cloud): st.secrets["supabase"] 가 있으면 Postgres 사용
#    → 재배포·잠들어도 데이터가 안 사라진다(영구 저장).
#  · 로컬/테스트: 시크릿이 없거나 CATING_FORCE_SQLITE=1 이면 기존 SQLite 파일 사용.
# ======================================================================
_PG_CFG = None
_PG_READY = False


def _pg_config():
    """Supabase 접속 정보를 (있으면) dict로, 없으면 None."""
    if os.environ.get("CATING_FORCE_SQLITE"):
        return None
    # 1) Streamlit secrets
    try:
        import streamlit as st
        if "supabase" in st.secrets:
            s = st.secrets["supabase"]
            return dict(host=s["host"], port=int(s.get("port", 5432)),
                        user=s["user"], password=s["password"],
                        dbname=s.get("dbname", "postgres"))
    except Exception:
        pass
    # 2) 환경변수(비-Streamlit 컨텍스트/테스트용)
    if os.environ.get("SUPABASE_HOST"):
        return dict(host=os.environ["SUPABASE_HOST"],
                    port=int(os.environ.get("SUPABASE_PORT", "5432")),
                    user=os.environ["SUPABASE_USER"],
                    password=os.environ["SUPABASE_PASSWORD"],
                    dbname=os.environ.get("SUPABASE_DB", "postgres"))
    return None


def _ensure_backend():
    global _PG_CFG, _PG_READY
    if not _PG_READY:
        _PG_CFG = _pg_config()
        _PG_READY = True
    return _PG_CFG


def _xlate(sql: str) -> str:
    """SQLite 문법 SQL을 Postgres용으로 살짝 바꾼다."""
    sql = sql.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "BIGSERIAL PRIMARY KEY")
    sql = sql.replace(" BLOB", " BYTEA")
    sql = sql.replace("?", "%s")
    return sql


def _split_sql(script: str) -> list:
    return [s for s in (part.strip() for part in script.split(";")) if s]


# 연결 풀(Postgres 전용) — 매 쿼리 새 연결을 여는 비용(네트워크 핸드셰이크)을 없애 빠르게.
_POOL = None


def _get_pool():
    """Postgres 연결 풀(프로세스당 1개). 실패 시 예외 → 호출부가 직접연결로 폴백."""
    global _POOL
    if _POOL is None:
        from psycopg_pool import ConnectionPool
        cfg = _ensure_backend()
        _POOL = ConnectionPool(
            conninfo="",
            kwargs=dict(autocommit=True, connect_timeout=15, **cfg),
            min_size=1, max_size=5, timeout=20,
            max_idle=60, max_lifetime=600, open=True,
        )
    return _POOL


class _Conn:
    """sqlite3 연결처럼 .execute(...) / .executescript(...) 를 쓰게 감싼 래퍼.
    백엔드가 Postgres면 psycopg(풀에서 빌림), 아니면 sqlite3."""

    def __init__(self):
        cfg = _ensure_backend()
        if cfg is not None:
            self.pg = True
            self._pooled = False
            try:
                self._pool = _get_pool()
                self._c = self._pool.getconn()
                self._pooled = True
            except Exception:
                import psycopg            # 풀 실패 시 직접 연결로 폴백(안전)
                self._c = psycopg.connect(autocommit=True, connect_timeout=15, **cfg)
        else:
            self.pg = False
            self._pooled = False
            os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
            self._c = sqlite3.connect(DB_PATH)
            self._c.execute("PRAGMA journal_mode=WAL")   # 동시 접근에 안정적

    def execute(self, sql, params=()):
        if self.pg:
            sql2 = _xlate(sql)
            return self._c.execute(sql2, tuple(params)) if params else self._c.execute(sql2)
        return self._c.execute(sql, params)

    def executescript(self, script):
        if self.pg:
            for stmt in _split_sql(script):
                self._c.execute(_xlate(stmt))
            return None
        return self._c.executescript(script)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.pg:
            if self._pooled:
                try:
                    self._pool.putconn(self._c)      # 풀에 반납(연결 재사용)
                except Exception:
                    try:
                        self._c.close()
                    except Exception:
                        pass
            else:
                try:
                    self._c.close()
                except Exception:
                    pass
        else:
            try:
                if exc_type is None:
                    self._c.commit()
            finally:
                try:
                    self._c.close()
                except Exception:
                    pass
        return False


def _conn():
    return _Conn()


def storage_mode() -> str:
    """'postgres'(영구 저장) 또는 'sqlite'(임시). 화면에서 저장 상태 표시용."""
    return "postgres" if _ensure_backend() is not None else "sqlite"


def _insert(c, sql, params=()):
    """INSERT 후 새 행의 id를 돌려준다(양 백엔드 호환).
    Postgres는 RETURNING id, SQLite는 lastrowid."""
    if c.pg:
        cur = c.execute(sql.rstrip().rstrip(";") + " RETURNING id", params)
        row = cur.fetchone()
        return row[0] if row else None
    cur = c.execute(sql, params)
    return cur.lastrowid


def _alter_add(c, table, col, coltype):
    """없을 수 있는 칸을 추가(이미 있으면 통과). 양 백엔드 호환."""
    if c.pg:
        c.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {coltype}")
    else:
        try:
            c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")
        except sqlite3.OperationalError:
            pass


def _upsert(c, table, conflict_col, cols, params):
    """있으면 갱신·없으면 삽입(SQLite=INSERT OR REPLACE, PG=ON CONFLICT)."""
    placeholders = ",".join(["?"] * len(cols))
    collist = ",".join(cols)
    if c.pg:
        updates = ",".join(f"{col}=EXCLUDED.{col}" for col in cols if col != conflict_col)
        sql = (f"INSERT INTO {table}({collist}) VALUES({placeholders}) "
               f"ON CONFLICT ({conflict_col}) DO UPDATE SET {updates}")
    else:
        sql = f"INSERT OR REPLACE INTO {table}({collist}) VALUES({placeholders})"
    c.execute(sql, params)


def init_db():
    """테이블이 없으면 만든다(있으면 그대로 둠). 앱 시작 때 한 번 호출."""
    with _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS searches (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text     TEXT NOT NULL,      -- 감독이 입력한 검색어 원문
                expanded_text  TEXT,               -- AI가 풀어 쓴 해석문(있으면)
                query_embedding BLOB,              -- (나중에 채움) 비슷한 검색어끼리 묶기용
                created_at     TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                search_id   INTEGER,               -- 어떤 검색에서 일어난 신호인가(searches.id)
                actor_uid   TEXT,                  -- 대상 지원자 식별자
                actor_name  TEXT,
                type        TEXT NOT NULL,         -- click/shortlist/contact/up/down
                rank        INTEGER,               -- 그 검색에서 보여준 순위(1=맨 위)
                score       REAL,                  -- 그때의 분위기 점수(코사인)
                created_at  TEXT NOT NULL,
                FOREIGN KEY (search_id) REFERENCES searches(id)
            );
            CREATE INDEX IF NOT EXISTS idx_events_search ON events(search_id);
            CREATE INDEX IF NOT EXISTS idx_events_actor  ON events(actor_uid);
            CREATE INDEX IF NOT EXISTS idx_events_type   ON events(type);

            -- 배우(지원자)별 피드백 누적 — '이 배우는 어떤 의미의 검색에서 평이 좋은가'를
            -- 한 곳에 모은다. 검색어 글자는 매번 달라도, 신호가 들어올 때마다
            --   pref_vec += (신호세기) × (그 검색의 의미 벡터)
            -- 로 더해두면, pref_vec의 '방향'이 곧 그 배우가 호평받는 검색 의미가 된다.
            -- (배우는 고정이라 데이터가 흩어지지 않고 여기 쌓인다)
            CREATE TABLE IF NOT EXISTS actor_signals (
                actor_uid    TEXT PRIMARY KEY,    -- 지원자 식별자
                actor_name   TEXT,
                pref_vec     BLOB,                -- Σ (신호세기 × 검색벡터)  ← 호평 '방향'
                sum_weight   REAL DEFAULT 0,      -- Σ 신호세기(부호 포함) — 전반적 호/불호
                abs_weight   REAL DEFAULT 0,      -- Σ |신호세기| — 데이터가 얼마나 모였나(신뢰도)
                pos_count    INTEGER DEFAULT 0,   -- 긍정 신호 수(up/click/shortlist/contact)
                neg_count    INTEGER DEFAULT 0,   -- 부정 신호 수(down)
                event_count  INTEGER DEFAULT 0,   -- 반영된 신호 총 건수
                updated_at   TEXT
            );

            -- 캐스팅 공고(감독이 올림 → 배우가 봄). 검색/랭킹과 무관한 단순 게시판.
            CREATE TABLE IF NOT EXISTS casting_calls (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                director_uid  TEXT,                -- 올린 감독 식별자
                director_name TEXT,
                title         TEXT NOT NULL,        -- 공고 제목
                production    TEXT,                 -- 작품명
                synopsis      TEXT,                 -- 작품 시놉시스(줄거리)
                role_name     TEXT,                 -- 찾는 배역(짧게)
                gender        TEXT,                 -- (구버전 호환) 모집 성별
                age_min       INTEGER,
                age_max       INTEGER,
                deadline      TEXT,                 -- 마감일(자유 텍스트)
                description   TEXT,                 -- 원하는 배역 이미지/조건 설명
                created_at    TEXT NOT NULL,
                active        INTEGER DEFAULT 1     -- 1=모집중, 0=마감
            );
            CREATE INDEX IF NOT EXISTS idx_calls_director ON casting_calls(director_uid);
            """
        )
        # 구버전 DB에 없을 수 있는 칸들을 안전하게 추가(이미 있으면 통과).
        _alter_add(c, "casting_calls", "synopsis", "TEXT")
        # 공고에 '필수 제출 항목'(JSON 배열)과 '동영상 링크 필수' 여부.
        _alter_add(c, "casting_calls", "required_fields", "TEXT")
        _alter_add(c, "casting_calls", "video_required", "INTEGER DEFAULT 0")
        # 모집 배역 목록(JSON 배열) — 배우가 지원할 때 어느 배역에 지원할지 먼저 고른다.
        _alter_add(c, "casting_calls", "roles", "TEXT")
        # 공개 링크용 랜덤 토큰 — 순번(1,2,3) 노출/추측 방지. 링크는 ?apply=<token>.
        _alter_add(c, "casting_calls", "token", "TEXT")
        # 회원 프로필(감독·배우) — 관리자 페이지에서 전체 조회/삭제할 수 있도록 DB에 보관.
        # (검색/랭킹과 무관. 세션에만 있던 프로필을 여기에도 저장해 여러 세션에서 공유)
        c.execute(
            """CREATE TABLE IF NOT EXISTS profiles (
                uid         TEXT PRIMARY KEY,   -- 로그인 사용자 식별자
                role        TEXT,               -- director / actor
                name        TEXT,
                email       TEXT,
                data        TEXT,               -- 프로필 전체(JSON 문자열)
                created_at  TEXT,
                updated_at  TEXT
            )"""
        )
        # 배우 지원자(사진→AI 인상분석 카드 + 검색 임베딩) — 세션에만 있던 것을 DB에 영속화.
        # 앱이 켜질 때 여기서 불러와 모든 세션이 같은 지원자 풀을 공유하게 한다.
        c.execute(
            """CREATE TABLE IF NOT EXISTS applicants (
                uid         TEXT PRIMARY KEY,   -- 지원자 식별자
                name        TEXT,
                data        TEXT,               -- actor dict 전체(JSON: 사진b64·인상·점수 등)
                emb         BLOB,               -- 검색용 임베딩 벡터(float32)
                created_at  TEXT
            )"""
        )
        # 감독 ↔ 배우 메신저. 한 대화는 (감독, 배우) 한 쌍으로 묶인다.
        # 감독이 '배우 탐색'에서 본인 등록 배우에게 먼저 말을 걸어 시작한다.
        c.execute(
            """CREATE TABLE IF NOT EXISTS messages (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                director_uid  TEXT,             -- 감독(로그인 사용자) 식별자
                actor_uid     TEXT,             -- 배우 지원자(카드) 식별자
                director_name TEXT,
                actor_name    TEXT,
                sender        TEXT,             -- 'director' / 'actor'
                text          TEXT,
                created_at    TEXT
            )"""
        )
        # 공고 '방'에 들어온 지원서. 링크(?apply=공고번호)로 들어와 작성한다.
        # 비회원도 지원 가능(actor_login 비어 있음). 로그인하면 본인 식별자가 들어가
        # 결과 알림을 받을 수 있다. status: pending(검토중)/accepted(합격)/rejected(불합격)
        c.execute(
            """CREATE TABLE IF NOT EXISTS applications (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                call_id       INTEGER,          -- 어느 공고에 지원했나
                actor_uid     TEXT,             -- 배우 지원자 카드 uid(있으면)
                actor_login   TEXT,             -- 로그인 사용자 식별자(비회원이면 비어 있음)
                applicant_name TEXT,
                data          TEXT,             -- 지원서 전체(JSON: 연락처·답변 등)
                video_link    TEXT,             -- 동영상 링크(자기소개/연기영상)
                status        TEXT DEFAULT 'pending',
                created_at    TEXT
            )"""
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_app_call ON applications(call_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_app_login ON applications(actor_login)")
        # 앱 알림함. 합격/불합격 결과나 일정 안내 등을 사용자에게 보낸다.
        # (카톡/이메일 알림은 카톡 로그인 붙인 뒤 단계. 지금은 앱 안에서만.)
        c.execute(
            """CREATE TABLE IF NOT EXISTS notifications (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_uid    TEXT,               -- 받는 사람(로그인 식별자 또는 배우 uid)
                kind        TEXT,               -- result/schedule/message 등
                title       TEXT,
                body        TEXT,
                read        INTEGER DEFAULT 0,  -- 0=안읽음, 1=읽음
                created_at  TEXT
            )"""
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_noti_user ON notifications(user_uid)")
        # 알림이 어떤 공고에 대한 것인지(클릭하면 공고 다시 보기). 공고 id를 담는다.
        _alter_add(c, "notifications", "ref", "TEXT")
        # 오디션 일정: 감독이 후보 시간(슬롯)을 올리고, 배우가 그중 하나를 고른다.
        c.execute(
            """CREATE TABLE IF NOT EXISTS audition_slots (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                call_id     INTEGER,            -- 어느 공고의 오디션인가
                director_uid TEXT,
                label       TEXT,               -- 후보 시간(자유 텍스트: '6/20 오후 2시' 등)
                place       TEXT,               -- 장소(선택)
                capacity    INTEGER DEFAULT 0,  -- 정원(0=무제한)
                created_at  TEXT
            )"""
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_slot_call ON audition_slots(call_id)")
        c.execute(
            """CREATE TABLE IF NOT EXISTS audition_picks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                slot_id     INTEGER,
                call_id     INTEGER,
                actor_uid   TEXT,               -- 배우 식별자(로그인/카드)
                actor_name  TEXT,
                created_at  TEXT
            )"""
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_pick_call ON audition_picks(call_id)")
        # 회원 계정(이메일+비밀번호) — 회원가입/로그인용. 비밀번호는 해시로만 저장.
        c.execute(
            """CREATE TABLE IF NOT EXISTS users (
                email       TEXT PRIMARY KEY,   -- 소문자 이메일(아이디)
                pw_salt     TEXT,
                pw_hash     TEXT,
                name        TEXT,
                verified    INTEGER DEFAULT 1,  -- 이메일 인증 완료 여부
                created_at  TEXT
            )"""
        )
        # 로그인 유지용 세션 토큰 — 쿠키에 이 토큰을 심어 새로고침·재접속해도 로그인 유지.
        c.execute(
            """CREATE TABLE IF NOT EXISTS sessions (
                token       TEXT PRIMARY KEY,
                email       TEXT,
                expires_at  TEXT,
                created_at  TEXT
            )"""
        )


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


# ==================================================================
#  회원 계정(이메일+비밀번호) — 회원가입/로그인. 비밀번호는 해시(pbkdf2)로만 저장.
# ==================================================================

def _hash_pw(password: str, salt: str | None = None) -> tuple[str, str]:
    if salt is None:
        salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", (password or "").encode("utf-8"),
                            bytes.fromhex(salt), 200_000)
    return salt, h.hex()


def user_exists(email: str) -> bool:
    em = (email or "").strip().lower()
    if not em:
        return False
    with _conn() as c:
        return c.execute("SELECT 1 FROM users WHERE email=?", (em,)).fetchone() is not None


def get_user(email: str) -> dict | None:
    em = (email or "").strip().lower()
    if not em:
        return None
    with _conn() as c:
        row = c.execute(
            "SELECT email, name, verified, created_at FROM users WHERE email=?", (em,)
        ).fetchone()
    if not row:
        return None
    return {"email": row[0], "name": row[1], "verified": bool(row[2]), "created_at": row[3]}


def create_user(email: str, password: str, name: str = "") -> bool:
    """새 계정 생성. 이미 있으면 False. 비밀번호는 해시로 저장."""
    em = (email or "").strip().lower()
    if not em or not password:
        return False
    if user_exists(em):
        return False
    salt, h = _hash_pw(password)
    with _conn() as c:
        c.execute(
            "INSERT INTO users(email, pw_salt, pw_hash, name, verified, created_at) "
            "VALUES(?,?,?,?,1,?)",
            (em, salt, h, (name or "").strip(), _now()),
        )
    return True


def verify_password(email: str, password: str) -> bool:
    """이메일+비밀번호가 맞으면 True."""
    em = (email or "").strip().lower()
    if not em or not password:
        return False
    with _conn() as c:
        row = c.execute(
            "SELECT pw_salt, pw_hash FROM users WHERE email=?", (em,)).fetchone()
    if not row or not row[0] or not row[1]:
        return False
    _, h = _hash_pw(password, row[0])
    return hmac.compare_digest(h, row[1])


def set_password(email: str, password: str) -> bool:
    """비밀번호 변경/설정(계정이 있을 때)."""
    em = (email or "").strip().lower()
    if not em or not password or not user_exists(em):
        return False
    salt, h = _hash_pw(password)
    with _conn() as c:
        c.execute("UPDATE users SET pw_salt=?, pw_hash=? WHERE email=?", (salt, h, em))
    return True


# ---- 로그인 유지(세션 토큰) ----
def create_session(email: str, days: int = 30) -> str | None:
    """로그인 유지 토큰 발급. 쿠키에 저장해 두면 days일 동안 로그인 유지."""
    em = (email or "").strip().lower()
    if not em:
        return None
    tok = secrets.token_urlsafe(32)
    exp = (datetime.datetime.now() + datetime.timedelta(days=days)).isoformat(timespec="seconds")
    with _conn() as c:
        c.execute("INSERT INTO sessions(token, email, expires_at, created_at) VALUES(?,?,?,?)",
                  (tok, em, exp, _now()))
    return tok


def get_session_email(token: str) -> str | None:
    """세션 토큰이 유효하면 그 이메일을, 만료/없음이면 None."""
    if not token:
        return None
    with _conn() as c:
        row = c.execute("SELECT email, expires_at FROM sessions WHERE token=?", (token,)).fetchone()
    if not row:
        return None
    email, exp = row[0], row[1]
    try:
        if exp and datetime.datetime.fromisoformat(exp) < datetime.datetime.now():
            return None
    except Exception:
        pass
    return email


def delete_session(token: str) -> None:
    if not token:
        return
    with _conn() as c:
        c.execute("DELETE FROM sessions WHERE token=?", (token,))


def get_or_create_search(query_text: str, expanded_text: str | None = None,
                         embedding=None) -> int | None:
    """검색어 한 건을 searches에 기록(같은 검색어면 기존 행 재사용). search_id 반환.
    검색어가 비어 있으면 기록하지 않고 None.
    embedding(검색 의미 벡터)을 주면 함께 저장한다 — 비슷한 의미 검색끼리 묶는 핵심."""
    q = (query_text or "").strip()
    if not q:
        return None
    blob = _vec_to_blob(embedding)
    with _conn() as c:
        row = c.execute(
            "SELECT id, query_embedding FROM searches WHERE query_text=? ORDER BY id LIMIT 1", (q,)
        ).fetchone()
        if row:
            # 기존 행에 벡터가 비어 있고 이번에 벡터가 들어오면 채워준다.
            if blob is not None and not row[1]:
                c.execute("UPDATE searches SET query_embedding=? WHERE id=?", (blob, row[0]))
            return row[0]
        return _insert(
            c,
            "INSERT INTO searches(query_text, expanded_text, query_embedding, created_at) "
            "VALUES(?,?,?,?)",
            (q, expanded_text, blob, _now()),
        )


def set_search_embedding(search_id, embedding) -> None:
    """이미 만든 검색 행에 의미 벡터를 채워 넣는다(비어 있을 때만).
    앱은 search_id를 먼저 만들고 벡터를 그 다음에 계산하므로 이 함수로 뒤늦게 붙인다."""
    if search_id is None or embedding is None:
        return
    blob = _vec_to_blob(embedding)
    with _conn() as c:
        c.execute(
            "UPDATE searches SET query_embedding=? WHERE id=? AND query_embedding IS NULL",
            (blob, search_id),
        )


def log_event(search_id, actor_uid, actor_name, etype,
              rank: int | None = None, score: float | None = None):
    """신호 한 건을 events에 기록한다. (학습은 하지 않음 — 저장만)
    덤으로, 그 검색에 의미 벡터가 저장돼 있고 대상 배우가 있으면
    actor_signals(배우별 누적)에도 '검색 의미 × 신호세기'를 더해 둔다.
    → 이것도 '기록'일 뿐, 검색 순위는 바꾸지 않는다."""
    with _conn() as c:
        c.execute(
            "INSERT INTO events(search_id, actor_uid, actor_name, type, rank, score, created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (search_id, actor_uid, actor_name, etype,
             rank, float(score) if score is not None else None, _now()),
        )
        # 배우별 누적: 대상 배우가 있고, 점수화 가능한 신호이며, 그 검색에 벡터가 있을 때만.
        if actor_uid and etype in SIGNAL_WEIGHTS and search_id is not None:
            row = c.execute(
                "SELECT query_embedding FROM searches WHERE id=?", (search_id,)
            ).fetchone()
            qvec = _blob_to_vec(row[0]) if row else None
            if qvec is not None:
                _accumulate_actor_signal(c, actor_uid, actor_name, etype, qvec)


def _accumulate_actor_signal(c, actor_uid, actor_name, etype, qvec: np.ndarray) -> None:
    """배우 한 명의 누적 칸을 갱신한다(있으면 더하고, 없으면 새로 만든다).
    pref_vec 에 '신호세기(부호 포함) × 검색벡터'를 더해 호평 방향을 쌓는다."""
    w = float(SIGNAL_WEIGHTS[etype])
    qvec = np.asarray(qvec, dtype=np.float32)
    row = c.execute(
        "SELECT pref_vec, sum_weight, abs_weight, pos_count, neg_count, event_count "
        "FROM actor_signals WHERE actor_uid=?", (actor_uid,),
    ).fetchone()
    if row and row[0]:
        pref = _blob_to_vec(row[0]).astype(np.float32).copy()
        sum_w, abs_w, pos_c, neg_c, ev_c = row[1], row[2], row[3], row[4], row[5]
        if pref.shape != qvec.shape:          # 차원이 다르면(모델 교체 등) 새로 시작
            pref = np.zeros_like(qvec)
            sum_w = abs_w = 0.0; pos_c = neg_c = ev_c = 0
    else:
        pref = np.zeros_like(qvec)
        sum_w = abs_w = 0.0; pos_c = neg_c = ev_c = 0
    pref = pref + w * qvec
    sum_w += w
    abs_w += abs(w)
    ev_c += 1
    if w > 0:
        pos_c += 1
    elif w < 0:
        neg_c += 1
    _upsert(
        c, "actor_signals", "actor_uid",
        ["actor_uid", "actor_name", "pref_vec", "sum_weight", "abs_weight",
         "pos_count", "neg_count", "event_count", "updated_at"],
        (actor_uid, actor_name, _vec_to_blob(pref), sum_w, abs_w, pos_c, neg_c, ev_c, _now()),
    )


# ==================================================================
#  미래의 '의미 기반 재랭킹' — 지금은 설계/조회만, 실제 순위 보정은 LEARNING_ENABLED=False라 꺼짐.
#  데이터가 충분히 쌓이면 LEARNING_ENABLED=True 로 켜고, 검색 단계에서 rerank_bonus()를 더하면 된다.
# ==================================================================

def get_actor_signal(actor_uid: str) -> dict | None:
    """배우 한 명의 누적 피드백을 돌려준다(없으면 None).
    pref_vec(호평 방향, 단위벡터), 신뢰도, 긍/부정 수 등."""
    with _conn() as c:
        row = c.execute(
            "SELECT actor_name, pref_vec, sum_weight, abs_weight, pos_count, neg_count, event_count "
            "FROM actor_signals WHERE actor_uid=?", (actor_uid,),
        ).fetchone()
    if not row:
        return None
    pref = _blob_to_vec(row[1])
    return {
        "actor_name": row[0],
        "pref_unit": _unit(pref) if pref is not None else None,  # 호평 '방향'(길이1)
        "sum_weight": row[2], "abs_weight": row[3],
        "pos_count": row[4], "neg_count": row[5], "event_count": row[6],
    }


def nearby_feedback(query_vec, top_searches: int = 50, min_sim: float = 0.55) -> dict:
    """이번 검색 벡터와 '의미가 가까운' 과거 검색들을 찾아, 거기서 나온 신호를
    배우별로 모아 돌려준다. (글자가 달라도 의미가 가까우면 피드백이 공유되는 핵심 조회)
    반환: { actor_uid: {"name":.., "score":누적보정점수, "events":건수} }
    ※ 조회만 한다 — 실제 순위 반영은 호출하는 쪽에서 LEARNING_ENABLED일 때만."""
    q = np.asarray(query_vec, dtype=np.float32)
    qn = _unit(q)
    out: dict[str, dict] = {}
    with _conn() as c:
        rows = c.execute(
            "SELECT id, query_embedding FROM searches WHERE query_embedding IS NOT NULL"
        ).fetchall()
        sims = []
        for sid, blob in rows:
            v = _blob_to_vec(blob)
            if v is None or v.shape != qn.shape:
                continue
            sim = float(_unit(v) @ qn)            # 검색끼리 코사인 유사도
            if sim >= min_sim:
                sims.append((sim, sid))
        sims.sort(reverse=True)
        for sim, sid in sims[:top_searches]:
            evs = c.execute(
                "SELECT actor_uid, actor_name, type FROM events "
                "WHERE search_id=? AND actor_uid IS NOT NULL", (sid,),
            ).fetchall()
            for uid, name, etype in evs:
                w = SIGNAL_WEIGHTS.get(etype)
                if w is None:
                    continue
                rec = out.setdefault(uid, {"name": name, "score": 0.0, "events": 0})
                rec["score"] += w * sim          # 의미가 가까울수록(sim 큰) 더 세게 반영
                rec["events"] += 1
    return out


def rerank_bonus(query_vec, actor_uids: list[str], strength: float = 0.05) -> dict:
    """검색 결과 순위에 '더해줄 보정값'을 배우별로 돌려준다.
    LEARNING_ENABLED=False 인 지금은 항상 0(=순위 안 바뀜). 켜면 비로소 작동.
    조합: (1) 비슷한 의미 과거검색의 신호 + (2) 그 배우의 누적 호평방향 일치도."""
    zeros = {uid: 0.0 for uid in actor_uids}
    if not LEARNING_ENABLED or query_vec is None:
        return zeros                              # ← 지금 단계: 항상 보정 없음
    qn = _unit(np.asarray(query_vec, dtype=np.float32))
    near = nearby_feedback(query_vec)
    bonus = {}
    for uid in actor_uids:
        b = 0.0
        if uid in near:
            b += near[uid]["score"]               # 비슷한 검색에서의 평
        sig = get_actor_signal(uid)
        if sig and sig["pref_unit"] is not None and sig["pref_unit"].shape == qn.shape:
            # 이 배우가 호평받던 '방향'과 이번 검색 방향이 얼마나 일치하나 × 신뢰도
            conf = min(1.0, sig["abs_weight"] / 5.0)
            b += float(sig["pref_unit"] @ qn) * conf
        bonus[uid] = b * strength
    return bonus


def stats() -> dict:
    """쌓인 양을 한눈에 보기 위한 요약(화면에 '눈으로 보여주기'용)."""
    with _conn() as c:
        n_searches = c.execute("SELECT COUNT(*) FROM searches").fetchone()[0]
        n_distinct_q = c.execute("SELECT COUNT(DISTINCT query_text) FROM searches").fetchone()[0]
        n_vec = c.execute(
            "SELECT COUNT(*) FROM searches WHERE query_embedding IS NOT NULL"
        ).fetchone()[0]
        n_events = c.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        by_type = dict(c.execute("SELECT type, COUNT(*) FROM events GROUP BY type").fetchall())
        n_actors = c.execute("SELECT COUNT(*) FROM actor_signals").fetchone()[0]
    return {"searches": n_searches, "distinct_queries": n_distinct_q,
            "searches_with_vector": n_vec, "events": n_events, "by_type": by_type,
            "actors_with_signal": n_actors, "learning_enabled": LEARNING_ENABLED}


def actor_activity(actor_uid: str) -> dict:
    """배우 한 명이 받은 신호를 종류별로 센다(배우 활동지표 표시용).
    예: {'shortlist': 3, 'contact': 1, 'click': 5, 'up': 2, 'down': 0}."""
    if not actor_uid:
        return {}
    with _conn() as c:
        rows = c.execute(
            "SELECT type, COUNT(*) FROM events WHERE actor_uid=? GROUP BY type",
            (actor_uid,),
        ).fetchall()
    return dict(rows)


def recent_events(limit: int = 20):
    """최근 신호 몇 건(검색어·대상·종류·순위)을 화면 확인용으로 가져온다."""
    with _conn() as c:
        return c.execute(
            """SELECT e.created_at, COALESCE(s.query_text,'(검색어 없음)'),
                      e.actor_name, e.type, e.rank, e.score
               FROM events e LEFT JOIN searches s ON e.search_id = s.id
               ORDER BY e.id DESC LIMIT ?""",
            (limit,),
        ).fetchall()


# ==================================================================
#  캐스팅 공고(게시판) — 감독이 올리고, 배우가 본다. 검색/랭킹과 완전히 분리.
# ==================================================================

def create_casting_call(director_uid, director_name, title, production="",
                        synopsis="", role_name="", gender="무관", age_min=None,
                        age_max=None, deadline="", description="",
                        required_fields=None, video_required=False,
                        roles=None) -> int | None:
    """공고 한 건을 저장하고 id를 돌려준다. 제목이 비면 저장하지 않음.
    이제 성별·나이 제한은 쓰지 않고(누구나 지원), synopsis(줄거리)와
    description(원하는 배역 이미지)을 중심으로 받는다.
    required_fields: 배우가 꼭 내야 하는 항목 리스트(예: ['전신사진','경력']).
    video_required: 동영상 링크를 필수로 받을지.
    roles: 모집 배역 목록(예: ['남자주인공 준호','여자주인공 서연']) — 지원 시 먼저 고른다."""
    t = (title or "").strip()
    if not t:
        return None
    try:
        req = _json.dumps(required_fields or [], ensure_ascii=False)
    except Exception:
        req = "[]"
    try:
        roles_clean = []
        for r in (roles or []):
            if isinstance(r, dict):
                nm = str(r.get("name") or "").strip()
                if nm:
                    roles_clean.append({
                        "name": nm,
                        "desc": str(r.get("desc") or "").strip(),
                        "image": str(r.get("image") or "").strip(),
                        "closed": bool(r.get("closed")),
                    })
            elif str(r or "").strip():
                roles_clean.append(str(r).strip())
        roles_json = _json.dumps(roles_clean, ensure_ascii=False)
    except Exception:
        roles_json = "[]"
    with _conn() as c:
        token = _gen_call_token(c)
        return _insert(
            c,
            "INSERT INTO casting_calls(director_uid, director_name, title, production, "
            "synopsis, role_name, gender, age_min, age_max, deadline, description, "
            "required_fields, video_required, roles, token, created_at, active) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)",
            (director_uid, director_name, t, production, synopsis, role_name, gender,
             age_min, age_max, deadline, description, req,
             1 if video_required else 0, roles_json, token, _now()),
        )


def _gen_call_token(c) -> str:
    """겹치지 않는 공개 링크 토큰(10자리 hex)을 만든다."""
    for _ in range(12):
        tok = secrets.token_hex(5)   # 예: 'a3f9c2e1b4'
        row = c.execute("SELECT 1 FROM casting_calls WHERE token=?", (tok,)).fetchone()
        if not row:
            return tok
    return secrets.token_hex(8)      # 극히 드문 충돌 시 더 긴 토큰


def set_call_role_closed(call_id, role_name: str, closed: bool = True) -> bool:
    """공고 안의 '특정 배역' 하나만 마감(또는 재모집)한다. 바뀌면 True."""
    try:
        cid = int(call_id)
    except (TypeError, ValueError):
        return False
    with _conn() as c:
        row = c.execute("SELECT roles FROM casting_calls WHERE id=?", (cid,)).fetchone()
        if not row:
            return False
        try:
            roles = _json.loads(row[0]) if row[0] else []
        except Exception:
            roles = []
        norm, changed = [], False
        for r in roles:
            if isinstance(r, dict):
                d = {"name": str(r.get("name") or "").strip(),
                     "desc": str(r.get("desc") or "").strip(),
                     "image": str(r.get("image") or "").strip(),
                     "closed": bool(r.get("closed"))}
            else:
                d = {"name": str(r or "").strip(), "desc": "", "image": "", "closed": False}
            if d["name"] and d["name"] == role_name:
                d["closed"] = bool(closed)
                changed = True
            norm.append(d)
        if changed:
            c.execute("UPDATE casting_calls SET roles=? WHERE id=?",
                      (_json.dumps(norm, ensure_ascii=False), cid))
        return changed


def list_casting_calls(active_only: bool = True, director_uid: str | None = None) -> list[dict]:
    """공고 목록(최신순)을 돌려준다. active_only면 모집중만, director_uid면 그 감독 것만."""
    q = ("SELECT id, director_uid, director_name, title, production, synopsis, role_name, "
         "gender, age_min, age_max, deadline, description, required_fields, video_required, "
         "roles, token, created_at, active FROM casting_calls")
    conds, args = [], []
    if active_only:
        conds.append("active=1")
    if director_uid:
        conds.append("director_uid=?"); args.append(director_uid)
    if conds:
        q += " WHERE " + " AND ".join(conds)
    q += " ORDER BY id DESC"
    cols = ["id", "director_uid", "director_name", "title", "production", "synopsis",
            "role_name", "gender", "age_min", "age_max", "deadline", "description",
            "required_fields", "video_required", "roles", "token", "created_at", "active"]
    out = []
    with _conn() as c:
        for r in c.execute(q, args).fetchall():
            d = dict(zip(cols, r))
            try:
                d["required_fields"] = _json.loads(d["required_fields"]) if d.get("required_fields") else []
            except Exception:
                d["required_fields"] = []
            try:
                d["roles"] = _json.loads(d["roles"]) if d.get("roles") else []
            except Exception:
                d["roles"] = []
            d["video_required"] = bool(d.get("video_required"))
            out.append(d)
    return out


def get_casting_call(key) -> dict | None:
    """공고 한 건을 가져온다(활성/마감 무관). 링크로 들어온 지원자용.
    key는 공개 토큰(예 'a3f9c2e1b4') 또는 옛 숫자 id 둘 다 받는다."""
    if key is None:
        return None
    skey = str(key)
    calls = list_casting_calls(active_only=False)
    # 1) 토큰 우선 매칭
    for d in calls:
        if d.get("token") and d["token"] == skey:
            return d
    # 2) 옛 링크 호환: 숫자 id 매칭
    if skey.isdigit():
        cid = int(skey)
        for d in calls:
            if d["id"] == cid:
                return d
    return None


def set_casting_call_active(call_id: int, active: bool) -> None:
    """공고를 모집중(1)/마감(0)으로 바꾼다."""
    with _conn() as c:
        c.execute("UPDATE casting_calls SET active=? WHERE id=?",
                  (1 if active else 0, call_id))


def delete_casting_call(call_id: int, director_uid: str | None = None) -> None:
    """공고를 삭제한다. director_uid를 주면 본인 공고만 지운다(안전)."""
    with _conn() as c:
        if director_uid:
            c.execute("DELETE FROM casting_calls WHERE id=? AND director_uid=?",
                      (call_id, director_uid))
        else:
            c.execute("DELETE FROM casting_calls WHERE id=?", (call_id,))


# ==================================================================
#  회원 프로필 저장/조회/삭제 — 관리자 페이지용(전체 회원 관리)
# ==================================================================
import json as _json


def _json_default(o):
    """JSON으로 못 바꾸는 값(numpy 배열·정수·실수)을 안전하게 변환.
    배우 데이터엔 사진 임베딩(numpy)이 들어 있어, 이게 없으면 저장이 통째로 실패한다."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, (bytes, bytearray)):
        return None
    raise TypeError(f"not JSON serializable: {type(o)}")


def save_profile_db(uid: str, prof: dict) -> None:
    """회원 프로필을 DB에 저장(있으면 갱신). 세션 저장과 별개로 여러 세션에서 공유·관리하기 위함."""
    if not uid or not isinstance(prof, dict):
        return
    role = prof.get("role")
    name = prof.get("name")
    email = prof.get("email")
    try:
        data = _json.dumps(prof, ensure_ascii=False)
    except Exception:
        data = "{}"
    with _conn() as c:
        row = c.execute("SELECT created_at FROM profiles WHERE uid=?", (uid,)).fetchone()
        created = row[0] if row else _now()
        _upsert(
            c, "profiles", "uid",
            ["uid", "role", "name", "email", "data", "created_at", "updated_at"],
            (uid, role, name, email, data, created, _now()),
        )


def list_profiles_db(role: str | None = None) -> list[dict]:
    """회원 프로필 목록(최신 등록순). role을 주면 그 역할만(director/actor)."""
    q = "SELECT uid, role, name, email, data, created_at FROM profiles"
    args: list = []
    if role:
        q += " WHERE role=?"; args.append(role)
    q += " ORDER BY created_at DESC, uid DESC"
    out = []
    with _conn() as c:
        for uid, r, name, email, data, created in c.execute(q, args).fetchall():
            try:
                d = _json.loads(data) if data else {}
            except Exception:
                d = {}
            out.append({"uid": uid, "role": r, "name": name, "email": email,
                        "data": d, "created_at": created})
    return out


def delete_profile_db(uid: str) -> None:
    """회원 프로필 한 건을 DB에서 삭제(관리자용)."""
    if not uid:
        return
    with _conn() as c:
        c.execute("DELETE FROM profiles WHERE uid=?", (uid,))


def delete_account(email_or_uid: str) -> bool:
    """계정 탈퇴/삭제 — 그 사람과 연결된 데이터를 전부 정리한다.
    users(로그인 계정), profiles, 배우 지원자(applicants), 지원서, 알림, 메시지,
    그리고 감독이면 그가 올린 공고와 그 공고의 지원/오디션까지 삭제.
    (이 함수를 써야 '이미 가입된 계정' 잔여 없이 깨끗이 지워진다.)"""
    em = (email_or_uid or "").strip().lower()
    if not em:
        return False
    with _conn() as c:
        # 프로필에서 배우 식별자(actor_uid) 찾기
        row = c.execute("SELECT data FROM profiles WHERE uid=?", (em,)).fetchone()
        actor_uid = None
        if row and row[0]:
            try:
                actor_uid = (_json.loads(row[0]) or {}).get("actor_uid")
            except Exception:
                actor_uid = None
        # 1) 계정·프로필 본체
        c.execute("DELETE FROM users WHERE email=?", (em,))
        c.execute("DELETE FROM sessions WHERE email=?", (em,))
        c.execute("DELETE FROM profiles WHERE uid=?", (em,))
        # 2) 이 사람이 '지원자(배우)'로서 남긴 것
        c.execute("DELETE FROM applications WHERE actor_login=?", (em,))
        c.execute("DELETE FROM notifications WHERE user_uid=?", (em,))
        c.execute("DELETE FROM messages WHERE director_uid=?", (em,))
        if actor_uid:
            c.execute("DELETE FROM applicants WHERE uid=?", (actor_uid,))
            c.execute("DELETE FROM messages WHERE actor_uid=?", (actor_uid,))
            c.execute("DELETE FROM notifications WHERE user_uid=?", (actor_uid,))
            c.execute("DELETE FROM audition_picks WHERE actor_uid=?", (actor_uid,))
        # 3) 이 사람이 '감독'으로서 올린 공고와 거기 딸린 것
        cids = [r[0] for r in c.execute(
            "SELECT id FROM casting_calls WHERE director_uid=?", (em,)).fetchall()]
        for cid in cids:
            c.execute("DELETE FROM applications WHERE call_id=?", (cid,))
            c.execute("DELETE FROM audition_slots WHERE call_id=?", (cid,))
            c.execute("DELETE FROM audition_picks WHERE call_id=?", (cid,))
        c.execute("DELETE FROM casting_calls WHERE director_uid=?", (em,))
    return True


# ==================================================================
#  배우 지원자(검색 대상) 저장/조회/삭제 — 세션 간 공유·영속화
# ==================================================================

def save_applicant_db(actor: dict, emb=None) -> None:
    """지원자 한 명(actor dict + 임베딩)을 DB에 저장(있으면 갱신)."""
    uid = (actor or {}).get("uid")
    if not uid:
        return
    try:
        data = _json.dumps(actor, ensure_ascii=False, default=_json_default)
    except Exception:
        return
    blob = _vec_to_blob(emb) if emb is not None else None
    with _conn() as c:
        row = c.execute("SELECT created_at FROM applicants WHERE uid=?", (uid,)).fetchone()
        created = row[0] if row else _now()
        _upsert(
            c, "applicants", "uid",
            ["uid", "name", "data", "emb", "created_at"],
            (uid, actor.get("name"), data, blob, created),
        )


def update_applicant_data(uid: str, actor: dict) -> bool:
    """지원자의 표시·연락 정보(data JSON, name)만 갱신. 임베딩(emb)은 그대로 둔다.
    (이름·키·링크 등 텍스트 수정은 인상 분석/검색 벡터에 영향이 없으므로 재분석 불필요.)"""
    if not uid or not isinstance(actor, dict):
        return False
    try:
        data = _json.dumps(actor, ensure_ascii=False, default=_json_default)
    except Exception:
        return False
    with _conn() as c:
        cur = c.execute("UPDATE applicants SET name=?, data=? WHERE uid=?",
                        (actor.get("name"), data, uid))
        try:
            return cur.rowcount != 0
        except Exception:
            return True


def list_applicants_db() -> tuple[list[dict], list]:
    """저장된 지원자 전체를 (actor dict 목록, 임베딩 목록)으로 돌려준다(등록 오래된 순).
    검색 엔진이 쓰던 (applicants, app_embs)와 같은 형태라 그대로 끼워 넣을 수 있다."""
    actors: list[dict] = []
    embs: list = []
    with _conn() as c:
        rows = c.execute(
            "SELECT data, emb FROM applicants ORDER BY created_at ASC, uid ASC"
        ).fetchall()
    for data, blob in rows:
        try:
            a = _json.loads(data) if data else None
        except Exception:
            a = None
        if not a:
            continue
        # 배우 '본인'이 직접 등록한 사람만 공유 풀(배우 탐색)에 노출한다.
        # 감독이 '지원서 업로드'로 올린 사람은 그 감독에게만 보이므로 여기서 제외.
        if not a.get("self_registered"):
            continue
        v = _blob_to_vec(blob)
        actors.append(a)
        embs.append(v if v is not None else [])
    return actors, embs


def delete_applicant_db(uid: str) -> None:
    """지원자 한 명을 DB에서 삭제."""
    if not uid:
        return
    with _conn() as c:
        c.execute("DELETE FROM applicants WHERE uid=?", (uid,))


def clear_applicants_db() -> None:
    """지원자 전체 삭제(전체 비우기용)."""
    with _conn() as c:
        c.execute("DELETE FROM applicants")


# ---------- 메신저 (감독 ↔ 배우) ----------
def send_message(director_uid: str, actor_uid: str, sender: str, text: str,
                 director_name: str = "", actor_name: str = "") -> None:
    """메시지 한 통을 저장한다. sender는 'director' 또는 'actor'."""
    text = (text or "").strip()
    if not (director_uid and actor_uid and text):
        return
    with _conn() as c:
        c.execute(
            "INSERT INTO messages(director_uid, actor_uid, director_name, actor_name, "
            "sender, text, created_at) VALUES(?,?,?,?,?,?,?)",
            (director_uid, actor_uid, director_name, actor_name, sender, text, _now()),
        )


def list_messages(director_uid: str, actor_uid: str) -> list[dict]:
    """한 대화(감독 1명 ↔ 배우 1명)의 메시지를 시간순(오래된 → 최신)으로 돌려준다."""
    with _conn() as c:
        rows = c.execute(
            "SELECT sender, text, created_at, director_name, actor_name FROM messages "
            "WHERE director_uid=? AND actor_uid=? ORDER BY id ASC",
            (director_uid, actor_uid),
        ).fetchall()
    return [{"sender": r[0], "text": r[1], "created_at": r[2],
             "director_name": r[3], "actor_name": r[4]} for r in rows]


def list_conversations_for_director(director_uid: str) -> list[dict]:
    """그 감독이 가진 대화 목록(상대 배우별 1줄, 최근 대화 먼저)."""
    if not director_uid:
        return []
    with _conn() as c:
        rows = c.execute(
            "SELECT actor_uid, "
            "       MAX(actor_name) AS actor_name, "
            "       (SELECT text FROM messages m2 WHERE m2.director_uid=m1.director_uid "
            "         AND m2.actor_uid=m1.actor_uid ORDER BY id DESC LIMIT 1) AS last_text, "
            "       MAX(created_at) AS last_at "
            "FROM messages m1 WHERE director_uid=? "
            "GROUP BY actor_uid, director_uid ORDER BY last_at DESC",
            (director_uid,),
        ).fetchall()
    return [{"actor_uid": r[0], "actor_name": r[1], "last_text": r[2], "last_at": r[3]}
            for r in rows]


def list_conversations_for_actor(actor_uid: str) -> list[dict]:
    """그 배우가 가진 대화 목록(상대 감독별 1줄, 최근 대화 먼저)."""
    if not actor_uid:
        return []
    with _conn() as c:
        rows = c.execute(
            "SELECT director_uid, "
            "       MAX(director_name) AS director_name, "
            "       (SELECT text FROM messages m2 WHERE m2.director_uid=m1.director_uid "
            "         AND m2.actor_uid=m1.actor_uid ORDER BY id DESC LIMIT 1) AS last_text, "
            "       MAX(created_at) AS last_at "
            "FROM messages m1 WHERE actor_uid=? "
            "GROUP BY director_uid, actor_uid ORDER BY last_at DESC",
            (actor_uid,),
        ).fetchall()
    return [{"director_uid": r[0], "director_name": r[1], "last_text": r[2], "last_at": r[3]}
            for r in rows]


# ==================================================================
#  공고 지원(방) — 링크로 들어와 지원, 감독이 합격/불합격 결정
# ==================================================================

def create_application(call_id, applicant_name, data=None, video_link="",
                       actor_uid="", actor_login="") -> int | None:
    """공고에 지원서 한 건을 저장하고 id를 돌려준다. 이름이 비면 저장 안 함."""
    name = (applicant_name or "").strip()
    if call_id is None or not name:
        return None
    try:
        d = _json.dumps(data or {}, ensure_ascii=False)
    except Exception:
        d = "{}"
    with _conn() as c:
        return _insert(
            c,
            "INSERT INTO applications(call_id, actor_uid, actor_login, applicant_name, "
            "data, video_link, status, created_at) VALUES(?,?,?,?,?,?, 'pending', ?)",
            (int(call_id), actor_uid or "", actor_login or "", name, d,
             video_link or "", _now()),
        )


def _app_row_to_dict(r) -> dict:
    cols = ["id", "call_id", "actor_uid", "actor_login", "applicant_name",
            "data", "video_link", "status", "created_at"]
    d = dict(zip(cols, r))
    try:
        d["data"] = _json.loads(d["data"]) if d.get("data") else {}
    except Exception:
        d["data"] = {}
    return d


_APP_COLS = ("id, call_id, actor_uid, actor_login, applicant_name, "
             "data, video_link, status, created_at")


def list_applications(call_id) -> list[dict]:
    """한 공고에 들어온 지원서 목록(최신순)."""
    if call_id is None:
        return []
    with _conn() as c:
        rows = c.execute(
            f"SELECT {_APP_COLS} FROM applications WHERE call_id=? ORDER BY id DESC",
            (int(call_id),),
        ).fetchall()
    return [_app_row_to_dict(r) for r in rows]


def list_applications_for_actor(actor_login) -> list[dict]:
    """로그인한 배우가 낸 지원서 목록(최신순) — 내 지원 현황 보기용."""
    if not actor_login:
        return []
    with _conn() as c:
        rows = c.execute(
            f"SELECT {_APP_COLS} FROM applications WHERE actor_login=? ORDER BY id DESC",
            (actor_login,),
        ).fetchall()
    return [_app_row_to_dict(r) for r in rows]


def get_application(app_id) -> dict | None:
    if app_id is None:
        return None
    with _conn() as c:
        row = c.execute(
            f"SELECT {_APP_COLS} FROM applications WHERE id=?", (int(app_id),)
        ).fetchone()
    return _app_row_to_dict(row) if row else None


def set_application_status(app_id, status: str) -> None:
    """지원서 상태를 pending/accepted/rejected 로 바꾼다."""
    if app_id is None or status not in ("pending", "accepted", "rejected"):
        return
    with _conn() as c:
        c.execute("UPDATE applications SET status=? WHERE id=?", (status, int(app_id)))


# ==================================================================
#  앱 알림함 — 합격/불합격 결과, 일정 안내 등
# ==================================================================

def add_notification(user_uid: str, kind: str, title: str, body: str = "",
                     ref=None) -> None:
    """사용자에게 앱 알림 한 건을 보낸다. ref에 공고 id를 넣으면 알림에서 그 공고를 다시 볼 수 있다."""
    if not user_uid:
        return
    with _conn() as c:
        c.execute(
            "INSERT INTO notifications(user_uid, kind, title, body, ref, read, created_at) "
            "VALUES(?,?,?,?,?,0,?)",
            (user_uid, kind or "info", title or "", body or "",
             (str(ref) if ref is not None else None), _now()),
        )


def list_notifications(user_uid: str) -> list[dict]:
    """그 사용자의 알림 목록(최신순)."""
    if not user_uid:
        return []
    with _conn() as c:
        rows = c.execute(
            "SELECT id, kind, title, body, ref, read, created_at FROM notifications "
            "WHERE user_uid=? ORDER BY id DESC",
            (user_uid,),
        ).fetchall()
    return [{"id": r[0], "kind": r[1], "title": r[2], "body": r[3],
             "ref": r[4], "read": bool(r[5]), "created_at": r[6]} for r in rows]


def count_unread(user_uid: str) -> int:
    """안 읽은 알림 개수."""
    if not user_uid:
        return 0
    with _conn() as c:
        row = c.execute(
            "SELECT COUNT(*) FROM notifications WHERE user_uid=? AND read=0",
            (user_uid,),
        ).fetchone()
    return int(row[0]) if row else 0


def mark_notifications_read(user_uid: str) -> None:
    """그 사용자의 알림을 모두 읽음 처리."""
    if not user_uid:
        return
    with _conn() as c:
        c.execute("UPDATE notifications SET read=1 WHERE user_uid=?", (user_uid,))


# ==================================================================
#  오디션 일정 — 감독이 후보 시간 제시 → 배우가 선택
# ==================================================================

def add_audition_slot(call_id, director_uid, label, place="", capacity=0) -> int | None:
    """후보 오디션 시간(슬롯) 하나를 추가."""
    lab = (label or "").strip()
    if call_id is None or not lab:
        return None
    with _conn() as c:
        return _insert(
            c,
            "INSERT INTO audition_slots(call_id, director_uid, label, place, capacity, created_at) "
            "VALUES(?,?,?,?,?,?)",
            (int(call_id), director_uid or "", lab, place or "", int(capacity or 0), _now()),
        )


def list_audition_slots(call_id) -> list[dict]:
    """한 공고의 후보 시간 목록(추가 순). 각 슬롯에 현재 선택 인원(picked) 포함."""
    if call_id is None:
        return []
    with _conn() as c:
        rows = c.execute(
            "SELECT id, call_id, director_uid, label, place, capacity, created_at "
            "FROM audition_slots WHERE call_id=? ORDER BY id ASC",
            (int(call_id),),
        ).fetchall()
        out = []
        for r in rows:
            picked = c.execute(
                "SELECT COUNT(*) FROM audition_picks WHERE slot_id=?", (r[0],)
            ).fetchone()[0]
            out.append({"id": r[0], "call_id": r[1], "director_uid": r[2],
                        "label": r[3], "place": r[4], "capacity": r[5],
                        "created_at": r[6], "picked": int(picked)})
    return out


def delete_audition_slot(slot_id) -> None:
    """후보 시간 하나와 그에 달린 배우 선택을 함께 삭제."""
    if slot_id is None:
        return
    with _conn() as c:
        c.execute("DELETE FROM audition_picks WHERE slot_id=?", (int(slot_id),))
        c.execute("DELETE FROM audition_slots WHERE id=?", (int(slot_id),))


def pick_audition_slot(slot_id, call_id, actor_uid, actor_name="") -> None:
    """배우가 후보 시간 하나를 고른다(한 공고당 한 번만 — 기존 선택은 교체)."""
    if slot_id is None or call_id is None or not actor_uid:
        return
    with _conn() as c:
        # 같은 공고에서 이 배우의 기존 선택을 지우고 새로 넣는다(한 명 = 한 슬롯).
        c.execute("DELETE FROM audition_picks WHERE call_id=? AND actor_uid=?",
                  (int(call_id), actor_uid))
        c.execute(
            "INSERT INTO audition_picks(slot_id, call_id, actor_uid, actor_name, created_at) "
            "VALUES(?,?,?,?,?)",
            (int(slot_id), int(call_id), actor_uid, actor_name or "", _now()),
        )


def get_actor_pick(call_id, actor_uid) -> int | None:
    """그 배우가 이 공고에서 고른 슬롯 id(없으면 None)."""
    if call_id is None or not actor_uid:
        return None
    with _conn() as c:
        row = c.execute(
            "SELECT slot_id FROM audition_picks WHERE call_id=? AND actor_uid=? "
            "ORDER BY id DESC LIMIT 1",
            (int(call_id), actor_uid),
        ).fetchone()
    return int(row[0]) if row else None


def list_slot_picks(slot_id) -> list[dict]:
    """한 슬롯을 고른 배우 목록(감독이 누가 그 시간에 오는지 확인)."""
    if slot_id is None:
        return []
    with _conn() as c:
        rows = c.execute(
            "SELECT actor_uid, actor_name, created_at FROM audition_picks "
            "WHERE slot_id=? ORDER BY id ASC",
            (int(slot_id),),
        ).fetchall()
    return [{"actor_uid": r[0], "actor_name": r[1], "created_at": r[2]} for r in rows]
