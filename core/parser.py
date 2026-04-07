"""
HANA Trace Parser
SAP HANA .trc / .txt / .log 파일을 파싱하여 구조화된 TraceEntry 목록으로 변환합니다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# TraceEntry: 개별 로그 항목을 표현하는 데이터 클래스
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TraceEntry:
    timestamp: Optional[datetime] = None
    timestamp_raw: str = ""
    level: str = "INFO"           # FATAL / ERROR / WARNING / INFO / DEBUG
    component: str = ""           # 컴포넌트 / 서비스 이름
    thread_id: str = ""           # 스레드 ID
    message: str = ""             # 본문 메시지
    raw_line: str = ""            # 원본 라인
    line_number: int = 0          # 파일 내 라인 번호
    extras: Dict[str, Any] = field(default_factory=dict)  # 추가 메타 정보

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat() if self.timestamp else self.timestamp_raw,
            "level": self.level,
            "component": self.component,
            "thread_id": self.thread_id,
            "message": self.message,
            "line_number": self.line_number,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 정규식 패턴 모음
# ─────────────────────────────────────────────────────────────────────────────

# 패턴 1: SAP HANA 표준 trace 포맷
# [2024-01-15 10:23:45.123456] [THREAD:12345] [ERROR] component_name: message
_PATTERN_HANA_FULL = re.compile(
    r"^\[(?P<ts>[^\]]+)\]"                           # [timestamp]
    r"\s*\[THREAD:(?P<thread>\d+)\]"                 # [THREAD:id]
    r"\s*\[(?P<level>FATAL|ERROR|WARNING|INFO|DEBUG)\]"  # [LEVEL]
    r"\s*(?P<component>[^:]+):\s*(?P<message>.*)$",
    re.IGNORECASE,
)

# 패턴 2: SAP HANA 간소화 trace 포맷
# 2024-01-15 10:23:45.123456  e  indexserver  message
_PATTERN_HANA_COMPACT = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)"
    r"\s+(?P<level_char>[fedwi])\s+"
    r"(?P<component>\S+)\s+"
    r"(?P<message>.+)$",
    re.IGNORECASE,
)

# 패턴 3: nameserver/indexserver 표준 trace 포맷
# [1234]{5678}[6/6] 2024-01-15 10:23:45.123456 e  srv  message
_PATTERN_HANA_NS = re.compile(
    r"^\[(?P<pid>\d+)\]\{(?P<thread>[\d-]+)\}\[[\d/]+\]\s+"
    r"(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)"
    r"\s+(?P<level_char>[fedwi])\s+"
    r"(?P<component>\S+)\s+"
    r"(?P<message>.+)$",
    re.IGNORECASE,
)

# 패턴 4: 표준 시스템 로그 포맷
# 2024-01-15T10:23:45.123456 ERROR [component] message
_PATTERN_ISO_LEVEL = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)"
    r"(?:Z|[+-]\d{2}:?\d{2})?\s+"
    r"(?P<level>FATAL|ERROR|WARNING|WARN|INFO|DEBUG)\s+"
    r"(?:\[(?P<component>[^\]]+)\]\s*)?"
    r"(?P<message>.+)$",
    re.IGNORECASE,
)

# 패턴 5: 심플 타임스탬프 + 메시지 (레벨 키워드 포함)
# 2024-01-15 10:23:45 message with ERROR keyword
_PATTERN_SIMPLE_TS = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)"
    r"(?:Z|[+-]\d{2}:?\d{2})?\s+"
    r"(?P<message>.+)$",
    re.IGNORECASE,
)

# 패턴 6: 레벨만 있는 라인 (타임스탬프 없음)
# ERROR: message / FATAL - message
_PATTERN_LEVEL_ONLY = re.compile(
    r"^(?P<level>FATAL|ERROR|WARNING|WARN|INFO|DEBUG)\s*[:\-]\s*(?P<message>.+)$",
    re.IGNORECASE,
)


# 레벨 문자 -> 레벨 이름 매핑 (HANA compact format)
_LEVEL_CHAR_MAP = {
    "f": "FATAL",
    "e": "ERROR",
    "d": "WARNING",   # HANA 에서 'w' 또는 'd' 를 경고에 사용
    "w": "WARNING",
    "i": "INFO",
    "b": "DEBUG",
}

# 메시지 내 레벨 키워드 감지
_MSG_LEVEL_RE = re.compile(
    r"\b(FATAL|CRASH|CRITICAL|PANIC|EXCEPTION|ABORT|ERROR|FAIL(?:URE|ED)?|"
    r"WARN(?:ING)?|INFO|DEBUG)\b",
    re.IGNORECASE,
)

_TIMESTAMP_FORMATS = [
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y/%m/%d %H:%M:%S.%f",
    "%Y/%m/%d %H:%M:%S",
]


def _parse_timestamp(ts_str: str) -> Optional[datetime]:
    """다양한 형식의 타임스탬프 문자열을 datetime 으로 변환."""
    ts_str = ts_str.strip().replace(",", ".")
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    return None


def _normalize_level(raw: str) -> str:
    """레벨 문자열을 표준화된 레벨명으로 변환."""
    raw = raw.strip().upper()
    mapping = {
        "FATAL": "FATAL",
        "CRASH": "FATAL",
        "CRITICAL": "FATAL",
        "PANIC": "FATAL",
        "ABORT": "FATAL",
        "ERROR": "ERROR",
        "ERR": "ERROR",
        "EXCEPTION": "ERROR",
        "WARNING": "WARNING",
        "WARN": "WARNING",
        "INFO": "INFO",
        "DEBUG": "DEBUG",
        "DBG": "DEBUG",
        "TRACE": "DEBUG",
    }
    return mapping.get(raw, "INFO")


def _infer_level_from_message(message: str) -> str:
    """메시지 내용에서 레벨을 추론."""
    m = _MSG_LEVEL_RE.search(message)
    if m:
        return _normalize_level(m.group(1))
    return "INFO"


# ─────────────────────────────────────────────────────────────────────────────
# HANATraceParser
# ─────────────────────────────────────────────────────────────────────────────
class HANATraceParser:
    """SAP HANA trace 파일을 파싱하여 TraceEntry 목록으로 반환합니다."""

    def __init__(self, merge_multiline: bool = True):
        """
        Parameters
        ----------
        merge_multiline : bool
            True 이면 스택 트레이스 등 연속된 들여쓰기 라인을 직전 엔트리에 합칩니다.
        """
        self.merge_multiline = merge_multiline

    # ── 공개 API ────────────────────────────────────────────────────────────

    def parse_lines(self, lines: List[str]) -> List[TraceEntry]:
        """라인 목록을 파싱하여 TraceEntry 리스트를 반환합니다."""
        entries: List[TraceEntry] = []
        for idx, raw_line in enumerate(lines, start=1):
            line = raw_line.rstrip("\n\r")
            if not line.strip():
                continue

            entry = self._parse_line(line, idx)
            if entry is None:
                # 파싱 실패 라인: multiline merge 또는 Unknown 항목으로 처리
                if self.merge_multiline and entries and line.startswith((" ", "\t", "#", ">")):
                    entries[-1].message += "\n" + line.strip()
                    continue
                # 최소한 raw 내용이라도 저장
                entry = TraceEntry(
                    level=_infer_level_from_message(line),
                    message=line.strip(),
                    raw_line=raw_line,
                    line_number=idx,
                )
            entries.append(entry)

        return entries

    def parse_file(self, filepath: str, encoding: str = "utf-8") -> List[TraceEntry]:
        """파일 경로에서 직접 파싱합니다."""
        encodings = [encoding, "utf-8", "euc-kr", "cp1252", "latin-1"]
        seen = set()
        for enc in encodings:
            if enc in seen:
                continue
            seen.add(enc)
            try:
                with open(filepath, "r", encoding=enc) as f:
                    return self.parse_lines(f.readlines())
            except UnicodeDecodeError:
                continue
        raise ValueError(f"지원되지 않는 파일 인코딩: {filepath}")

    def to_dataframe(self, entries: List[TraceEntry]) -> pd.DataFrame:
        """TraceEntry 목록을 pandas DataFrame 으로 변환합니다."""
        if not entries:
            return pd.DataFrame()
        rows = [e.to_dict() for e in entries]
        df = pd.DataFrame(rows)
        # timestamp 열 타입 변환
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df

    # ── 내부 파싱 로직 ───────────────────────────────────────────────────────

    def _parse_line(self, line: str, line_number: int) -> Optional[TraceEntry]:
        """단일 라인을 파싱하여 TraceEntry 또는 None 을 반환합니다."""

        # 1) HANA Full format: [ts][THREAD:id][LEVEL] comp: msg
        m = _PATTERN_HANA_FULL.match(line)
        if m:
            ts = _parse_timestamp(m.group("ts"))
            return TraceEntry(
                timestamp=ts,
                timestamp_raw=m.group("ts"),
                level=_normalize_level(m.group("level")),
                component=m.group("component").strip(),
                thread_id=m.group("thread"),
                message=m.group("message").strip(),
                raw_line=line,
                line_number=line_number,
            )

        # 2) HANA NS format: [pid]{thread}[x/x] ts level comp msg
        m = _PATTERN_HANA_NS.match(line)
        if m:
            ts = _parse_timestamp(m.group("ts"))
            level_char = m.group("level_char").lower()
            return TraceEntry(
                timestamp=ts,
                timestamp_raw=m.group("ts"),
                level=_LEVEL_CHAR_MAP.get(level_char, "INFO"),
                component=m.group("component").strip(),
                thread_id=m.group("thread"),
                message=m.group("message").strip(),
                raw_line=line,
                line_number=line_number,
                extras={"pid": m.group("pid")},
            )

        # 3) HANA Compact format: ts level_char comp msg
        m = _PATTERN_HANA_COMPACT.match(line)
        if m:
            ts = _parse_timestamp(m.group("ts"))
            level_char = m.group("level_char").lower()
            return TraceEntry(
                timestamp=ts,
                timestamp_raw=m.group("ts"),
                level=_LEVEL_CHAR_MAP.get(level_char, "INFO"),
                component=m.group("component").strip(),
                message=m.group("message").strip(),
                raw_line=line,
                line_number=line_number,
            )

        # 4) ISO level format: ts LEVEL [comp] msg
        m = _PATTERN_ISO_LEVEL.match(line)
        if m:
            ts = _parse_timestamp(m.group("ts"))
            return TraceEntry(
                timestamp=ts,
                timestamp_raw=m.group("ts"),
                level=_normalize_level(m.group("level")),
                component=(m.group("component") or "").strip(),
                message=m.group("message").strip(),
                raw_line=line,
                line_number=line_number,
            )

        # 5) Simple timestamp + message (레벨은 메시지에서 추론)
        m = _PATTERN_SIMPLE_TS.match(line)
        if m:
            ts = _parse_timestamp(m.group("ts"))
            msg = m.group("message").strip()
            return TraceEntry(
                timestamp=ts,
                timestamp_raw=m.group("ts"),
                level=_infer_level_from_message(msg),
                message=msg,
                raw_line=line,
                line_number=line_number,
            )

        # 6) Level-only format: LEVEL: msg
        m = _PATTERN_LEVEL_ONLY.match(line)
        if m:
            return TraceEntry(
                level=_normalize_level(m.group("level")),
                message=m.group("message").strip(),
                raw_line=line,
                line_number=line_number,
            )

        return None
