"""
HANA Trace AI Analyzer
OpenAI API 를 이용해 TraceEntry 목록을 분석하고 구조화된 결과를 반환합니다.
"""

from __future__ import annotations

import json
import os
import textwrap
from typing import Any, Dict, List, Optional

from .parser import TraceEntry

# ─────────────────────────────────────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_MODEL = "gpt-4o"
_MAX_ENTRIES_PER_CALL = 200       # API 호출 당 전달할 최대 엔트리 수
_PRIORITY_LEVELS = {"FATAL", "ERROR", "WARNING"}  # 우선 포함 레벨

# 시스템 프롬프트
_SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert SAP HANA database administrator and performance analyst.
    Your task is to analyze SAP HANA trace log entries and provide:
    1. A concise summary of the main findings (in Korean).
    2. A prioritized list of recommended actions (in Korean).
    3. A list of critical issues that need immediate attention (in Korean).

    Respond ONLY with a valid JSON object in the following format:
    {
      "summary": "<Korean text: 2–5 sentences summarizing the log findings>",
      "actions": ["<action 1>", "<action 2>", ...],
      "critical_issues": ["<issue 1>", "<issue 2>", ...]
    }

    Rules:
    - Focus on anomalies, errors, performance bottlenecks, and security concerns.
    - critical_issues should only include FATAL or severe ERROR entries.
    - actions must be specific and actionable.
    - Keep summary under 200 words.
    - If there are no critical issues, return an empty list for critical_issues.
    - Return ONLY the JSON object, no extra text.
""").strip()


# ─────────────────────────────────────────────────────────────────────────────
# HANAAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class HANAAnalyzer:
    """
    SAP HANA trace 분석기.
    
    사용 예:
        analyzer = HANAAnalyzer(api_key="sk-...", model="gpt-4o")
        result = analyzer.analyze(entries)
        print(result["summary"])
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = _DEFAULT_MODEL,
        max_entries: int = _MAX_ENTRIES_PER_CALL,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self.max_entries = max_entries
        self.client = self._build_client()

    # ── 공개 API ────────────────────────────────────────────────────────────

    def analyze(self, entries: List[TraceEntry]) -> Dict[str, Any]:
        """
        TraceEntry 목록을 분석하여 요약, 조치 사항, 치명적 이슈를 반환합니다.

        Parameters
        ----------
        entries : List[TraceEntry]
            파싱된 trace 엔트리 목록.

        Returns
        -------
        dict with keys:
            summary        : str   – 전반적인 요약
            actions        : list  – 권장 조치 사항 (문자열 리스트)
            critical_issues: list  – 즉각 대응이 필요한 치명적 이슈 (문자열 리스트)
        """
        if not entries:
            return {
                "summary": "분석할 엔트리가 없습니다.",
                "actions": [],
                "critical_issues": [],
            }

        if not self.api_key:
            raise ValueError("OpenAI API Key 가 설정되지 않았습니다.")

        # 분석 대상 엔트리 선택
        selected = self._select_entries(entries)
        user_prompt = self._build_user_prompt(entries, selected)

        # OpenAI API 호출
        response_text = self._call_openai(user_prompt)

        # JSON 파싱
        return self._parse_response(response_text)

    def analyze_single(self, entry: TraceEntry) -> str:
        """
        단일 엔트리에 대한 간략한 설명을 반환합니다 (한 문장).
        """
        if not self.api_key:
            raise ValueError("OpenAI API Key 가 설정되지 않았습니다.")

        prompt = (
            f"다음 SAP HANA trace 로그 항목 한 줄을 한국어로 간략히 설명해 주세요 (1–2 문장):\n"
            f"레벨: {entry.level}\n"
            f"컴포넌트: {entry.component}\n"
            f"메시지: {entry.message}"
        )
        return self._call_openai(prompt, system="You are a SAP HANA expert. Answer in Korean.")

    # ── 내부 헬퍼 ───────────────────────────────────────────────────────────

    def _build_client(self):
        """openai.OpenAI 클라이언트를 생성합니다 (openai 패키지 미설치 시 None)."""
        try:
            import openai  # noqa: PLC0415
            if self.api_key:
                return openai.OpenAI(api_key=self.api_key)
        except ImportError:
            pass
        return None

    def _select_entries(self, entries: List[TraceEntry]) -> List[TraceEntry]:
        """
        API 전송을 위해 중요 엔트리를 우선 선택합니다.
        - FATAL / ERROR / WARNING 를 먼저 채우고
        - 나머지는 INFO / DEBUG 에서 균등 샘플
        """
        priority = [e for e in entries if e.level in _PRIORITY_LEVELS]
        others = [e for e in entries if e.level not in _PRIORITY_LEVELS]

        if len(priority) >= self.max_entries:
            return priority[: self.max_entries]

        remaining = self.max_entries - len(priority)
        # others 에서 균등하게 샘플링
        if others and remaining > 0:
            step = max(1, len(others) // remaining)
            sampled_others = others[::step][:remaining]
        else:
            sampled_others = []

        return priority + sampled_others

    def _build_user_prompt(
        self, all_entries: List[TraceEntry], selected: List[TraceEntry]
    ) -> str:
        """사용자 프롬프트를 생성합니다."""
        # 통계 요약
        level_counts: Dict[str, int] = {}
        for e in all_entries:
            level_counts[e.level] = level_counts.get(e.level, 0) + 1

        stats_lines = "\n".join(
            f"  - {lvl}: {cnt} 건" for lvl, cnt in sorted(level_counts.items())
        )

        # 선택된 엔트리를 텍스트로 변환
        entry_lines = []
        for e in selected:
            ts = e.timestamp.strftime("%Y-%m-%d %H:%M:%S") if e.timestamp else e.timestamp_raw
            entry_lines.append(
                f"[{ts}] [{e.level}] [{e.component}] {e.message}"
            )

        entries_text = "\n".join(entry_lines)

        return (
            f"총 엔트리 수: {len(all_entries)}\n"
            f"레벨별 분포:\n{stats_lines}\n\n"
            f"분석 대상 엔트리 ({len(selected)} 건):\n"
            f"{entries_text}"
        )

    def _call_openai(
        self, user_prompt: str, system: Optional[str] = None
    ) -> str:
        """OpenAI Chat Completion API 를 호출하고 응답 텍스트를 반환합니다."""
        if self.client is None:
            self.client = self._build_client()

        if self.client is None:
            raise RuntimeError(
                "OpenAI 클라이언트를 초기화할 수 없습니다. "
                "'pip install openai' 명령으로 패키지를 설치하고 API Key 를 확인해주세요."
            )

        messages = [
            {"role": "system", "content": system or _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=1500,
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def _parse_response(text: str) -> Dict[str, Any]:
        """OpenAI 응답 텍스트에서 JSON 을 추출하여 파싱합니다."""
        # 마크다운 코드 블록 제거
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            # 첫 줄(```json 등)과 마지막 줄(```) 제거
            inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            cleaned = "\n".join(inner).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 원본 텍스트를 summary 에 담아 반환
            return {
                "summary": text,
                "actions": [],
                "critical_issues": [],
            }

        return {
            "summary": data.get("summary", ""),
            "actions": data.get("actions", []),
            "critical_issues": data.get("critical_issues", []),
        }
