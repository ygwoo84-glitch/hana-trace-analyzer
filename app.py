"""
HANA Trace AI Analyzer - Streamlit UI
실행: streamlit run app.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

from core.parser import HANATraceParser, TraceEntry
from core.ai_analyzer import HANAAnalyzer

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit 페이지 설정
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HANA Trace AI Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS 스타일
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* 전체 배경 */
.main .block-container { padding-top: 1.5rem; }

/* 경보 박스 */
.alert-critical {
    background: #ffe0e0; border-left: 4px solid #d32f2f;
    padding: 10px 14px; border-radius: 6px; margin: 4px 0;
    font-size: 0.9rem;
}
.alert-high {
    background: #fff3e0; border-left: 4px solid #f57c00;
    padding: 10px 14px; border-radius: 6px; margin: 4px 0;
    font-size: 0.9rem;
}
.alert-medium {
    background: #fff9c4; border-left: 4px solid #f9a825;
    padding: 10px 14px; border-radius: 6px; margin: 4px 0;
    font-size: 0.9rem;
}
.alert-low {
    background: #e8f5e9; border-left: 4px solid #388e3c;
    padding: 10px 14px; border-radius: 6px; margin: 4px 0;
    font-size: 0.9rem;
}

/* 메트릭 박스 */
.metric-box {
    background: #f5f5f5; padding: 14px; border-radius: 10px;
    text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,.08);
}
.metric-label { font-size: 0.8rem; color: #666; margin-bottom: 4px; }
.metric-value { font-size: 2rem; font-weight: 700; }

/* 섹션 구분선 */
hr { margin: 1.2rem 0; }

/* 사이드바 로고 영역 */
.sidebar-logo { display: flex; align-items: center; gap: 10px; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼 함수
# ─────────────────────────────────────────────────────────────────────────────

def level_badge(level: str) -> str:
    """레벨에 맞는 HTML 배지를 반환합니다."""
    colors = {
        "FATAL":   ("#c62828", "#fff"),
        "ERROR":   ("#e53935", "#fff"),
        "WARNING": ("#fb8c00", "#fff"),
        "INFO":    ("#1e88e5", "#fff"),
        "DEBUG":   ("#757575", "#fff"),
    }
    bg, fg = colors.get(level, ("#9e9e9e", "#fff"))
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 8px;'
        f'border-radius:12px;font-size:0.75rem;font-weight:600;">{level}</span>'
    )


def render_alert(level: str, message: str) -> None:
    """레벨에 맞는 알림 박스를 렌더링합니다."""
    cls_map = {
        "FATAL": "alert-critical",
        "ERROR": "alert-critical",
        "WARNING": "alert-high",
        "INFO": "alert-low",
        "DEBUG": "alert-low",
    }
    cls = cls_map.get(level, "alert-low")
    st.markdown(
        f'<div class="{cls}">{level_badge(level)} {message}</div>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _try_decode(content_bytes: bytes) -> tuple[str, str]:
    """바이트를 여러 인코딩으로 시도하여 (decoded_str, encoding) 반환."""
    for enc in ["utf-8", "euc-kr", "cp1252", "latin-1"]:
        try:
            return content_bytes.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("all", b"", 0, 1, "지원하지 않는 인코딩입니다.")


# ─────────────────────────────────────────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # 로고 + 타이틀
    st.markdown(
        '<div class="sidebar-logo">'
        '<img src="https://www.sap.com/dam/application/shared/logos/sap-logo-svg.svg" width="60">'
        '</div>',
        unsafe_allow_html=True,
    )
    st.title("⚙️ 설정")

    # ── API Key ────────────────────────────────────────────────────────────
    st.subheader("🔑 OpenAI API Key")
    api_key_input = st.text_input(
        "API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="sk-... 형식의 OpenAI API Key를 입력하세요.",
    )
    model_choice = st.selectbox(
        "GPT 모델",
        ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
        index=0,
        help="분석에 사용할 GPT 모델을 선택하세요.",
    )

    # Analyzer 초기화
    analyzer: HANAAnalyzer | None = None
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input
        analyzer = HANAAnalyzer(api_key=api_key_input, model=model_choice)

    st.divider()

    # ── 파일 입력 ──────────────────────────────────────────────────────────
    st.subheader("📁 파일 입력")
    upload_method = st.radio("입력 방법 선택", ["파일 업로드", "로컬 파일 경로"])

    parser = HANATraceParser(merge_multiline=True)
    entries: list[TraceEntry] = []

    if upload_method == "파일 업로드":
        uploaded_file = st.file_uploader(
            "HANA Trace 파일 업로드",
            type=["trc", "txt", "log"],
            help=".trc / .txt / .log 파일을 업로드하세요.",
        )
        if uploaded_file is not None:
            try:
                content_bytes = uploaded_file.read()
                decoded_content, used_enc = _try_decode(content_bytes)

                col_a, col_b = st.columns(2)
                col_a.info(f"📁 {uploaded_file.name}")
                col_b.info(f"🔤 {used_enc}")
                st.info(f"📏 {len(content_bytes):,} bytes")

                lines = decoded_content.splitlines()

                # 미리보기
                with st.expander("🔍 파일 미리보기 (처음 15 줄)"):
                    for i, ln in enumerate(lines[:15]):
                        st.code(f"{i+1:>4}: {ln}", language="text")

                with st.spinner("🔍 파싱 중..."):
                    entries = parser.parse_lines(lines)

                if entries:
                    st.success(f"✅ {len(entries):,} 개 엔트리 파싱 완료")
                else:
                    st.warning("⚠️ 엔트리가 파싱되지 않았습니다. 파일 형식을 확인하세요.")
                    with st.expander("파일 내용 확인"):
                        st.code(decoded_content[:2000], language="text")

            except Exception as exc:
                st.error(f"❌ 파일 처리 실패: {exc}")
                import traceback
                st.code(traceback.format_exc(), language="python")

    else:  # 로컬 파일 경로
        filepath = st.text_input("HANA Trace 파일 경로", placeholder="/path/to/trace.trc")
        if filepath:
            if not os.path.exists(filepath):
                st.error("❌ 파일을 찾을 수 없습니다.")
            else:
                try:
                    for enc in ["utf-8", "euc-kr", "cp1252", "latin-1"]:
                        try:
                            with open(filepath, "r", encoding=enc) as f:
                                decoded_content = f.read()
                            used_enc = enc
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        raise ValueError("지원하지 않는 인코딩입니다.")

                    with st.spinner("🔍 파싱 중..."):
                        entries = parser.parse_lines(decoded_content.splitlines())

                    st.success(
                        f"✅ {os.path.basename(filepath)} ({len(entries):,} 개 엔트리)"
                    )
                except Exception as exc:
                    st.error(f"❌ 파일 읽기 실패: {exc}")
                    import traceback
                    st.code(traceback.format_exc(), language="python")

    st.divider()

    # 필터 옵션 (엔트리가 있을 때만)
    filter_levels: list[str] = []
    if entries:
        st.subheader("🔎 필터")
        available_levels = sorted({e.level for e in entries})
        filter_levels = st.multiselect(
            "레벨 필터",
            options=available_levels,
            default=available_levels,
            help="표시할 로그 레벨을 선택하세요.",
        )

    st.divider()
    st.caption("🏢 Dev.AI Code Agent | kt ds")


# ─────────────────────────────────────────────────────────────────────────────
# 메인 헤더
# ─────────────────────────────────────────────────────────────────────────────
st.title("🔍 HANA Trace AI Analyzer")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 엔트리가 있을 때만 표시
# ─────────────────────────────────────────────────────────────────────────────
if entries:
    # 필터 적용
    filtered = [e for e in entries if e.level in filter_levels] if filter_levels else entries

    # ── 개요 메트릭 ──────────────────────────────────────────────────────────
    st.subheader("📊 분석 개요")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("총 엔트리", f"{len(entries):,}")
    c2.metric("필터 결과", f"{len(filtered):,}")
    fatal_n  = sum(1 for e in filtered if e.level == "FATAL")
    error_n  = sum(1 for e in filtered if e.level == "ERROR")
    warn_n   = sum(1 for e in filtered if e.level == "WARNING")
    c3.metric("🔴 FATAL",   fatal_n,  delta_color="inverse")
    c4.metric("🟠 ERROR",   error_n,  delta_color="inverse")
    c5.metric("🟡 WARNING", warn_n,   delta_color="inverse")

    # ── 시각화 ──────────────────────────────────────────────────────────────
    st.subheader("📈 로그 분석 차트")
    tab_pie, tab_timeline, tab_component = st.tabs(
        ["레벨 분포", "시간대별 분포", "컴포넌트별 분포"]
    )

    level_counts: dict[str, int] = {}
    for e in filtered:
        level_counts[e.level] = level_counts.get(e.level, 0) + 1

    color_map = {
        "FATAL":   "#c62828",
        "ERROR":   "#e53935",
        "WARNING": "#fb8c00",
        "INFO":    "#1e88e5",
        "DEBUG":   "#757575",
    }

    with tab_pie:
        if level_counts:
            level_df = pd.DataFrame(
                [{"level": k, "count": v} for k, v in level_counts.items()]
            )
            fig_pie = px.pie(
                level_df, names="level", values="count",
                color="level", color_discrete_map=color_map,
                hole=0.45,
                title="로그 레벨 분포",
            )
            fig_pie.update_layout(margin=dict(t=40, b=0, l=0, r=0), height=320)
            st.plotly_chart(fig_pie, use_container_width=True)

    with tab_timeline:
        ts_entries = [e for e in filtered if e.timestamp is not None]
        if ts_entries:
            ts_df = pd.DataFrame(
                [{"timestamp": e.timestamp, "level": e.level} for e in ts_entries]
            )
            ts_df["timestamp"] = pd.to_datetime(ts_df["timestamp"])
            ts_df["hour"] = ts_df["timestamp"].dt.floor("h")
            agg = ts_df.groupby(["hour", "level"]).size().reset_index(name="count")
            fig_bar = px.bar(
                agg, x="hour", y="count", color="level",
                color_discrete_map=color_map,
                title="시간대별 로그 분포",
                labels={"hour": "시간", "count": "개수", "level": "레벨"},
            )
            fig_bar.update_layout(height=320, bargap=0.15)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("타임스탬프 정보가 있는 엔트리가 없습니다.")

    with tab_component:
        comp_counts: dict[str, int] = {}
        for e in filtered:
            if e.component:
                comp_counts[e.component] = comp_counts.get(e.component, 0) + 1
        if comp_counts:
            top_n = sorted(comp_counts.items(), key=lambda x: x[1], reverse=True)[:20]
            comp_df = pd.DataFrame(top_n, columns=["component", "count"])
            fig_comp = px.bar(
                comp_df, x="count", y="component", orientation="h",
                title="컴포넌트별 로그 빈도 (상위 20)",
                labels={"count": "개수", "component": "컴포넌트"},
                color="count",
                color_continuous_scale="Reds",
            )
            fig_comp.update_layout(height=max(300, 30 * len(comp_df)), yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.info("컴포넌트 정보가 있는 엔트리가 없습니다.")

    # ── 치명 / 오류 빠른 보기 ────────────────────────────────────────────────
    critical_entries = [e for e in filtered if e.level in ("FATAL", "ERROR")]
    if critical_entries:
        st.subheader(f"🚨 치명적 오류 / ERROR ({len(critical_entries):,} 건)")
        with st.expander(f"오류 상세 보기 ({len(critical_entries)} 건)", expanded=False):
            for e in critical_entries[:50]:  # 최대 50개 표시
                ts_str = (
                    e.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    if e.timestamp
                    else e.timestamp_raw or "-"
                )
                comp_str = f"[{e.component}] " if e.component else ""
                render_alert(e.level, f"{ts_str} {comp_str}{e.message}")
            if len(critical_entries) > 50:
                st.caption(f"... 이하 {len(critical_entries) - 50} 건 생략 (CSV 다운로드로 전체 확인)")

    # ── 원본 데이터 테이블 ───────────────────────────────────────────────────
    st.subheader("📄 원본 데이터 미리보기")
    df = parser.to_dataframe(filtered)

    # 검색
    search_text = st.text_input("🔍 메시지 검색", placeholder="검색어를 입력하세요...")
    if search_text and "message" in df.columns:
        df = df[df["message"].str.contains(search_text, case=False, na=False)]
        st.caption(f"검색 결과: {len(df):,} 건")

    st.dataframe(
        df.head(200),
        use_container_width=True,
        height=400,
    )

    # CSV 다운로드
    full_df = parser.to_dataframe(filtered)
    csv_data = full_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="📥 전체 데이터 CSV 다운로드",
        data=csv_data,
        file_name=f"hana_trace_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

    # ── AI 분석 ──────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🤖 AI 분석 결과")

    if analyzer is None:
        st.warning("⚠️ AI 분석을 사용하려면 사이드바에서 OpenAI API Key를 입력해주세요.")
    else:
        col_btn, col_info = st.columns([2, 5])
        with col_btn:
            run_ai = st.button("🚀 AI 분석 실행", type="primary", use_container_width=True)
        with col_info:
            st.caption(
                f"모델: **{model_choice}** | "
                f"분석 대상: **{min(200, len(entries))}** 개 (우선순위 기반 선택)"
            )

        if run_ai:
            with st.spinner("🤖 AI 분석 중... 잠시 기다려 주세요."):
                try:
                    analysis = analyzer.analyze(filtered)

                    st.markdown("---")

                    # 요약
                    st.markdown("### 📋 주요 발견 사항")
                    st.info(analysis["summary"])

                    # 권장 조치
                    if analysis.get("actions"):
                        st.markdown("### ⚠️ 권장 조치")
                        for i, action in enumerate(analysis["actions"], 1):
                            st.markdown(f"**{i}.** {action}")

                    # 치명적 이슈
                    if analysis.get("critical_issues"):
                        st.markdown("### 🔴 즉각 대응이 필요한 이슈")
                        for issue in analysis["critical_issues"]:
                            st.error(f"🚨 {issue}")
                    else:
                        st.success("✅ 즉각 대응이 필요한 치명적 이슈가 감지되지 않았습니다.")

                except Exception as exc:
                    st.error(f"❌ AI 분석 실패: {exc}")
                    import traceback
                    st.code(traceback.format_exc(), language="python")

# ─────────────────────────────────────────────────────────────────────────────
# 엔트리가 없을 때 안내 화면
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; color: #888;">
        <div style="font-size: 4rem;">📂</div>
        <h3>분석할 파일이 없습니다</h3>
        <p>사이드바에서 HANA Trace 파일을 업로드하거나 경로를 입력해주세요.</p>
        <br>
        <div style="text-align:left; max-width:500px; margin:auto; background:#f9f9f9;
                    padding:20px; border-radius:10px; font-size:0.9rem;">
            <b>지원 파일 형식</b><br>
            • .trc &nbsp;– SAP HANA Trace 파일<br>
            • .txt &nbsp;– 텍스트 로그 파일<br>
            • .log &nbsp;– 일반 로그 파일<br><br>
            <b>지원 인코딩</b><br>
            UTF-8, EUC-KR, CP1252, Latin-1<br><br>
            <b>지원 로그 포맷</b><br>
            • SAP HANA 표준 trace 포맷<br>
            • nameserver / indexserver 포맷<br>
            • ISO 타임스탬프 + 레벨 포맷<br>
            • 일반 시스템 로그 포맷
        </div>
    </div>
    """, unsafe_allow_html=True)
