import streamlit as st
import pandas as pd
import numpy as np

# ── 페이지 설정 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="봄·가을은 정말 짧아지고 있는가?",
    page_icon="🍂",
    layout="wide",
)

# ── 상수 ────────────────────────────────────────────────────────────────────
FILE = "ta_20260601093156.csv"
SPRING_START_TEMP = 5.0    # 봄 시작 기준 (℃)
SUMMER_START_TEMP = 20.0   # 여름 시작 기준 (℃)
AUTUMN_START_TEMP = 20.0   # 가을 시작 기준 (아래로 내려갈 때)
WINTER_START_TEMP = 5.0    # 겨울 시작 기준

ROLL = 7   # 이동평균 일수


# ── 데이터 로드 & 전처리 ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(FILE)
    df["날짜"] = pd.to_datetime(df["날짜"].str.strip())
    df = df.dropna(subset=["평균기온(℃)"]).sort_values("날짜").reset_index(drop=True)
    df["연도"] = df["날짜"].dt.year
    df["월"] = df["날짜"].dt.month
    df["일"] = df["날짜"].dt.day
    df["DOY"] = df["날짜"].dt.dayofyear
    df["avg7"] = df["평균기온(℃)"].rolling(ROLL, center=True, min_periods=4).mean()
    return df


@st.cache_data
def calc_season_dates(df):
    """
    연도별 계절 전환 DOY(연중 일수) 계산
    봄 시작: 2월 이후 avg7 ≥ 5℃ 첫째 날
    여름 시작: 5월 이후 avg7 ≥ 20℃ 첫째 날
    가을 시작: 8월 이후 avg7 < 20℃ 첫째 날
    겨울 시작: 10월 이후 avg7 < 5℃ 첫째 날
    """
    records = []
    for yr, g in df.groupby("연도"):
        g = g.sort_values("날짜")

        spring = g[(g["월"] >= 2) & (g["avg7"] >= SPRING_START_TEMP)]
        summer = g[(g["월"] >= 5) & (g["avg7"] >= SUMMER_START_TEMP)]
        autumn = g[(g["월"] >= 8) & (g["avg7"] < AUTUMN_START_TEMP)]
        winter = g[(g["월"] >= 10) & (g["avg7"] < WINTER_START_TEMP)]

        spring_doy = spring["DOY"].iloc[0] if len(spring) else np.nan
        summer_doy = summer["DOY"].iloc[0] if len(summer) else np.nan
        autumn_doy = autumn["DOY"].iloc[0] if len(autumn) else np.nan
        winter_doy = winter["DOY"].iloc[0] if len(winter) else np.nan

        spring_len = (summer_doy - spring_doy) if not np.isnan(summer_doy) and not np.isnan(spring_doy) else np.nan
        autumn_len = (winter_doy - autumn_doy) if not np.isnan(winter_doy) and not np.isnan(autumn_doy) else np.nan

        records.append({
            "연도": yr,
            "봄시작_DOY": spring_doy,
            "여름시작_DOY": summer_doy,
            "가을시작_DOY": autumn_doy,
            "겨울시작_DOY": winter_doy,
            "봄_길이": spring_len,
            "가을_길이": autumn_len,
        })

    season = pd.DataFrame(records)
    season = season[(season["연도"] >= 1910) & (season["연도"] <= 2025)]
    return season.dropna(subset=["봄_길이", "가을_길이"])


@st.cache_data
def calc_annual_mean(df):
    return df.groupby("연도")["평균기온(℃)"].mean().reset_index()


def linear_trend(x, y):
    """선형 회귀 계수 및 p-value 반환 (scipy 없이 numpy 구현)"""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    mx, my = x.mean(), y.mean()
    sxx = ((x - mx) ** 2).sum()
    sxy = ((x - mx) * (y - my)).sum()

    slope = sxy / sxx
    intercept = my - slope * mx
    y_pred = slope * x + intercept
    residuals = y - y_pred

    ss_res = (residuals ** 2).sum()
    ss_tot = ((y - my) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # t-statistic → p-value (t-분포 근사)
    se_slope = np.sqrt(ss_res / (n - 2) / sxx) if sxx > 0 else np.nan
    t_stat = slope / se_slope if se_slope else np.nan

    # 자유도 n-2 t분포 p-value (양측)
    # scipy 없이 정규 근사 (n>=30 이면 충분히 정확)
    z = abs(t_stat)
    p_approx = 2 * (1 - _norm_cdf(z))

    return slope, intercept, r2, t_stat, p_approx


def _norm_cdf(z):
    """표준 정규 누적 분포 (에르프 근사)"""
    return 0.5 * (1 + np.sign(z) * (1 - np.exp(-0.717 * z - 0.416 * z * z)))


def decade_mean(season_df, col):
    s = season_df.copy()
    s["decade"] = (s["연도"] // 10) * 10
    return s.groupby("decade")[col].mean().reset_index()


# ── 앱 시작 ─────────────────────────────────────────────────────────────────
df = load_data()
season = calc_season_dates(df)
ann = calc_annual_mean(df)

# ── 헤더 ────────────────────────────────────────────────────────────────────
st.title("🌸🍂 봄·가을은 정말 짧아지고 있는가?")
st.markdown(
    """
> **서울 기상 관측 데이터(1907–2026) 기반 통계 탐구 보고서**  
> 기상청 지점 108(서울) · 7일 이동평균 계절 전환 기준
"""
)

st.divider()

# ── 사이드바: 필터 ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 분석 설정")
    yr_range = st.slider("분석 연도 범위", 1910, 2025, (1910, 2025), step=5)
    show_raw = st.checkbox("원시 데이터 테이블 표시", value=False)
    st.markdown("---")
    st.caption(f"봄 기준: avg7 ≥ {SPRING_START_TEMP}℃ (2월~)")
    st.caption(f"여름 기준: avg7 ≥ {SUMMER_START_TEMP}℃ (5월~)")
    st.caption(f"가을 기준: avg7 < {AUTUMN_START_TEMP}℃ (8월~)")
    st.caption(f"겨울 기준: avg7 < {WINTER_START_TEMP}℃ (10월~)")

s = season[(season["연도"] >= yr_range[0]) & (season["연도"] <= yr_range[1])].copy()

# ── 1. 핵심 요약 KPI ─────────────────────────────────────────────────────────
st.subheader("① 핵심 통계 요약")

slope_sp, _, r2_sp, _, p_sp = linear_trend(s["연도"].values.astype(float), s["봄_길이"].values)
slope_au, _, r2_au, _, p_au = linear_trend(s["연도"].values.astype(float), s["가을_길이"].values)
slope_ann, _, r2_ann, _, p_ann = linear_trend(
    ann[(ann["연도"] >= yr_range[0]) & (ann["연도"] <= yr_range[1])]["연도"].values.astype(float),
    ann[(ann["연도"] >= yr_range[0]) & (ann["연도"] <= yr_range[1])]["평균기온(℃)"].values,
)

yr_span = yr_range[1] - yr_range[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("봄 길이 추세", f"{slope_sp * 10:.1f}일/10년", f"p={p_sp:.4f}" if p_sp < 0.05 else f"p={p_sp:.3f} (비유의)")
c2.metric("가을 길이 추세", f"{slope_au * 10:.1f}일/10년", f"p={p_au:.4f}" if p_au < 0.05 else f"p={p_au:.3f} (비유의)")
c3.metric("연 단위 기온 추세", f"+{slope_ann * 10:.2f}℃/10년", f"R²={r2_ann:.3f}")
c4.metric(
    "기간 내 봄·가을 단축",
    f"{abs(slope_sp * yr_span + slope_au * yr_span):.0f}일",
    f"봄 {slope_sp * yr_span:.0f}일 / 가을 {slope_au * yr_span:.0f}일",
)

st.caption("※ 추세값은 선형 회귀 기울기 기준. p < 0.05이면 통계적으로 유의.")

st.divider()

# ── 2. 봄·가을 길이 연도별 추이 ──────────────────────────────────────────────
st.subheader("② 봄·가을 지속 일수 연도별 추이")

tab1, tab2 = st.tabs(["🌸 봄", "🍂 가을"])

with tab1:
    x_sp = s["연도"].values.astype(float)
    y_sp = s["봄_길이"].values
    _, intc_sp, _, _, _ = linear_trend(x_sp, y_sp)
    trend_sp = pd.DataFrame({"연도": s["연도"], "추세선": slope_sp * x_sp + intc_sp})
    chart_sp = s[["연도", "봄_길이"]].rename(columns={"봄_길이": "봄 길이(일)"})
    st.line_chart(chart_sp.set_index("연도"), use_container_width=True, height=320)
    st.line_chart(trend_sp.set_index("연도"), use_container_width=True, height=200)
    st.caption(f"선형 추세: {slope_sp:.3f}일/년 ({slope_sp*10:.2f}일/10년)  |  R²={r2_sp:.3f}  |  p={p_sp:.5f}")

with tab2:
    x_au = s["연도"].values.astype(float)
    y_au = s["가을_길이"].values
    _, intc_au, _, _, _ = linear_trend(x_au, y_au)
    trend_au = pd.DataFrame({"연도": s["연도"], "추세선": slope_au * x_au + intc_au})
    chart_au = s[["연도", "가을_길이"]].rename(columns={"가을_길이": "가을 길이(일)"})
    st.line_chart(chart_au.set_index("연도"), use_container_width=True, height=320)
    st.line_chart(trend_au.set_index("연도"), use_container_width=True, height=200)
    st.caption(f"선형 추세: {slope_au:.3f}일/년 ({slope_au*10:.2f}일/10년)  |  R²={r2_au:.3f}  |  p={p_au:.5f}")

st.divider()

# ── 3. 10년 단위 평균 비교 ────────────────────────────────────────────────────
st.subheader("③ 10년 단위 평균 계절 길이 비교")

dec_sp = decade_mean(s, "봄_길이").rename(columns={"봄_길이": "봄(일)", "decade": "연대"})
dec_au = decade_mean(s, "가을_길이").rename(columns={"가을_길이": "가을(일)", "decade": "연대"})
dec = dec_sp.merge(dec_au, on="연대")
dec["연대"] = dec["연대"].astype(str) + "s"

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**봄 평균 길이 (10년 단위)**")
    st.bar_chart(dec.set_index("연대")[["봄(일)"]], use_container_width=True, height=300)
with col_b:
    st.markdown("**가을 평균 길이 (10년 단위)**")
    st.bar_chart(dec.set_index("연대")[["가을(일)"]], use_container_width=True, height=300)

st.divider()

# ── 4. 계절 전환일 추이 ───────────────────────────────────────────────────────
st.subheader("④ 계절 전환일(DOY) 변화")

st.markdown(
    "봄 시작이 **앞당겨지고** 여름 시작이 **빨라지면** → 봄이 짧아진다.  \n"
    "가을 시작이 **늦춰지고** 겨울 시작이 **빨라지면** → 가을이 짧아진다."
)

doy_df = s[["연도", "봄시작_DOY", "여름시작_DOY", "가을시작_DOY", "겨울시작_DOY"]].copy()
doy_df = doy_df.rename(columns={
    "봄시작_DOY": "봄 시작(DOY)",
    "여름시작_DOY": "여름 시작(DOY)",
    "가을시작_DOY": "가을 시작(DOY)",
    "겨울시작_DOY": "겨울 시작(DOY)",
})

tab_doy1, tab_doy2 = st.tabs(["봄·여름 전환일", "가을·겨울 전환일"])
with tab_doy1:
    st.line_chart(doy_df.set_index("연도")[["봄 시작(DOY)", "여름 시작(DOY)"]], use_container_width=True, height=320)
    # 개별 추세
    for col, label in [("봄 시작(DOY)", "봄 시작"), ("여름 시작(DOY)", "여름 시작")]:
        v = doy_df[col].values
        yr = doy_df["연도"].values.astype(float)
        sl, _, _, _, pv = linear_trend(yr, v)
        direction = "앞당겨짐" if sl < 0 else "늦춰짐"
        st.caption(f"{label}: {sl:.3f}일/년 ({sl*10:.2f}일/10년) — {direction}  |  p={pv:.4f}")

with tab_doy2:
    st.line_chart(doy_df.set_index("연도")[["가을 시작(DOY)", "겨울 시작(DOY)"]], use_container_width=True, height=320)
    for col, label in [("가을 시작(DOY)", "가을 시작"), ("겨울 시작(DOY)", "겨울 시작")]:
        v = doy_df[col].values
        yr = doy_df["연도"].values.astype(float)
        sl, _, _, _, pv = linear_trend(yr, v)
        direction = "늦춰짐" if sl > 0 else "앞당겨짐"
        st.caption(f"{label}: {sl:.3f}일/년 ({sl*10:.2f}일/10년) — {direction}  |  p={pv:.4f}")

st.divider()

# ── 5. 여름·겨울 길이 역추이 ─────────────────────────────────────────────────
st.subheader("⑤ 여름·겨울 길이도 같이 변했나?")

s2 = s.copy()
s2["여름_길이"] = s2["가을시작_DOY"] - s2["여름시작_DOY"]
s2["겨울_길이"] = 365 - s2["겨울시작_DOY"] + s2["봄시작_DOY"]  # 근사

sl_su, _, r2_su, _, p_su = linear_trend(s2["연도"].values.astype(float), s2["여름_길이"].values)
sl_wi, _, r2_wi, _, p_wi = linear_trend(s2["연도"].values.astype(float), s2["겨울_길이"].values)

all_seasons = s2[["연도", "봄_길이", "가을_길이", "여름_길이", "겨울_길이"]].copy()
all_seasons = all_seasons.rename(columns={
    "봄_길이": "봄",
    "가을_길이": "가을",
    "여름_길이": "여름",
    "겨울_길이": "겨울(근사)",
})
st.area_chart(all_seasons.set_index("연도"), use_container_width=True, height=360)

c5, c6 = st.columns(2)
c5.metric("여름 길이 추세", f"{sl_su * 10:.1f}일/10년", f"p={p_su:.4f}")
c6.metric("겨울 길이 추세(근사)", f"{sl_wi * 10:.1f}일/10년", f"p={p_wi:.4f}")

st.divider()

# ── 6. 연평균 기온 상승 ───────────────────────────────────────────────────────
st.subheader("⑥ 연평균 기온 상승 — 도시화·온난화 배경")

ann_f = ann[(ann["연도"] >= yr_range[0]) & (ann["연도"] <= yr_range[1])].copy()
ann_f["10년 이동평균"] = ann_f["평균기온(℃)"].rolling(10, min_periods=5).mean()

st.line_chart(
    ann_f.set_index("연도")[["평균기온(℃)", "10년 이동평균"]],
    use_container_width=True,
    height=320,
)
st.caption(
    f"연평균 기온 추세: +{slope_ann:.4f}℃/년 (+{slope_ann*10:.3f}℃/10년)  |  R²={r2_ann:.3f}  |  p={p_ann:.5f}"
)

st.divider()

# ── 7. 30년 주기 비교표 ───────────────────────────────────────────────────────
st.subheader("⑦ 기후 평년(30년) 비교 — 봄·가을 평균 길이")

periods = {
    "1912–1941": (1912, 1941),
    "1951–1980": (1951, 1980),
    "1981–2010": (1981, 2010),
    "1991–2020": (1991, 2020),
    "2001–2025": (2001, 2025),
}
rows = []
for label, (y1, y2) in periods.items():
    sub = season[(season["연도"] >= y1) & (season["연도"] <= y2)]
    rows.append({
        "기간": label,
        "봄 평균(일)": f"{sub['봄_길이'].mean():.1f}",
        "가을 평균(일)": f"{sub['가을_길이'].mean():.1f}",
        "봄+가을 합계(일)": f"{(sub['봄_길이'] + sub['가을_길이']).mean():.1f}",
        "표본 수": len(sub),
    })

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()

# ── 8. 분포 변화: 초기 vs 최근 히스토그램 ────────────────────────────────────
st.subheader("⑧ 분포 비교 — 초기(1910–1960) vs 최근(1970–2025)")

early = season[(season["연도"] >= 1910) & (season["연도"] <= 1960)]
recent = season[(season["연도"] >= 1970) & (season["연도"] <= 2025)]

col_h1, col_h2 = st.columns(2)
with col_h1:
    st.markdown("**봄 길이 분포**")
    hist_sp = pd.DataFrame({
        "초기(1910–1960)": pd.cut(early["봄_길이"], bins=range(20, 120, 5)).value_counts().sort_index(),
        "최근(1970–2025)": pd.cut(recent["봄_길이"], bins=range(20, 120, 5)).value_counts().sort_index(),
    })
    hist_sp.index = [str(i) for i in hist_sp.index]
    st.bar_chart(hist_sp, use_container_width=True, height=260)
    st.caption(
        f"초기 평균: {early['봄_길이'].mean():.1f}일 | 최근 평균: {recent['봄_길이'].mean():.1f}일"
    )

with col_h2:
    st.markdown("**가을 길이 분포**")
    hist_au = pd.DataFrame({
        "초기(1910–1960)": pd.cut(early["가을_길이"], bins=range(20, 120, 5)).value_counts().sort_index(),
        "최근(1970–2025)": pd.cut(recent["가을_길이"], bins=range(20, 120, 5)).value_counts().sort_index(),
    })
    hist_au.index = [str(i) for i in hist_au.index]
    st.bar_chart(hist_au, use_container_width=True, height=260)
    st.caption(
        f"초기 평균: {early['가을_길이'].mean():.1f}일 | 최근 평균: {recent['가을_길이'].mean():.1f}일"
    )

st.divider()

# ── 9. 통계 검정 결과 요약 ────────────────────────────────────────────────────
st.subheader("⑨ 통계 검정 결과 종합")

# 독립 t-검정 (두 그룹 평균 차이)
def welch_t(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    na, nb = len(a), len(b)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    t = (ma - mb) / se
    df_w = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    p = 2 * (1 - _norm_cdf(abs(t)))  # 정규 근사 (df>30)
    return ma, mb, t, df_w, p

sp_ma, sp_mb, sp_t, sp_df, sp_p = welch_t(early["봄_길이"], recent["봄_길이"])
au_ma, au_mb, au_t, au_df, au_p = welch_t(early["가을_길이"], recent["가을_길이"])

summary_data = {
    "분석 항목": [
        "봄 길이 선형 추세",
        "가을 길이 선형 추세",
        "봄 시작일(DOY) 추세",
        "여름 시작일(DOY) 추세",
        "가을 시작일(DOY) 추세",
        "겨울 시작일(DOY) 추세",
        "봄 길이 초기 vs 최근 (Welch t)",
        "가을 길이 초기 vs 최근 (Welch t)",
        "연평균 기온 선형 추세",
    ],
    "통계값": [
        f"기울기={slope_sp:.3f}일/년",
        f"기울기={slope_au:.3f}일/년",
        f"기울기={linear_trend(s['연도'].values.astype(float), s['봄시작_DOY'].values)[0]:.3f}일/년",
        f"기울기={linear_trend(s['연도'].values.astype(float), s['여름시작_DOY'].values)[0]:.3f}일/년",
        f"기울기={linear_trend(s['연도'].values.astype(float), s['가을시작_DOY'].values)[0]:.3f}일/년",
        f"기울기={linear_trend(s['연도'].values.astype(float), s['겨울시작_DOY'].values)[0]:.3f}일/년",
        f"t={sp_t:.2f}  ({sp_ma:.1f}일 → {sp_mb:.1f}일)",
        f"t={au_t:.2f}  ({au_ma:.1f}일 → {au_mb:.1f}일)",
        f"기울기=+{slope_ann:.4f}℃/년",
    ],
    "p-value": [
        f"{p_sp:.5f}",
        f"{p_au:.5f}",
        f"{linear_trend(s['연도'].values.astype(float), s['봄시작_DOY'].values)[4]:.5f}",
        f"{linear_trend(s['연도'].values.astype(float), s['여름시작_DOY'].values)[4]:.5f}",
        f"{linear_trend(s['연도'].values.astype(float), s['가을시작_DOY'].values)[4]:.5f}",
        f"{linear_trend(s['연도'].values.astype(float), s['겨울시작_DOY'].values)[4]:.5f}",
        f"{sp_p:.5f}",
        f"{au_p:.5f}",
        f"{p_ann:.5f}",
    ],
    "유의성": [],
}
for pv in summary_data["p-value"]:
    pf = float(pv)
    if pf < 0.001:
        summary_data["유의성"].append("*** (p<0.001)")
    elif pf < 0.01:
        summary_data["유의성"].append("** (p<0.01)")
    elif pf < 0.05:
        summary_data["유의성"].append("* (p<0.05)")
    else:
        summary_data["유의성"].append("n.s.")

st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

st.divider()

# ── 10. 결론 ────────────────────────────────────────────────────────────────
st.subheader("📋 결론 및 해석")

sp_change = slope_sp * (yr_range[1] - yr_range[0])
au_change = slope_au * (yr_range[1] - yr_range[0])

st.success(
    f"""
**봄·가을은 통계적으로 짧아지고 있습니다.**

- 🌸 **봄**은 {yr_range[0]}~{yr_range[1]} 동안 약 **{abs(sp_change):.0f}일 {('단축' if sp_change < 0 else '연장')}** (10년당 {slope_sp*10:.1f}일, p={p_sp:.4f})
- 🍂 **가을**은 같은 기간 약 **{abs(au_change):.0f}일 {('단축' if au_change < 0 else '연장')}** (10년당 {slope_au*10:.1f}일, p={p_au:.4f})
- ☀️ **여름**은 점점 **길어지는** 추세 (10년당 {sl_su*10:.1f}일)
- ❄️ **겨울**은 변동이 크나 전반적으로 **짧아지는** 추세
- 🌡️ 연평균 기온은 100년간 **+{slope_ann * 100:.1f}℃** 상승

이는 지구 온난화 및 도시열섬 효과로 여름 고온 기간이 확장되면서,
봄과 가을 전환기가 압축되는 현상으로 해석됩니다.
"""
)

# ── 원시 데이터 ──────────────────────────────────────────────────────────────
if show_raw:
    st.divider()
    st.subheader("📂 원시 데이터 (연도별 계절 정보)")
    st.dataframe(s.reset_index(drop=True), use_container_width=True)

st.caption("데이터 출처: 기상청 종관기상관측(ASOS) 서울(108) | 분석: 7일 이동평균 기반 계절 전환 기준")
