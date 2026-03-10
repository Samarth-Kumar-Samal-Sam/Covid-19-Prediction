import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from prophet import Prophet

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="COVID-19 Analytics & Forecasting",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
    :root {
        --bg: #07111f;
        --card: rgba(20, 35, 58, 0.72);
        --card-2: rgba(16, 28, 46, 0.85);
        --border: rgba(255,255,255,0.08);
        --text: #f8fbff;
        --muted: #a8b7cc;
        --accent: #4da3ff;
        --accent2: #7b61ff;
        --success: #22c55e;
        --danger: #ef4444;
        --warning: #f59e0b;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(77,163,255,0.18), transparent 30%),
            radial-gradient(circle at top right, rgba(123,97,255,0.14), transparent 25%),
            linear-gradient(180deg, #040b16 0%, #081322 45%, #091827 100%);
        color: var(--text);
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1627 0%, #0f1d33 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    .hero {
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, rgba(10,25,47,0.95), rgba(42,111,191,0.82));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 26px;
        padding: 2rem 2rem 1.6rem 2rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.28);
    }

    .hero:before {
        content: "";
        position: absolute;
        inset: 0;
        background: radial-gradient(circle at 85% 20%, rgba(255,255,255,0.12), transparent 20%);
        pointer-events: none;
    }

    .hero-title {
        font-size: 2.35rem;
        font-weight: 800;
        line-height: 1.15;
        color: white;
        margin-bottom: 0.4rem;
        letter-spacing: -0.02em;
    }

    .hero-sub {
        color: rgba(255,255,255,0.9);
        font-size: 1.03rem;
        max-width: 900px;
        line-height: 1.6;
    }

    .glass-card {
        background: var(--card);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
    }

    .metric-wrap {
        background: linear-gradient(180deg, rgba(16,29,48,0.92), rgba(12,22,37,0.92));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 1.05rem 1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.20);
    }

    .metric-label {
        color: #a9b9cf;
        font-size: 0.92rem;
        font-weight: 600;
        margin-bottom: 0.45rem;
    }

    .metric-value {
        color: white;
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .metric-foot {
        margin-top: 0.45rem;
        color: #7fc4ff;
        font-size: 0.82rem;
    }

    .section-title {
        color: white;
        font-size: 1.25rem;
        font-weight: 760;
        margin: 0.2rem 0 0.9rem 0;
    }

    .subtle {
        color: var(--muted);
        font-size: 0.94rem;
    }

    div[data-testid="stTabs"] button {
        color: #cdd8e8;
        font-weight: 650;
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: white !important;
    }

    .pill {
        display: inline-block;
        padding: 0.28rem 0.7rem;
        border-radius: 999px;
        background: rgba(77,163,255,0.12);
        border: 1px solid rgba(77,163,255,0.25);
        color: #bfe0ff;
        font-size: 0.82rem;
        margin-right: 0.45rem;
        margin-top: 0.35rem;
    }

    .insight-box {
        background: linear-gradient(180deg, rgba(13, 25, 42, 0.95), rgba(10, 19, 32, 0.95));
        border: 1px solid rgba(255,255,255,0.08);
        border-left: 4px solid #4da3ff;
        border-radius: 18px;
        padding: 1rem 1rem;
        color: #dce8f7;
    }

    .footer-note {
        color: #93a6be;
        font-size: 0.9rem;
        text-align: center;
        padding-top: 0.5rem;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    .stDateInput > div > div,
    .stMultiSelect > div > div {
        background: rgba(15, 25, 40, 0.88) !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
    }

    .stButton > button {
        border-radius: 14px;
        font-weight: 700;
        border: 1px solid rgba(255,255,255,0.08);
        padding: 0.65rem 1rem;
    }

    .stDownloadButton > button {
        border-radius: 14px;
        font-weight: 700;
    }

    div[data-testid="stMetric"] {
        background: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPERS
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("./Dataset/covid-19.csv")

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    required = ["Country", "Date", "Confirmed", "Recovered", "Deaths"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in ["Confirmed", "Recovered", "Deaths"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df = df.sort_values(["Country", "Date"]).reset_index(drop=True)
    return df


def format_number(x):
    x = float(x)
    if x >= 1_000_000_000:
        return f"{x/1_000_000_000:.2f}B"
    if x >= 1_000_000:
        return f"{x/1_000_000:.2f}M"
    if x >= 1_000:
        return f"{x/1_000:.2f}K"
    return f"{x:,.0f}"


def metric_card(title, value, foot=""):
    st.markdown(
        f"""
        <div class="metric-wrap">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-foot">{foot}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


@st.cache_data
def get_filtered_data(df, start_date, end_date, countries):
    dff = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].copy()
    if countries:
        dff = dff[dff["Country"].isin(countries)].copy()
    return dff


@st.cache_data
def latest_snapshot(dff):
    if dff.empty:
        return pd.DataFrame(), None
    latest_date = dff["Date"].max()
    snap = dff[dff["Date"] == latest_date].copy()
    snap = snap.groupby("Country", as_index=False)[["Confirmed", "Recovered", "Deaths"]].sum()
    return snap, latest_date


@st.cache_data
def prepare_country_series(dff, country, metric):
    s = (
        dff[dff["Country"] == country][["Date", metric]]
        .rename(columns={"Date": "ds", metric: "y"})
        .groupby("ds", as_index=False)["y"].sum()
        .sort_values("ds")
    )
    return s


@st.cache_data
def top_country_table(dff, metric, n):
    return (
        dff.groupby("Country", as_index=False)[metric]
        .sum()
        .sort_values(metric, ascending=False)
        .head(n)
    )


def build_forecast(history, periods):
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True
    )
    model.fit(history)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return model, forecast


def forecast_plot(history, forecast, metric, country, years):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=history["ds"],
        y=history["y"],
        mode="lines",
        name="Historical",
        line=dict(width=3)
    ))

    fig.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat"],
        mode="lines",
        name="Forecast",
        line=dict(width=3, dash="dash")
    ))

    fig.add_trace(go.Scatter(
        x=pd.concat([forecast["ds"], forecast["ds"][::-1]]),
        y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(77,163,255,0.18)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="Confidence Interval"
    ))

    fig.update_layout(
        title=f"{metric} Forecast for {country} ({years} year(s))",
        template="plotly_dark",
        height=520,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", y=1.04, x=1, xanchor="right")
    )
    return fig


# =========================================================
# LOAD
# =========================================================
try:
    df = load_data()
except Exception as e:
    st.error(f"Dataset loading failed: {e}")
    st.stop()

# =========================================================
# HERO
# =========================================================
st.markdown("""
<div class="hero">
    <div class="hero-title">🦠 COVID-19 Analytics & Forecasting Studio</div>
    <div class="hero-sub">
        Explore global COVID-19 progression with an interface designed like a modern analytics platform.
        Analyze country-level trends, compare outbreak patterns, and generate forecast scenarios using Prophet.
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# TOP FILTER BAR
# =========================================================
min_date = df["Date"].min().date()
max_date = df["Date"].max().date()
countries = sorted(df["Country"].dropna().unique().tolist())

with st.container():
    c1, c2, c3, c4 = st.columns([1.3, 1.5, 1.1, 1.1])

    with c1:
        date_range = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

    with c2:
        selected_countries = st.multiselect(
            "Countries",
            options=countries,
            default=[]
        )

    with c3:
        metric_focus = st.selectbox(
            "Primary Metric",
            ["Confirmed", "Recovered", "Deaths"],
            index=0
        )

    with c4:
        top_n = st.slider(
            "Top Countries",
            min_value=5,
            max_value=20,
            value=10
        )

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])
else:
    start_date = pd.to_datetime(min_date)
    end_date = pd.to_datetime(max_date)

dff = get_filtered_data(df, start_date, end_date, selected_countries)

if dff.empty:
    st.warning("No data is available for the selected filters.")
    st.stop()

snap, latest_date = latest_snapshot(dff)

# =========================================================
# KPI CARDS
# =========================================================
total_confirmed = snap["Confirmed"].sum()
total_recovered = snap["Recovered"].sum()
total_deaths = snap["Deaths"].sum()
country_count = snap["Country"].nunique()

k1, k2, k3, k4 = st.columns(4)
with k1:
    metric_card("Latest Confirmed", format_number(total_confirmed), f"Snapshot: {latest_date.date()}")
with k2:
    metric_card("Latest Recovered", format_number(total_recovered), f"Filtered range")
with k3:
    metric_card("Latest Deaths", format_number(total_deaths), f"Across selected view")
with k4:
    metric_card("Countries in Scope", format_number(country_count), "Active selection")

st.markdown(
    f"<div class='subtle' style='margin-top:0.5rem;'>Showing data from <b>{start_date.date()}</b> to <b>{end_date.date()}</b>.</div>",
    unsafe_allow_html=True
)

st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Country Deep Dive",
    "Forecasting",
    "Dataset"
])

# =========================================================
# TAB 1
# =========================================================
with tab1:
    left, right = st.columns([1.55, 1])

    with left:
        st.markdown("<div class='section-title'>Global Situation Map</div>", unsafe_allow_html=True)

        map_fig = px.choropleth(
            snap,
            locations="Country",
            locationmode="country names",
            color=metric_focus,
            hover_name="Country",
            color_continuous_scale="Blues" if metric_focus == "Confirmed" else ("Greens" if metric_focus == "Recovered" else "Reds"),
            template="plotly_dark"
        )
        map_fig.update_layout(
            height=540,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            geo=dict(bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(map_fig, use_container_width=True)

    with right:
        st.markdown("<div class='section-title'>Quick Insights</div>", unsafe_allow_html=True)

        top_confirmed = top_country_table(dff, "Confirmed", 1)
        top_deaths = top_country_table(dff, "Deaths", 1)
        top_recovered = top_country_table(dff, "Recovered", 1)

        top_confirmed_country = top_confirmed.iloc[0]["Country"]
        top_deaths_country = top_deaths.iloc[0]["Country"]
        top_recovered_country = top_recovered.iloc[0]["Country"]

        st.markdown(f"""
        <div class="insight-box">
            <b>Top confirmed cases:</b> {top_confirmed_country}<br><br>
            <b>Top deaths:</b> {top_deaths_country}<br><br>
            <b>Top recoveries:</b> {top_recovered_country}<br><br>
            <b>Current focus metric:</b> {metric_focus}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

        summary_series = dff.groupby("Date", as_index=False)[metric_focus].sum()
        spark = px.area(
            summary_series,
            x="Date",
            y=metric_focus,
            template="plotly_dark",
            title=f"Global {metric_focus} Trend"
        )
        spark.update_layout(
            height=250,
            margin=dict(l=10, r=10, t=45, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(spark, use_container_width=True)

    b1, b2 = st.columns([1, 1])

    with b1:
        st.markdown("<div class='section-title'>Top Countries Ranking</div>", unsafe_allow_html=True)
        top_table = top_country_table(dff, metric_focus, top_n)

        bar_fig = px.bar(
            top_table.sort_values(metric_focus),
            x=metric_focus,
            y="Country",
            orientation="h",
            color=metric_focus,
            color_continuous_scale="Viridis",
            template="plotly_dark"
        )
        bar_fig.update_layout(
            height=500,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(autorange="reversed"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    with b2:
        st.markdown("<div class='section-title'>Metric Comparison</div>", unsafe_allow_html=True)

        latest_compare = snap.sort_values(metric_focus, ascending=False).head(top_n)
        comp_fig = go.Figure()
        for metric in ["Confirmed", "Recovered", "Deaths"]:
            comp_fig.add_trace(go.Bar(
                x=latest_compare["Country"],
                y=latest_compare[metric],
                name=metric
            ))
        comp_fig.update_layout(
            barmode="group",
            template="plotly_dark",
            height=500,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(comp_fig, use_container_width=True)

# =========================================================
# TAB 2
# =========================================================
with tab2:
    st.markdown("<div class='section-title'>Country-Level Exploration</div>", unsafe_allow_html=True)

    available = sorted(dff["Country"].unique().tolist())
    default_country = "India" if "India" in available else available[0]

    csel1, csel2 = st.columns([1, 1])
    with csel1:
        country_a = st.selectbox("Primary Country", available, index=available.index(default_country))
    with csel2:
        country_b = st.selectbox("Compare With", available, index=0)

    df_a = dff[dff["Country"] == country_a].sort_values("Date")
    df_b = dff[dff["Country"] == country_b].sort_values("Date")

    metric_view = st.radio(
        "Choose metric",
        ["Confirmed", "Recovered", "Deaths"],
        horizontal=True
    )

    p1, p2, p3 = st.columns(3)
    latest_a = df_a.iloc[-1]
    with p1:
        metric_card(f"{country_a} {metric_view}", format_number(latest_a[metric_view]), "Latest value")
    with p2:
        growth = df_a[metric_view].diff().fillna(0).tail(7).mean()
        metric_card("Avg Daily Change (7d)", format_number(max(growth, 0)), "Recent momentum")
    with p3:
        peak = df_a[metric_view].max()
        metric_card("Peak Observed", format_number(peak), "Within selected window")

    trend_fig = go.Figure()
    trend_fig.add_trace(go.Scatter(
        x=df_a["Date"], y=df_a[metric_view], mode="lines", name=country_a, line=dict(width=3)
    ))
    trend_fig.add_trace(go.Scatter(
        x=df_b["Date"], y=df_b[metric_view], mode="lines", name=country_b, line=dict(width=3, dash="dot")
    ))
    trend_fig.update_layout(
        title=f"{metric_view}: {country_a} vs {country_b}",
        template="plotly_dark",
        height=520,
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(trend_fig, use_container_width=True)

    lower1, lower2 = st.columns([1.1, 0.9])

    with lower1:
        daily_df = df_a[["Date", metric_view]].copy()
        daily_df["Daily Change"] = daily_df[metric_view].diff().fillna(0)

        daily_fig = px.bar(
            daily_df,
            x="Date",
            y="Daily Change",
            title=f"{country_a}: Daily Change in {metric_view}",
            template="plotly_dark"
        )
        daily_fig.update_layout(
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(daily_fig, use_container_width=True)

    with lower2:
        st.markdown("<div class='section-title'>Country Insight Panel</div>", unsafe_allow_html=True)
        ratio = (latest_a["Deaths"] / latest_a["Confirmed"] * 100) if latest_a["Confirmed"] > 0 else 0
        rec_ratio = (latest_a["Recovered"] / latest_a["Confirmed"] * 100) if latest_a["Confirmed"] > 0 else 0

        st.markdown(f"""
        <div class="insight-box">
            <b>Country:</b> {country_a}<br><br>
            <b>Death-to-confirmed ratio:</b> {ratio:.2f}%<br><br>
            <b>Recovery-to-confirmed ratio:</b> {rec_ratio:.2f}%<br><br>
            <b>Comparison country:</b> {country_b}<br><br>
            <b>Analysis metric:</b> {metric_view}
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# TAB 3
# =========================================================
with tab3:
    st.markdown("<div class='section-title'>Forecasting Studio</div>", unsafe_allow_html=True)

    f1, f2, f3 = st.columns([1, 1, 1])
    with f1:
        forecast_country = st.selectbox("Forecast Country", sorted(dff["Country"].unique()))
    with f2:
        forecast_metric = st.selectbox("Forecast Metric", ["Confirmed", "Recovered", "Deaths"])
    with f3:
        forecast_years = st.slider("Forecast Horizon (Years)", 1, 5, 2)

    run = st.button("Generate Forecast")

    if run:
        history = prepare_country_series(dff, forecast_country, forecast_metric)

        if len(history) < 10:
            st.warning("Not enough observations to build a reliable forecast.")
        else:
            with st.spinner("Training forecasting model..."):
                periods = forecast_years * 365
                model, forecast = build_forecast(history, periods)

            card1, card2, card3 = st.columns(3)
            final_row = forecast.iloc[-1]
            with card1:
                metric_card("Forecast Value", format_number(max(final_row["yhat"], 0)), "Projected endpoint")
            with card2:
                metric_card("Upper Bound", format_number(max(final_row["yhat_upper"], 0)), "Optimistic range")
            with card3:
                metric_card("Lower Bound", format_number(max(final_row["yhat_lower"], 0)), "Conservative range")

            st.plotly_chart(
                forecast_plot(history, forecast, forecast_metric, forecast_country, forecast_years),
                use_container_width=True
            )

            cplot1, cplot2 = st.columns([1.1, 0.9])

            with cplot1:
                components_fig = model.plot_components(forecast)
                st.pyplot(components_fig)

            with cplot2:
                st.markdown("<div class='section-title'>Forecast Summary</div>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="insight-box">
                    <b>Country:</b> {forecast_country}<br><br>
                    <b>Metric:</b> {forecast_metric}<br><br>
                    <b>Forecast horizon:</b> {forecast_years} year(s)<br><br>
                    <b>Historical observations:</b> {len(history)}<br><br>
                    <b>Latest projected value:</b> {format_number(max(final_row["yhat"], 0))}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div class='section-title'>Forecast Table</div>", unsafe_allow_html=True)
            forecast_table = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(20).copy()
            forecast_table.columns = ["Date", "Forecast", "Lower Bound", "Upper Bound"]
            st.dataframe(forecast_table, use_container_width=True, height=380)

# =========================================================
# TAB 4
# =========================================================
with tab4:
    st.markdown("<div class='section-title'>Filtered Dataset</div>", unsafe_allow_html=True)
    st.dataframe(dff, use_container_width=True, height=500)

    csv = dff.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Filtered CSV",
        data=csv,
        file_name="filtered_covid_data.csv",
        mime="text/csv"
    )

    st.markdown("<div class='section-title'>Country Summary Table</div>", unsafe_allow_html=True)
    summary_table = (
        dff.groupby("Country", as_index=False)[["Confirmed", "Recovered", "Deaths"]]
        .sum()
        .sort_values("Confirmed", ascending=False)
    )
    st.dataframe(summary_table, use_container_width=True, height=420)

