import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from prophet import Prophet

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="COVID-19 Prediction Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 45%, #0ea5e9 100%);
        padding: 1.8rem 1.6rem;
        border-radius: 20px;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
        margin-bottom: 1.2rem;
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }

    .hero-subtitle {
        font-size: 1rem;
        opacity: 0.92;
    }

    .metric-card {
        background: #ffffff;
        padding: 1rem 1.1rem;
        border-radius: 16px;
        border: 1px solid rgba(0,0,0,0.08);
        box-shadow: 0 4px 18px rgba(0,0,0,0.06);
    }

    .metric-title {
        font-size: 0.95rem;
        color: #475569;
        margin-bottom: 0.35rem;
        font-weight: 600;
    }

    .metric-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #0f172a;
    }

    .section-header {
        font-size: 1.35rem;
        font-weight: 750;
        margin-top: 0.4rem;
        margin-bottom: 0.8rem;
        color: #0f172a;
    }

    .small-note {
        color: #64748b;
        font-size: 0.93rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #eef6ff 100%);
    }
</style>
""", unsafe_allow_html=True)

# =========================
# HELPERS
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("./Dataset/covid-19.csv")

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    required_cols = ["Country", "Date", "Confirmed", "Recovered", "Deaths"]
    missing = [c for c in required_cols if c not in df.columns]
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


def metric_card(title, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


@st.cache_data
def get_filtered_data(df, start_date, end_date, selected_countries):
    dff = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].copy()
    if selected_countries:
        dff = dff[dff["Country"].isin(selected_countries)].copy()
    return dff


@st.cache_data
def aggregate_latest_snapshot(dff):
    if dff.empty:
        return pd.DataFrame()

    latest_date = dff["Date"].max()
    latest_snapshot = dff[dff["Date"] == latest_date].copy()

    summary = latest_snapshot.groupby("Country", as_index=False)[["Confirmed", "Recovered", "Deaths"]].sum()
    return summary, latest_date


@st.cache_data
def top_countries(dff, metric, n=10):
    grouped = dff.groupby("Country", as_index=False)[metric].sum()
    grouped = grouped.sort_values(metric, ascending=False).head(n)
    return grouped


@st.cache_data
def prepare_country_metric_series(dff, country, metric):
    series = (
        dff[dff["Country"] == country][["Date", metric]]
        .rename(columns={"Date": "ds", metric: "y"})
        .groupby("ds", as_index=False)["y"].sum()
        .sort_values("ds")
    )
    return series


def build_forecast_plot(history, forecast, metric, country, years):
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
        fillcolor="rgba(14,165,233,0.18)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="Confidence Interval"
    ))

    fig.update_layout(
        title=f"{metric} Forecast for {country} ({years} year(s))",
        xaxis_title="Date",
        yaxis_title=metric,
        hovermode="x unified",
        template="plotly_white",
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


# =========================
# LOAD DATA
# =========================
try:
    df = load_data()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# =========================
# HERO SECTION
# =========================
st.markdown("""
<div class="hero-card">
    <div class="hero-title">🦠 COVID-19 Prediction Dashboard</div>
    <div class="hero-subtitle">
        Explore global COVID-19 trends, compare countries, visualize case progression,
        and generate interactive forecasts with Prophet.
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Dashboard Controls")
st.sidebar.markdown("Adjust filters and forecasting settings.")

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()
all_countries = sorted(df["Country"].dropna().unique().tolist())

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
else:
    start_date, end_date = pd.to_datetime(min_date), pd.to_datetime(max_date)

selected_countries = st.sidebar.multiselect(
    "Select Countries",
    options=all_countries,
    default=[]
)

top_n = st.sidebar.slider(
    "Top Countries to Show",
    min_value=5,
    max_value=20,
    value=10
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔮 Forecast Settings")

forecast_country_default = "India" if "India" in all_countries else all_countries[0]
forecast_country = st.sidebar.selectbox(
    "Country for Forecast",
    options=all_countries,
    index=all_countries.index(forecast_country_default)
)

forecast_years = st.sidebar.slider(
    "Years to Forecast",
    min_value=1,
    max_value=5,
    value=2
)

run_forecast = st.sidebar.button("Run Forecast", use_container_width=True)

# =========================
# FILTER DATA
# =========================
dff = get_filtered_data(df, start_date, end_date, selected_countries)

if dff.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

summary_df, latest_date = aggregate_latest_snapshot(dff)

# =========================
# KPI ROW
# =========================
total_confirmed = summary_df["Confirmed"].sum() if not summary_df.empty else 0
total_recovered = summary_df["Recovered"].sum() if not summary_df.empty else 0
total_deaths = summary_df["Deaths"].sum() if not summary_df.empty else 0
countries_count = summary_df["Country"].nunique() if not summary_df.empty else 0

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Latest Confirmed Cases", format_number(total_confirmed))
with c2:
    metric_card("Latest Recovered Cases", format_number(total_recovered))
with c3:
    metric_card("Latest Deaths", format_number(total_deaths))
with c4:
    metric_card("Countries in View", format_number(countries_count))

st.markdown(
    f"<div class='small-note'>Showing filtered data from <b>{start_date.date()}</b> to <b>{end_date.date()}</b>. "
    f"Latest snapshot used for KPI cards: <b>{latest_date.date()}</b>.</div>",
    unsafe_allow_html=True
)

st.markdown("---")

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs([
    "🌍 Global Overview",
    "📊 Country Analysis",
    "🗂 Dataset",
    "🔮 Forecasting"
])

# =========================
# TAB 1: GLOBAL OVERVIEW
# =========================
with tab1:
    st.markdown("<div class='section-header'>Global Overview</div>", unsafe_allow_html=True)

    map_metric = st.selectbox(
        "Select metric for choropleth map",
        ["Confirmed", "Recovered", "Deaths"],
        index=0,
        key="map_metric"
    )

    map_fig = px.choropleth(
        summary_df,
        locations="Country",
        locationmode="country names",
        color=map_metric,
        hover_name="Country",
        color_continuous_scale="Blues" if map_metric == "Confirmed" else ("Greens" if map_metric == "Recovered" else "Reds"),
        title=f"Global {map_metric} Cases by Country",
    )
    map_fig.update_layout(template="plotly_white", height=560, margin=dict(l=0, r=0, t=60, b=0))
    st.plotly_chart(map_fig, use_container_width=True)

    st.markdown("#### Top Countries")

    col_a, col_b = st.columns(2)

    with col_a:
        bar_metric = st.selectbox(
            "Select metric for top countries",
            ["Confirmed", "Recovered", "Deaths"],
            index=0,
            key="bar_metric"
        )

        top_df = top_countries(dff, bar_metric, top_n)
        bar_fig = px.bar(
            top_df.sort_values(by=bar_metric),
            x=bar_metric,
            y="Country",
            orientation="h",
            color=bar_metric,
            color_continuous_scale="Viridis",
            title=f"Top {top_n} Countries by {bar_metric}"
        )
        bar_fig.update_layout(
            template="plotly_white",
            height=520,
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    with col_b:
        timeline_metric = st.selectbox(
            "Select metric for time trend",
            ["Confirmed", "Recovered", "Deaths"],
            index=0,
            key="timeline_metric"
        )

        trend_df = dff.groupby("Date", as_index=False)[timeline_metric].sum()
        line_fig = px.line(
            trend_df,
            x="Date",
            y=timeline_metric,
            title=f"Global {timeline_metric} Trend Over Time",
            markers=True
        )
        line_fig.update_layout(template="plotly_white", height=520)
        st.plotly_chart(line_fig, use_container_width=True)

# =========================
# TAB 2: COUNTRY ANALYSIS
# =========================
with tab2:
    st.markdown("<div class='section-header'>Country Analysis</div>", unsafe_allow_html=True)

    available_countries = sorted(dff["Country"].unique().tolist())
    analysis_country_default = forecast_country if forecast_country in available_countries else available_countries[0]

    analysis_country = st.selectbox(
        "Select a country to analyze",
        options=available_countries,
        index=available_countries.index(analysis_country_default)
    )

    country_df = dff[dff["Country"] == analysis_country].copy()
    country_df = country_df.sort_values("Date")

    if country_df.empty:
        st.warning("No data available for this country.")
    else:
        latest_country = country_df.iloc[-1]

        k1, k2, k3 = st.columns(3)
        with k1:
            metric_card(f"{analysis_country} Confirmed", format_number(latest_country["Confirmed"]))
        with k2:
            metric_card(f"{analysis_country} Recovered", format_number(latest_country["Recovered"]))
        with k3:
            metric_card(f"{analysis_country} Deaths", format_number(latest_country["Deaths"]))

        metric_option = st.radio(
            "Select metric view",
            ["Confirmed", "Recovered", "Deaths", "All"],
            horizontal=True
        )

        if metric_option == "All":
            multi_fig = go.Figure()
            for col in ["Confirmed", "Recovered", "Deaths"]:
                multi_fig.add_trace(go.Scatter(
                    x=country_df["Date"],
                    y=country_df[col],
                    mode="lines",
                    name=col
                ))
            multi_fig.update_layout(
                title=f"{analysis_country}: COVID-19 Trend Comparison",
                xaxis_title="Date",
                yaxis_title="Cases",
                hovermode="x unified",
                template="plotly_white",
                height=550
            )
            st.plotly_chart(multi_fig, use_container_width=True)
        else:
            single_fig = px.area(
                country_df,
                x="Date",
                y=metric_option,
                title=f"{analysis_country}: {metric_option} Trend"
            )
            single_fig.update_layout(template="plotly_white", height=550)
            st.plotly_chart(single_fig, use_container_width=True)

        st.markdown("#### Daily Change Analysis")
        daily_metric = st.selectbox(
            "Select metric for daily change",
            ["Confirmed", "Recovered", "Deaths"],
            key="daily_metric"
        )

        daily_df = country_df[["Date", daily_metric]].copy()
        daily_df["Daily Change"] = daily_df[daily_metric].diff().fillna(0)

        daily_fig = px.bar(
            daily_df,
            x="Date",
            y="Daily Change",
            title=f"{analysis_country}: Daily Change in {daily_metric}"
        )
        daily_fig.update_layout(template="plotly_white", height=420)
        st.plotly_chart(daily_fig, use_container_width=True)

# =========================
# TAB 3: DATASET
# =========================
with tab3:
    st.markdown("<div class='section-header'>Filtered Dataset</div>", unsafe_allow_html=True)

    st.dataframe(dff, use_container_width=True, height=500)

    csv = dff.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_covid_data.csv",
        mime="text/csv"
    )

    st.markdown("#### Summary Table")
    summary_table = (
        dff.groupby("Country", as_index=False)[["Confirmed", "Recovered", "Deaths"]]
        .sum()
        .sort_values("Confirmed", ascending=False)
    )
    st.dataframe(summary_table, use_container_width=True, height=420)

# =========================
# TAB 4: FORECASTING
# =========================
with tab4:
    st.markdown("<div class='section-header'>Forecasting with Prophet</div>", unsafe_allow_html=True)
    st.caption("Forecasts are generated using historical country-level time series within the selected date range.")

    forecast_metrics = st.multiselect(
        "Select metrics to forecast",
        ["Confirmed", "Recovered", "Deaths"],
        default=["Confirmed", "Deaths", "Recovered"]
    )

    if not run_forecast:
        st.info("Choose forecasting settings from the sidebar and click 'Run Forecast'.")
    else:
        if forecast_country not in dff["Country"].unique():
            st.warning("The selected forecast country is not present in the filtered dataset. Adjust filters and try again.")
        else:
            period = forecast_years * 365

            for metric in forecast_metrics:
                with st.expander(f"Forecast: {metric} for {forecast_country}", expanded=(metric == forecast_metrics[0])):
                    history = prepare_country_metric_series(dff, forecast_country, metric)

                    if len(history) < 10:
                        st.warning(f"Not enough data to forecast {metric} for {forecast_country}.")
                        continue

                    hist_fig = px.line(
                        history,
                        x="ds",
                        y="y",
                        title=f"Historical {metric} Cases for {forecast_country}",
                        markers=True
                    )
                    hist_fig.update_layout(template="plotly_white", height=420)
                    st.plotly_chart(hist_fig, use_container_width=True)

                    with st.spinner(f"Training Prophet model for {metric}..."):
                        model = Prophet(
                            daily_seasonality=False,
                            weekly_seasonality=True,
                            yearly_seasonality=True
                        )
                        model.fit(history)

                        future = model.make_future_dataframe(periods=period)
                        forecast = model.predict(future)

                    forecast_fig = build_forecast_plot(
                        history=history,
                        forecast=forecast,
                        metric=metric,
                        country=forecast_country,
                        years=forecast_years
                    )
                    st.plotly_chart(forecast_fig, use_container_width=True)

                    latest_forecast = forecast.iloc[-1]
                    f1, f2, f3 = st.columns(3)
                    with f1:
                        metric_card("Predicted Value", format_number(max(latest_forecast["yhat"], 0)))
                    with f2:
                        metric_card("Upper Bound", format_number(max(latest_forecast["yhat_upper"], 0)))
                    with f3:
                        metric_card("Lower Bound", format_number(max(latest_forecast["yhat_lower"], 0)))

                    st.markdown("#### Forecast Components")
                    components_fig = model.plot_components(forecast)
                    st.pyplot(components_fig)

                    forecast_table = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(20).copy()
                    forecast_table.columns = ["Date", "Forecast", "Lower Bound", "Upper Bound"]
                    st.dataframe(forecast_table, use_container_width=True)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    """
    <div class="small-note">
        Built with Streamlit, Plotly, Pandas, and Prophet.<br>
        This dashboard is designed for interactive exploration and forecasting of country-level COVID-19 trends.
    </div>
    """,
    unsafe_allow_html=True
)
