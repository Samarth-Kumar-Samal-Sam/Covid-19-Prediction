import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import warnings
from prophet import Prophet

warnings.filterwarnings('ignore')

# Streamlit Page Config
st.set_page_config(
    page_title='Covid-19 Prediction Application',
    page_icon='🦠',
    layout='wide'
)

# Title
st.title('Covid-19 Prediction Web Application 🦠')

# Load Dataset
df = pd.read_csv('./Dataset/covid-19.csv')
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)
df['Date'] = pd.to_datetime(df['Date'])

# Default date range
default_start_date = pd.to_datetime('2020-01-01')
default_end_date = pd.to_datetime('2022-05-01')

# Date input widgets
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input(
        "Input Start Date",
        value=default_start_date,
        min_value=default_start_date,
        max_value=default_end_date
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("Selected Start Date:", start_date)

with col2:
    end_date = st.date_input(
        "Input End Date",
        value=default_end_date,
        min_value=default_start_date,
        max_value=default_end_date
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("Selected End Date:", end_date)

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Validate dates
if start_date < default_start_date:
    st.warning('Invalid Start Date')
elif end_date > default_end_date:
    st.warning('Invalid End Date')

# Reset predict_clicked if dates changed
if ('last_start_date' not in st.session_state or 'last_end_date' not in st.session_state) \
   or (st.session_state.last_start_date != start_date) \
   or (st.session_state.last_end_date != end_date):
    st.session_state.predict_clicked = False
    st.session_state.forecast_clicked = False

st.session_state.last_start_date = start_date
st.session_state.last_end_date = end_date

if 'predict_clicked' not in st.session_state:
    st.session_state.predict_clicked = False

if 'forecast_clicked' not in st.session_state:
    st.session_state.forecast_clicked = False

# Predict button for filtering and showing maps/bar charts
if st.button("Predict"):
    st.session_state.predict_clicked = True
    st.session_state.forecast_clicked = False  # reset forecasting on new filter

def show_predictions(dff):
    # Show filtered dataset
    st.subheader('Filtered Dataset:')
    st.write(dff)

    # Download CSV
    csv = dff.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name='filtered_covid_data.csv',
        mime='text/csv'
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Choropleth maps in dropdown inside expander
    with st.expander("World Covid Cases with Respect to Time"):
        st.write("### Select Map Type")
        map_type = st.selectbox(
            "Choose the Covid case type to display on the map:",
            ['Confirmed', 'Recovered', 'Deaths'],
            key="map_type"
        )

        color_scales = {
            'Confirmed': 'RdBu',
            'Recovered': 'BuPu',
            'Deaths': 'magma'
        }

        fig = px.choropleth(
            dff,
            locations='Country',
            locationmode='country names',
            color=map_type,
            animation_frame=dff['Date'].dt.strftime('%Y-%m-%d'),
            color_continuous_scale=color_scales[map_type],
            title=f"{map_type} Cases Over Time"
        )
        fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
        fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 100
        st.plotly_chart(fig, use_container_width=True)

    # Top 5 bar charts in dropdown inside expander
    with st.expander("Top 5 Countries by Covid-19 Cases"):
        bar_metric = st.selectbox(
            "Select metric for Top 5 Countries Bar Chart:",
            ['Confirmed', 'Recovered', 'Deaths'],
            key="bar_metric"
        )
        # Calculate top 5 countries
        top5 = dff.groupby('Country')[bar_metric].sum().sort_values(ascending=False).head(5).reset_index()

        # Create Plotly horizontal bar chart
        fig_bar = px.bar(
            top5.sort_values(by=bar_metric),  # Sort ascending for horizontal bars
            x=bar_metric,
            y='Country',
            orientation='h',
            color=bar_metric,
            color_continuous_scale='Viridis',
            title=f"Top 5 Countries by {bar_metric} Cases"
        )
        fig_bar.update_layout(yaxis=dict(autorange="reversed"))  # So highest is on top
        st.plotly_chart(fig_bar, use_container_width=True)

# Display filtered data, maps, bar charts after Predict clicked
if st.session_state.predict_clicked:
    dff = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
    show_predictions(dff)

    # Forecasting Section
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader('Forecasting the Covid-19 Cases')
    st.markdown("<br>", unsafe_allow_html=True)

    country = st.selectbox(label='Select the Country Name', options=sorted(dff['Country'].unique()), key="forecast_country")
    year = st.slider(label='Select the number of years to forecast', min_value=1, max_value=10, key="forecast_years")

    # Forecast Predict button
    if st.button("Run Forecast"):
        st.session_state.forecast_clicked = True

    if st.session_state.forecast_clicked:
        period = year * 365
        for metric in ['Confirmed', 'Deaths', 'Recovered']:
            with st.expander(f'Forecasting the {metric} Cases for {country}'):
                # Prepare data for Prophet
                data = dff[dff['Country'] == country][['Date', metric]].rename(columns={'Date': 'ds', metric: 'y'})

                # Check if enough data to forecast
                if len(data) < 10:
                    st.warning(f"Not enough data to forecast {metric} cases for {country}.")
                    continue

                # Plot original time series
                fig_hist = px.line(data, x='ds', y='y', title=f"Historical {metric} Cases")
                st.plotly_chart(fig_hist, use_container_width=True)

                # Fit Prophet model
                model = Prophet()
                model.fit(data)

                # Create future dataframe
                future = model.make_future_dataframe(periods=period)

                # Forecast
                forecast = model.predict(future)

                # Prepare forecast plot with confidence intervals
                fig_forecast = go.Figure()

                # Historical data
                fig_forecast.add_trace(go.Scatter(
                    x=data['ds'], y=data['y'],
                    mode='lines',
                    name='Historical'
                ))

                # Forecasted mean
                fig_forecast.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat'],
                    mode='lines',
                    name='Forecast'
                ))

                # Confidence interval
                fig_forecast.add_trace(go.Scatter(
                    x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
                    y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
                    fill='toself',
                    fillcolor='rgba(0,100,80,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=True,
                    name='Confidence Interval'
                ))

                fig_forecast.update_layout(
                    title=f'{metric} Cases Forecast for {country} ({year} year(s))',
                    xaxis_title='Date',
                    yaxis_title=f'{metric} Cases',
                    legend=dict(y=0.99, x=0.01),
                    hovermode="x unified"
                )
                st.plotly_chart(fig_forecast, use_container_width=True)

                # Prophet components plot (matplotlib)
                components_fig = model.plot_components(forecast)
                st.pyplot(components_fig)
