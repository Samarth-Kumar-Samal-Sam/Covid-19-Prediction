import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
from prophet.plot import plot_plotly
from prophet import Prophet
import streamlit as st
import warnings
import os

warnings.filterwarnings('ignore')

# Streamlit Page Config
st.set_page_config(
    page_title='Covid-19 Prediction Application',
    page_icon='💻',
    layout='wide'
)

# Title and Description
st.title('Covid-19 Prediction Web Application 📊')

# Safe Image Loading
image_path = r'./Assets/covid-19-image.jpg'
if os.path.exists(image_path):
    st.image(image_path, caption='Covid-19', use_container_width=True)
else:
    st.warning(f"Image '{image_path}' not found. Please add it in the same folder as app.py.")

st.write('Developing a COVID-19 prediction web app with Prophet for forecasting, Plotly for interactive visualizations, and Streamlit for a user-friendly interface.')
st.markdown("<br>", unsafe_allow_html=True)

# Load Dataset
df = pd.read_csv('./Dataset/covid-19.csv')
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)
df['Date'] = pd.to_datetime(df['Date'])

# Date Picker Range
default_start_date = df['Date'].min()
default_end_date = df['Date'].max()

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Input Start Date", value=default_start_date)
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("Selected Start Date:", start_date)

with col2:
    end_date = st.date_input("Input End Date", value=default_end_date)
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("Selected End Date:", end_date)

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

if start_date < default_start_date:
    st.warning('Invalid Start Date')
elif end_date > default_end_date:
    st.warning('Invalid End Date')

st.markdown("<br>", unsafe_allow_html=True)

# Filtered Dataset
st.subheader('Filtered Dataset:')
dff = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
st.write(dff)

st.markdown("<br>", unsafe_allow_html=True)
st.subheader('World Covid Cases with respect to time')
st.markdown("<br>", unsafe_allow_html=True)

# Choropleth: Confirmed
st.write("### Confirmed Cases with respect to Time")
fig = px.choropleth(df, locations='Country', locationmode='country names', color='Confirmed',
                    animation_frame='Date', color_continuous_scale='RdBu')
fig.update_layout(title='Choropleth Map for Confirmed Covid-19 Cases')
fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 0.1
st.plotly_chart(fig)

# Choropleth: Recovered
st.markdown("<br>", unsafe_allow_html=True)
st.write("### Recovered Cases with respect to Time")
fig_recovered = px.choropleth(df, locations='Country', locationmode='country names', color='Recovered',
                              animation_frame='Date', color_continuous_scale='BuPu')
fig_recovered.update_layout(title='Choropleth Map for Recovered Covid-19 Cases')
fig_recovered.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 0.1
st.plotly_chart(fig_recovered)

# Choropleth: Deaths
st.markdown("<br>", unsafe_allow_html=True)
st.write("### Death Cases with respect to Time")
fig_deaths = px.choropleth(df, locations='Country', locationmode='country names', color='Deaths',
                           animation_frame='Date', color_continuous_scale='magma')
fig_deaths.update_layout(title='Choropleth Map for Death Covid-19 Cases')
fig_deaths.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 0.1
st.plotly_chart(fig_deaths)

st.markdown("<br>", unsafe_allow_html=True)
st.subheader('Top 5 countries for Confirmed, Deaths, and Recovered Covid-19 cases')
st.markdown("<br>", unsafe_allow_html=True)

# Top 5 by Deaths
st.write('**Top 5 countries with maximum number of Covid-19 Death Cases**')
deaths_top5 = dff.groupby('Country')['Deaths'].sum().sort_values(ascending=False).head(5)
filtered_data = dff[dff['Country'].isin(deaths_top5.index)]

plt.figure(figsize=(6,6))
sns.set_palette("viridis")
sns.barplot(x='Country', y='Deaths', data=filtered_data, estimator=sum)
plt.title('Top 5 Countries by Death Cases')
plt.tight_layout()
st.pyplot(plt)

# Top 5 by Confirmed
st.markdown("<br>", unsafe_allow_html=True)
st.write('**Top 5 countries with maximum number of Covid-19 Confirmed Cases**')
confirmed_top5 = dff.groupby('Country')['Confirmed'].sum().sort_values(ascending=False).head(5)
filtered_data = dff[dff['Country'].isin(confirmed_top5.index)]

plt.figure(figsize=(6,6))
sns.set_palette("magma")
sns.barplot(x='Country', y='Confirmed', data=filtered_data, estimator=sum)
plt.title('Top 5 Countries by Confirmed Cases')
plt.tight_layout()
st.pyplot(plt)

# Top 5 by Recovered
st.markdown("<br>", unsafe_allow_html=True)
st.write('**Top 5 countries with maximum number of Covid-19 Recovered Cases**')
recovered_top5 = dff.groupby('Country')['Recovered'].sum().sort_values(ascending=False).head(5)
filtered_data = dff[dff['Country'].isin(recovered_top5.index)]

plt.figure(figsize=(6,6))
sns.set_palette("deep")
sns.barplot(x='Country', y='Recovered', data=filtered_data, estimator=sum)
plt.title('Top 5 Countries by Recovered Cases')
plt.tight_layout()
st.pyplot(plt)

# Pie Charts for Top 10
st.markdown("<br>", unsafe_allow_html=True)
st.subheader('Top 10 most affected countries by Confirmed, Recovered, and Death Cases')

for metric in ['Deaths', 'Confirmed', 'Recovered']:
    top10 = dff.groupby('Country')[metric].sum().sort_values(ascending=False).head(10)
    filtered_data = df[df['Country'].isin(top10.index)]
    fig = px.pie(filtered_data, values=metric, names='Country',
                 title=f"Percentage of Total {metric} in Top 10 Affected Countries")
    st.plotly_chart(fig)

# Forecasting Section
st.markdown("<br>", unsafe_allow_html=True)
st.subheader('Forecasting the Covid-19 Cases')
st.markdown("<br>", unsafe_allow_html=True)

country = st.selectbox(label='Select the Country Name', options=dff['Country'].unique())
year = st.slider(label='Select the number of years', min_value=1, max_value=10)
period = year * 365

for metric in ['Confirmed', 'Deaths', 'Recovered']:
    with st.expander(f'Forecasting the {metric} Cases for {country}'):
        data = dff[dff['Country'] == country][['Date', metric]].rename(columns={'Date': 'ds', metric: 'y'})
        fig = px.line(data, x='ds', y='y', title=f"Time Series Graph for {metric} Cases")
        st.write(fig)
        model = Prophet()
        model.fit(data)
        future_pred = model.make_future_dataframe(periods=period, freq='D')
        prediction = model.predict(future_pred)
        st.write(prediction)
        st.write(model.plot_components(prediction))
        fig3 = plot_plotly(model, prediction, xlabel='Time', ylabel=f'{metric} cases')
        fig3.update_layout(title=f'Forecast graph for {metric} Cases')
        st.write(fig3)
