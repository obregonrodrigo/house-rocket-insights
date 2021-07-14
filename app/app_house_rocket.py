import numpy as np
import streamlit as st
import pandas as pd
import folium
import geopandas
import plotly.express as px

from datetime import datetime
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster

st.set_page_config(layout='wide')


@st.cache(allow_output_mutation=True)
def get_data(path):
    data = pd.read_csv(path)
    return data


@st.cache(allow_output_mutation=True)
def get_geofile(url):
    geofile = geopandas.read_file(url)
    return geofile


# get data
path = "C:/Users/Rodrigo/Repos/house-rocket-insights/data/kc_house_data.csv"
data = get_data(path)

# get geofile
url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
geofile = get_geofile(url)

# add new feature
data['price_m2'] = data['price'] / data['sqft_lot']

##
# data overview
# filtro atributos
f_attributes = st.sidebar.multiselect('Enter columns', data.columns)

# filtro zipcode
f_zipcode = st.sidebar.multiselect('Enter zipcode', data['zipcode'].unique())

st.title('Data Overview')

if (f_zipcode != []) & (f_attributes != []):
    data = data.loc[data['zipcode'].isin(f_zipcode), f_attributes]

elif (f_zipcode != []) & (f_attributes == []):
    data = data.loc[data['zipcode'].isin(f_zipcode), :]

elif (f_zipcode == []) & (f_attributes != []):
    data = data.loc[:, f_attributes]

else:
    data = data.copy()

st.dataframe(data)

c1, c2 = st.beta_columns((1, 1))  # controla dimensão colunas

# Average Metrics
df1 = data[['id', 'zipcode']].groupby('zipcode').count().reset_index()
df2 = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
df3 = data[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
df4 = data[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()

# merge
m1 = pd.merge(df1, df2, on='zipcode', how='inner')
m2 = pd.merge(m1, df3, on='zipcode', how='inner')
df = pd.merge(m2, df4, on='zipcode', how='inner')

df.columns = ['ZIPCODE', 'TOTAL HOUSES', 'PRICE', 'SQRT LIVING', 'PRICE/M2']

c1.header('Average Values')
c1.dataframe(df, height=500)

# Statistic Descriptive
num_attributes = data.select_dtypes(include=['int64', 'float64'])

media = pd.DataFrame(num_attributes.apply(np.mean))
mediana = pd.DataFrame(num_attributes.apply(np.median))
std = pd.DataFrame(num_attributes.apply(np.std))

max_ = pd.DataFrame(num_attributes.apply(np.max))
min_ = pd.DataFrame(num_attributes.apply(np.min))

df1 = pd.concat([max_, min_, media, mediana, std], axis=1).reset_index()
df1.columns = ['attributes', 'max', 'min', 'mean', 'median', 'std']

c2.header('Descriptive Analysis')
c2.dataframe(df1, height=500)

##
# densidade de portifolio
st.title('Region Overview')

c1, c2 = st.beta_columns((1, 1))
c1.header('Portfolio Density')

df = data.sample(100)

# Base MAP - Folium
density_map = folium.Map(location=[data['lat'].mean(), data['long'].mean()],
                         default_zoom_start=15)

marker_cluster = MarkerCluster().add_to(density_map)
for name, row in df.iterrows():
    folium.Marker([row['lat'], row['long']],
                  popup='Sold R$ {0} on: {1}. Features {2} sqft, {3} bedrooms, {4} bathrooms,'
                        'year built: {5}'.format(row['price'], row['date'], row['sqft_living'],
                         row['bedrooms'], row['bathrooms'], row['yr_built'])).add_to(marker_cluster)

with c1:
    folium_static(density_map)

# Region Price Map
c2.header('Price Density')
df = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
df.columns = ['ZIP', 'PRICE']

#df = df.sample(10)

geofile = geofile[geofile['ZIP'].isin(df['ZIP'].tolist())]

region_price_map = folium.Map(location=[data['lat'].mean(),
                              data['long'].mean()],
                              default_zoom_start=15)

region_price_map.choropleth(data = df,
                            geo_data = geofile,
                            columns=['ZIP', 'PRICE'],
                            key_on='feature.properties.ZIP',
                            fill_color='YlOrRd',
                            fill_opacity=0.7,
                            line_opacity=0.2,
                            legend_name='AVG PRICE')

with c2:
    folium_static(region_price_map)

################################################
# distribuição dos imóveis por categorias comerciais
st.sidebar.title('Commercial Options')
st.title('Comercial Attrbitues')

# Average price per year
data['date'] = pd.to_datetime(data['date']).dt.strftime('%y-%m-%d')

#filters
min_year_built = int(data['yr_built'].min())
max_year_built = int(data['yr_built'].max())

st.sidebar.subheader('Select Max year built')
f_year_built = st.sidebar.slider('Year Built', min_year_built, max_year_built, max_year_built)

st.header('Average price per year built')
# data selection
df = data.loc[data['yr_built'] < f_year_built]
df = df[['yr_built', 'price']].groupby('yr_built').mean().reset_index()

# plot
fig = px.line(df, x='yr_built', y='price')
st.plotly_chart(fig, use_container_width=True)

# Average price per day
st.header('Average price per day')

#filters
min_date = datetime.strptime(data['date'].min(), '%y-%m-%d')
max_date = datetime.strptime(data['date'].max(), '%y-%m-%d')
st.sidebar.subheader('Select max Date')
f_date = st.sidebar.slider('Date', min_date, max_date, max_date)


data['date'] = pd.to_datetime(data['date']).dt.strftime('%y-%m-%d')
data['date'] = pd.to_datetime(data['date'])

df = data.loc[data['date'] < f_date]
df = data[['date', 'price']].groupby('date').mean().reset_index()

# plot
fig = px.line(df, x='date', y='price')
st.plotly_chart(fig, use_container_width=True)

#### histograma
st.header('Price Distribuition')
st.sidebar.subheader('Select Max Price')

#filter
min_price = int(data['price'].min())
max_price = int(data['price'].max())
avg_price = int(data['price'].mean())

#data filtering
f_price = st.sidebar.slider('price', min_price, max_price, avg_price)
df = data.loc[data['price'] < f_price]

# data plot
fig = px.histogram(df, x='price', nbins=50)
st.plotly_chart(fig, use_container_width=True)

# Distribuição dos imoveis por categoria
st.sidebar.title('Attributes Options')
st.title('House Attributes')

# filters
f_bedrooms = st.sidebar.selectbox('Max number of bedrooms', sorted(set(data['bedrooms'].unique())))
f_bathrooms = st.sidebar.selectbox('Max number of bathrooms', sorted(set(data['bathrooms'].unique())))

c1, c2 = st.beta_columns(2)

# house per bedrooms
c1.header('Houses per Bedrooms')
df = data[data['bedrooms'] < f_bedrooms]
fig = px.histogram(df, x='bedrooms', nbins=19)
c1.plotly_chart(fig, use_container_width=True)

# house per bathrooms
c2.header('Houses per bathrooms')
df = data[data['bathrooms'] < f_bathrooms]
fig = px.histogram(df, x='bathrooms', nbins=19)
c2.plotly_chart(fig, use_container_width=True)


#filter
f_floors = st.sidebar.selectbox('Max number of floor', sorted(set(data['floors'].unique())))
f_wahterview = st.sidebar.checkbox('Only Houses with water view')

c1, c2 = st.beta_columns(2)

# house per floors
c1.header('Houses per floor')
df = data[data['floors'] < f_floors]
fig = px.histogram(df, x='floors', nbins=19)
c1.plotly_chart(fig, use_container_width=True)

# house per water view
c2.header('Waterfront view')
if f_wahterview:
    df = data[data['waterfront'] == 1]
else:
    df = data.copy()

fig = px.histogram(df, x='waterfront', nbins=10)
c2.plotly_chart(fig, use_container_width=True)

################################################ textt
st.header('Average price per day')
st.sidebar.subheader('Select max Date')

data['date'] = pd.to_datetime(data['date']).dt.strftime('%y-%m-%d')
today = datetime.strptime(data['date'].min(), '%y-%m-%d')
tomorrow = datetime.strptime(data['date'].max(), '%y-%m-%d')
start_date = st.sidebar.date_input('Start date', today)
end_date = st.sidebar.date_input('End date', tomorrow)

df = pd.DataFrame({"start date": start_date, "end date":end_date, "value":[11,3,11,4]})
if start_date < end_date:
    st.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
else:
    st.error('Error: End date must fall after start date.')

