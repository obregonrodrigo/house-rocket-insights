import folium
import geopandas

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from datetime import datetime
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static

st.set_page_config(layout='wide')

@st.cache(allow_output_mutation=True)
def get_data(path):
    data = pd.read_csv(path)

    return data

@st.cache(allow_output_mutation=True)
def get_geofile(url):
    geofile = geopandas.read_file(url)

    return geofile

def set_feature(data):
    data['price_m2'] = data['price'] / data['sqft_lot']
    # data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
    return data

def data_transform(data):
    data['level'] = 'NA'
    for i in range(len(data)):
        if data.loc[i, 'price'] < 321950:
            data.loc[i, 'level'] = 1
        elif (data.loc[i, 'price'] > 321950) & (data.loc[i, 'price'] <= 450000):
            data.loc[i, 'level'] = 2
        elif (data.loc[i, 'price'] > 450000) & (data.loc[i, 'price'] < 645000):
            data.loc[i, 'level'] = 3
        else:
            data.loc[i, 'level'] = 4

    data['dormitory_type'] = 'NA'

    for i in range(len(data)):
        if data.loc[i, 'bedrooms'] == 1:
            data.loc[i, 'dormitory_type'] = 'studio'

        if data.loc[i, 'bedrooms'] == 2:
            data.loc[i, 'dormitory_type'] = 'apartment'

        else:
            data.loc[i, 'dormitory_type'] = 'house'

    data['condition_type'] = 'NA'
    data.loc[data['condition'] <= 2, 'condition_type'] = 'bad'
    data.loc[(data['condition'] > 2) & (data['condition'] < 5), 'condition_type'] = 'regular'
    data.loc[data['condition'] >= 5, 'condition_type'] = 'good'

    return data


def overview_data(data):
    f_attributes = st.sidebar.multiselect('Selecione as colunas', data.columns)
    f_zipcode = st.sidebar.multiselect('Selecione o zipcode', data['zipcode'].unique())

    st.title('Visão geral dos dados')

    if (f_zipcode != []) & (f_attributes != []):
        data = data.loc[data['zipcode'].isin(f_zipcode), f_attributes]

    elif (f_zipcode != []) & (f_attributes == []):
        data = data.loc[data['zipcode'].isin(f_zipcode), :]

    elif (f_zipcode == []) & (f_attributes != []):
        data = data.loc[:, f_attributes]

    else:
        data = data.copy()

    st.dataframe(data.head())

    c1, c2 = st.beta_columns((1, 1))

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

    c1.header('Médias')
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

    c2.header('Análise Descritiva')
    c2.dataframe(df1, height=500)

    return None

def portifolio_density(data, geofile):
    st.title('Visão Geral da Região')

    c1, c2 = st.beta_columns((1, 1))
    c1.header('Densidade do Portifólio')

    df = data.sample(100)

    # Base MAP - Folium
    density_map = folium.Map(location=[data['lat'].mean(), data['long'].mean()],
                             default_zoom_start=15)

    marker_cluster = MarkerCluster().add_to(density_map)
    for name, row in df.iterrows():
        folium.Marker([row['lat'], row['long']],
                      popup='Sold R$ {0} on: {1}. Features {2} sqft, {3} bedrooms, {4} bathrooms,'
                            'year built: {5}'.format(row['price'], row['date'], row['sqft_living'],
                                                     row['bedrooms'], row['bathrooms'], row['yr_built'])).add_to(
                                                     marker_cluster)

    with c1:
        folium_static(density_map)

    # Region Price Map
    c2.header('Densidade do Preço')
    df = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df.columns = ['ZIP', 'PRICE']

    # df = df.sample(10)

    geofile = geofile[geofile['ZIP'].isin(df['ZIP'].tolist())]

    region_price_map = folium.Map(location=[data['lat'].mean(), data['long'].mean()],
                                  default_zoom_start=15)

    region_price_map.choropleth(data=df,
                                geo_data=geofile,
                                columns=['ZIP', 'PRICE'],
                                key_on='feature.properties.ZIP',
                                fill_color='YlOrRd',
                                fill_opacity=0.7,
                                line_opacity=0.2,
                                legend_name='AVG PRICE')

    with c2:
        folium_static(region_price_map)

    return None

def comercial_distribution(data):
    st.sidebar.title('Opções Comerciais')
    st.title('Atributos Comerciais')

    # Average price per year
    data['date'] = pd.to_datetime(data['date']).dt.strftime('%y-%m-%d')

    # filters
    min_year_built = int(data['yr_built'].min())
    max_year_built = int(data['yr_built'].max())

    st.sidebar.subheader('Ano máximo de Construção')
    f_year_built = st.sidebar.slider('Ano de Contrução', min_year_built, max_year_built, max_year_built)

    st.header('Preço médio por ano de contrução')

    # data selection
    df = data.loc[data['yr_built'] < f_year_built]
    df = df[['yr_built', 'price']].groupby('yr_built').mean().reset_index()

    # plot
    fig = px.line(df, x='yr_built', y='price')
    st.plotly_chart(fig, use_container_width=True)

    # Average pricer per level
    st.header('Preço médio por nivel')
    df = data[['level', 'price']].groupby('level').mean().reset_index()
    # data plot
    fig = px.histogram(df, x='level', y='price', nbins=4)
    st.plotly_chart(fig, use_container_width=True)

    # Average price per day
    #st.header('Média de preço por dia')
    #st.sidebar.subheader('Selecione a data máxima')

    # filters
    #min_date = datetime.strptime(data['date'].min(), '%y-%m-%d')
    #max_date = datetime.strptime(data['date'].max(), '%y-%m-%d')

    #f_date = st.sidebar.slider('Date', min_date, max_date, min_date)

    #data['date'] = pd.to_datetime(data['date']).dt.strftime('%y-%m-%d')
    #data['date'] = pd.to_datetime(data['date'])
    #st.write(data.dtypes)
    #data['date'] = data['date'].astype('datetime64')
    #df = data.loc[data['date'] < f_date]
    #df = df[['date', 'price']].groupby('date').mean().reset_index()

    # plot
    #fig = px.line(df, x='date', y='price')
    #st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.beta_columns((1, 1))

    # Average price per level
    c1.header('Preço médio por nivel')
    df = data[['level', 'price']].groupby('level').mean().reset_index()
    # data plot
    fig = px.histogram(df, x='level', y='price', nbins=4)
    c1.plotly_chart(fig, use_container_width=True)

    # Average price per type
    c2.header('Preço médio por tipo de imóvel')
    df = data[['condition_type', 'price']].groupby('condition_type').mean().reset_index()
    # data plot
    fig = px.histogram(df, x='condition_type', y='price', nbins=4)
    c2.plotly_chart(fig, use_container_width=True)

    #### histograma
    st.header('Distribuição de preço')
    st.sidebar.subheader('Selecione o preço máximo')

    # filter
    min_price = int(data['price'].min())
    max_price = int(data['price'].max())
    avg_price = int(data['price'].mean())

    # data filtering
    f_price = st.sidebar.slider('preço', min_price, max_price, avg_price)
    df = data.loc[data['price'] < f_price]

    # data plot
    fig = px.histogram(df, x='price', nbins=50)
    st.plotly_chart(fig, use_container_width=True)

    return None

def attributes_distribution(data):
    st.sidebar.title('Attributos Opicionais')
    st.title('Atributo das Casas')

    # filters
    f_bedrooms = st.sidebar.selectbox('Número Minimo de Quartos', sorted(set(data['bedrooms'].unique())))
    f_bathrooms = st.sidebar.selectbox('Número Máximo de Quartos', sorted(set(data['bathrooms'].unique())))

    c1, c2 = st.beta_columns(2)

    # house per bedrooms
    c1.header('Casas por número de quartos')
    df = data[data['bedrooms'] < f_bedrooms]
    fig = px.histogram(df, x='bedrooms', nbins=19)
    c1.plotly_chart(fig, use_container_width=True)

    # house per bathrooms
    c2.header('Casas por número de banheiros')
    df = data[data['bathrooms'] < f_bathrooms]
    # fig = px.histogram(data, x='bathrooms', nbins=19)
    fig = px.histogram(df, x='bathrooms', nbins=19)
    c2.plotly_chart(fig, use_container_width=True)

    # filter
    f_floors = st.sidebar.selectbox('Número máximo de andar', sorted(set(data['floors'].unique())))
    f_waterview = st.sidebar.checkbox('Apenas casas com vista para a água')

    c1, c2 = st.beta_columns(2)

    # house per floors
    c1.header('Residências por andar')
    df = data[data['floors'] < f_floors]

    #plot
    fig = px.histogram(df, x='floors', nbins=19)
    c1.plotly_chart(fig, use_container_width=True)

    # house per water view
    c2.header('Vista para Água')
    if f_waterview:
        df = data[data['waterfront'] == 1]
    else:
        df = data.copy()

    fig = px.histogram(df, x='waterfront', nbins=10)
    c2.plotly_chart(fig, use_container_width=True)

    return None


if __name__ == '__main__':
    # etl
    # data extraction
    path = "C:/Users/Rodrigo/Repos/house-rocket-insights/data/kc_house_data.csv"
    url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'

    data = get_data(path)
    geofile = get_geofile(url)

    #transformation
    data = data_transform(data)

    data = set_feature(data)

    overview_data(data)

    portifolio_density(data, geofile)

    comercial_distribution(data)

    attributes_distribution(data)