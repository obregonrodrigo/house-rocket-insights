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


def date_season(date):
    year = str(date.year)
    seasons = {'spring': pd.date_range(start='21/03/' + year, end='20/06/' + year),
               'summer': pd.date_range(start='21/06/' + year, end='22/09/' + year),
               'fall': pd.date_range(start='23/09/' + year, end='20/12/' + year)}
    if date in seasons['spring']:
        return 'spring'
    if date in seasons['summer']:
        return 'summer'
    if date in seasons['fall']:
        return 'fall'
    else:
        return 'winter'

def set_feature(data):
    #preço por m2
    data['price_m2'] = data['price'] / data['sqft_lot']

    # média de preço por zipcode
    mz = data[['price', 'zipcode']].groupby('zipcode').median().reset_index().rename(columns={'price': 'median_price'})
    data = pd.merge(data, mz, on='zipcode', how='left')

    # decisão de compra
    data['decision'] = data[['price', 'median_price', 'condition']].apply(
        lambda x: 1 if ((x['price'] <= x['median_price']) & (x['condition'] >= 3)) else 0, axis=1)

    # sugestão de preço de venda
    data['selling_suggestion'] = data[['price', 'median_price', 'condition']].apply(lambda x: x['price'] * 1.25
    if ((x['price'] <= x['median_price']) & (x['condition'] >= 3)) else 0, axis=1)

    # retorno esperado
    data['expected_profit'] = data[['price', 'selling_suggestion']].apply(lambda x: 0 if x['selling_suggestion'] == 0
    else (x['selling_suggestion'] - x['price']), axis=1)

    # melhor estação do ano para venda
    data['season_sell'] = ''
    for i in range(len(data)):
        cols = ['med_fall', 'med_spring', 'med_summer', 'med_winter']
        if data.loc[i, 'decision'] != 0:
            if data.loc[i, cols[0]] >= data.loc[i, 'price']:
                data.loc[i, 'season_sell'] = data.loc[i, 'season_sell'] + 'autumn '
            if data.loc[i, cols[1]] >= data.loc[i, 'price']:
                data.loc[i, 'season_sell'] = data.loc[i, 'season_sell'] + 'spring '
            if data.loc[i, cols[2]] >= data.loc[i, 'price']:
                data.loc[i, 'season_sell'] = data.loc[i, 'season_sell'] + 'summer '
            if data.loc[i, cols[3]] >= data.loc[i, 'price']:
                data.loc[i, 'season_sell'] = data.loc[i, 'season_sell'] + 'winter '

    # exclui duplicados e imóveis que não sejam qualificados para compra
    # data = data[data['decision'] != 0].copy()
    data = data.sort_values('date', ascending=True)
    data = data.drop_duplicates(subset='id', keep='last').copy()

    return data

def data_transform(data):

    # criação da variável dormitory_type
    data['dormitory_type'] = 'NA'
    for i in range(len(data)):
        if data.loc[i, 'bedrooms'] == 1:
            data.loc[i, 'dormitory_type'] = 'studio'
        if data.loc[i, 'bedrooms'] == 2:
            data.loc[i, 'dormitory_type'] = 'apartment'
        else:
            data.loc[i, 'dormitory_type'] = 'house'

    # estações do ano
    data['season'] = data['date'].map(date_season)

    # agrupamento por zipcode e média de preço por estação do ano
    aux = data[['price', 'zipcode', 'season']].groupby(['zipcode', 'season']).median().reset_index()
    aux1 = aux.pivot(index='zipcode', columns='season', values='price').reset_index()
    aux1 = aux1.rename(
        columns={'fall': 'med_fall', 'spring': 'med_spring', 'summer': 'med_summer', 'winter': 'med_winter'})
    data = pd.merge(data, aux1, on='zipcode', how='left')

    return data


def overview_data(data):
    f_attributes = st.sidebar.multiselect('Selecione as colunas', data.columns)
    f_zipcode = st.sidebar.multiselect('Selecione os zipcodes', data['zipcode'].unique())

    st.title('Visão Geral')
    # select columns and lines

    if (f_zipcode != []) & (f_attributes != []):
        df0 = data.loc[data['zipcode'].isin(f_zipcode), f_attributes]
    elif (f_zipcode != []) & (f_attributes == []):
        df0 = data.loc[data['zipcode'].isin(f_zipcode), :]
    elif (f_zipcode == []) & (f_attributes != []):
        df0 = data.loc[:, f_attributes]
    else:
        df0 = data.copy()

    st.dataframe(df0)

    # average metrics
    df1 = data[['id', 'zipcode']].groupby('zipcode').count().reset_index()
    df2 = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df3 = data[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
    df4 = data[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()

    c1, c2 = st.beta_columns((1, 1))

    # merge
    m1 = pd.merge(df1, df2, on='zipcode', how='inner')
    m2 = pd.merge(m1, df3, on='zipcode', how='inner')
    df = pd.merge(m2, df4, on='zipcode', how='inner')

    df.columns = ['ZIPCODE', 'TOTAL HOUSES', 'PRICE', 'SQRT LIVING', 'PRICE/M2']

    c1.header('Média de valores')
    c1.dataframe(df, width=500, height=300)

    # statistic descriptive
    num_attributes = data.select_dtypes(include=['int64', 'float64'])
    media = pd.DataFrame(num_attributes.apply(np.mean))
    mediana = pd.DataFrame(num_attributes.apply(np.median))
    std = pd.DataFrame(num_attributes.apply(np.std))

    max_ = pd.DataFrame(num_attributes.apply(np.max))
    min_ = pd.DataFrame(num_attributes.apply(np.min))

    df1 = pd.concat([max_, min_, media, mediana, std], axis=1).reset_index()
    df1.columns = ['attirbutes', 'max', 'min', 'mean', 'median', 'std']

    c2.header('Análise descritiva')
    c2.dataframe(df1, width=500, height=300)

    return None

def portifolio_density(data, geofile):
    st.title('Visão Geral da Região - Imóveis Qualificados para Compra')

    c1, c2 = st.beta_columns((1, 1))
    c1.header('Densidade do Portifólio')

    df = data[data['decision'] != 0].copy()
    #df = data.sample(100)

    # Base MAP - Folium
    density_map = folium.Map(location=[data['lat'].mean(), data['long'].mean()], default_zoom_start=15)

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

    st.header('Preço médio por ano de contrução - Imóveis qualificados para compra')

    # data selection
    df = data[data['decision'] != 0].copy()
    df = data.loc[data['yr_built'] < f_year_built]
    df = df[['yr_built', 'price']].groupby('yr_built').mean().reset_index()

    # plot
    fig = px.line(df, x='yr_built', y='price')
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.beta_columns((1, 1))

    # Average price per level
    c1.header('Preço médio por nivel')
    df = data[['level', 'price']].groupby('level').mean().reset_index()
    # data plot
    fig = px.histogram(df, x='level', y='price', nbins=7)
    c1.plotly_chart(fig, use_container_width=True)

    ## Average price per type
    #c2.header('Preço médio por condição estrutural')
    #df = data[['condition_type', 'price']].groupby('condition_type').mean().reset_index()
    # data plot
    ##fig = px.histogram(df, x='condition_type', y='price', nbins=8)
    #c2.plotly_chart(fig, use_container_width=True)

    # Média de valor de venda e lucro por tipo de imóvel
    c2.header('Valor médio de venda e lucro')
    df = data[['dormitory_type', 'selling_suggestion']].groupby('dormitory_type').mean().reset_index()
    df1 = data[['dormitory_type', 'expected_profit']].groupby('expected_profit').mean().reset_index()
    # data plot
    width = 0.40
    fig = px.histogram(df-0.2, x='dormitory_type', y='selling_suggestion', width, color='blue', nbins=8)
    fig = px.histogram(df1+0.2, x='dormitory_type', y='expected_profit', width, color='green', nbins=8)
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
    df = data[data['bedrooms'] <= f_bedrooms]
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

st.markdown('---')

st.title('Informações adicionais')
st.write('')
st.markdown('Esses dashboard parte do projeto house rocket insights. Desenvolvido por Rodrigo Obregon')
st.markdown('For more information about the business context and checking the code go to the project repository on '
            'github: [Github](https://github.com/obregonrodrigo/house-rocket-insights)')
st.markdown('Outros projetos: [Portfolio](https://github.com/obregonrodrigo)')
st.markdown('Contato: [LinkedIn](https://www.linkedin.com/in/rodrigobregon/)')

st.markdown('---')