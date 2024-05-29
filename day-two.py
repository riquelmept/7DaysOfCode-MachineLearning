#Importando as bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpt

#Importando DataFrame
spotify_df = pd.read_csv('https://raw.githubusercontent.com/letpires/7DaysOfCodeSpotifyML/main/dataset.csv', index_col=0)
spotify_df.head()

#Verificando os dados (quantidade de linhas, colunas e tipo de cada coluna)

spotify_df.info()

#Buscando a descrição dos atributos numéricos

spotify_df.describe()

#Visualizando dimensões do DF

spotify_df.shape

#Visualizando nome das colunas do DF

spotify_df.columns

#Visualizando maiores durações e menores durações de músicas

spotify_df.duration_ms

#Validando quantidade única de Artista e Músicas x Artista

print(spotify_df['artists'].unique().shape)
print("-"*500)
print(spotify_df['artists'].value_counts())

#Ordenando em ordem decrescente por valor ausente

(spotify_df.isnull().sum() / spotify_df.shape[0]).sort_values(ascending=False)

#Verificando quantidade de valores ausentes

spotify_df.isnull().sum()

#função que gera uma gráfico de barras com colunas e frequência dos dados faltantes.
def missing_visualization(df):
  quant_isnull = df.isnull().sum()
  columns = df.columns
  dic = {"colunas":[],"quant_isnull":[]}
  for coluna,quant in zip(columns,quant_isnull):
    if quant > 0:
      dic["colunas"].append(quant)
      dic["quant_isnull"].append(coluna)
  df = pd.DataFrame(dic)
  plt.figure(figsize=(15,5))
  sns.barplot(x=df["quant_isnull"],y=df["colunas"],data=df, palette="rocket")
  plt.xticks(rotation=45);
  
missing_visualization(spotify_df)

#Quais são as 100 músicas mais populares?
sorted_df = spotify_df.sort_values('popularity', ascending = False).head(100)
sorted_df.head()

#Quais são os artistas mais populares?
artistas_popularidade = spotify_df[['artists', 'popularity']]
artistas_populares = artistas_popularidade.groupby("artists").mean().sort_values(by='popularity', ascending=False).reset_index()

#Trazendo somente os 5 primeiros
artistas_populares = artistas_populares.head()
artistas_populares

#Análise visual de popularidade dos artistas

artistas_populares.plot.barh(color="hotpink") ##visualize the data
plt.title("TOP 5 Most Popular Artists")
plt.show()

#Dentre os gêneros musicais, quais são os mais populares?

trend_genre = spotify_df[["track_genre", "popularity"]].sort_values(by="popularity", ascending=False)[:5]
trend_genre

#Análise visual dentre os gêneros mais populares

sns.barplot(x="track_genre",y="popularity", data=trend_genre, color = 'hotpink')
plt.title("Top trending genre")
plt.show()

#Analisando as músicas mais longas

long_songs = spotify_df[["track_name", "duration_ms"]].sort_values(by="duration_ms", ascending=False)[:5]
long_songs

#Análise visual das músicas mais longas comparando sua dimensão em milissegundos

sns.barplot(x="duration_ms", y="track_name", data= long_songs, color = 'hotpink')
plt.title("Top 5 Longest songs")
plt.show()

#Analisando as músicas mais "dançáveis" de acordo com o DF

danceable = spotify_df[["track_name", "artists", "danceability"]].sort_values(by="danceability", ascending=False)[:5]
danceable 

#Análise visual das músicas por danceability

colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99', '#ffb6c1']

plt.pie(x="danceability", data=danceable, autopct='%1.2f%%', labels=danceable.track_name, colors = colors)
plt.title("Top 5 Most Danceable Songs")
plt.show()

spotify_df.describe()

corr_table = spotify_df.select_dtypes(include=[np.number]).corr(method="pearson")
corr_table

#Análise Visual da correlação

#Plotando a tabela de correlação usando o Seaborn.
plt.figure(figsize=(16,4))
sns.heatmap(corr_table, annot=True, fmt=".1g")
plt.title("Correlation Heatmap between variables")
plt.show() #mostrando o plot

#Sequência Dia2

#Removendo itens duplicados

spotify_df = spotify_df.drop_duplicates()
spotify_df.head()

# Criando classes de popularidade
spotify_df[spotify_df["popularity"] >=80]

conditionlist = [
    (spotify_df['popularity'] >= 80) ,
    (spotify_df['popularity'] <80)]

choicelist = [1,0]
spotify_df['pop_classe'] = np.select(conditionlist, choicelist, default='Not Specified')
spotify_df['pop_classe'] = spotify_df['pop_classe'].astype(int)

spotify_df = spotify_df.dropna() #removendo valores nulos
spotify_df.columns

spotify_df = spotify_df.drop(columns=['popularity'])
spotify_df.info()

#Mantendo somente colunas quantitativas e que são importantes para o modelo

df_quantitative = spotify_df
cols_to_drop = []
for column in spotify_df:
    if spotify_df[column].dtype == 'object':
        cols_to_drop.append(column)
df_quantitative = spotify_df.drop(columns=cols_to_drop)

print(f"Tamanho da base de dados: {df_quantitative.shape}")

df_quantitative.head()

df_quantitative = df_quantitative.drop(columns=['explicit'])
df_quantitative.info()

# Normalizando os dados, deixando na mesma escala
df_quantitative_nm=(df_quantitative-df_quantitative.min())/(df_quantitative.max()-df_quantitative.min())
df_quantitative_nm.head()