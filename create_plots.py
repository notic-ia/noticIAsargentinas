# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
import string
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
try:
  from bertopic import BERTopic
except:
  !pip install bertopic
  from bertopic import BERTopic

class news_processor:
  # meses auto
  m = 3
  df = pd.DataFrame()
  sources_df = pd.DataFrame()
  dates_df = pd.DataFrame()
  timestamp = []

  def __init__(dataset, data_url, months, section, content, stopwords_url, embedding_model=None):
    dataset.months = months
    dataset.section = section
    dataset.content = content
    dataset.stopwords_url = stopwords_url
    dataset.data_url = data_url
    dataset.embedding_model = embedding_model

    if dataset.months == 'auto':
      import time
      m = 3
      now = time.localtime()
      rango = list([time.localtime(time.mktime((now.tm_year, now.tm_mon - n, 1, 0, 0, 0, 0, 0, 0)))[:2] for n in range(m)])
      dataset.months = []
      for y,m in rango:
        dataset.months.append(str(y)+str(m).rjust(2, '0'))

  def download(dataset):
    print('Downloading')
    files = {}
    for m in dataset.months:
        try:
            url = dataset.data_url + m + '.csv.gz?raw=true'
            files[m] = pd.read_csv(url, compression='gzip')
        except:
            dataset.months.remove(m)

    dataset.df = pd.concat(files.values(), ignore_index=True)
    # Descartar categoría sobredimensionada para uno de los diarios
    dataset.df = dataset.df[dataset.df.category != 'mundo']

  def prepare(dataset):
    print('Preparing dataset')
    m = 3
    ### Vamos a construir un campo que codifique las fechas para agrupar luego (YYYYMM)
    dataset.df['date'] = pd.to_datetime(dataset.df['date'], errors='coerce', utc=True).dt.tz_convert('America/Argentina/Buenos_Aires')
    # Filter errors in dates
    dataset.df = dataset.df[dataset.df['date'] > '2021-11-30']
    dataset.df['yyyymm'] = dataset.df['date'].dt.year.astype(str) + '-' + dataset.df['date'].dt.month.astype(str)

    # Ordenar filas por medio y AAAAMM y resetear los índices
    dataset.df.sort_values(by=['yyyymm','source'],ascending=True,inplace=True)
    dataset.df.reset_index(drop=True,inplace=True)

    # Limpiar
    import time
    now = time.localtime()
    rango = list([time.localtime(time.mktime((now.tm_year, now.tm_mon - n, 1, 0, 0, 0, 0, 0, 0)))[:2] for n in range(m)])
    months_clean = []
    for y,m in rango:
      months_clean.append(str(y)+'-'+str(m))
    dataset.df = dataset.df[dataset.df.yyyymm.isin(months_clean)]

  def create_corpus_df(dataset):
    print('Creating corpus')
    # Si se definió una sección, filtrar
    if dataset.section == '':
        dataset.corpus_df = dataset.df.copy()
    else:
        dataset.corpus_df = dataset.df[dataset.df.category == dataset.section].reset_index(drop=True)

    # Concatenar medio y mes-año
    dataset.corpus_df['source'] = dataset.corpus_df['yyyymm'] + '_' + dataset.corpus_df['source']

    # conservamos los origenes y el mes año para unir luego de procesar
    dataset.sources_df = dataset.corpus_df['source']
    dataset.dates_df = dataset.corpus_df['date']

    # Según se haya definido, filtrar Titulo, Cuerpo o concatenar todo
    if  dataset.content == 1:
        dataset.corpus_df = pd.DataFrame(dataset.corpus_df.title)
        col = 'title'

    elif dataset.content == 2:
        dataset.corpus_df = pd.DataFrame(dataset.corpus_df.text)
        col = 'text'
    else:
        dataset.corpus_df = pd.DataFrame(dataset.corpus_df.title + ' ' + dataset.corpus_df.text)
        col = 0

    # Renombrar columna y asegurar que sea STR
    dataset.corpus_df.rename(columns={col:'text'},inplace=True)
    dataset.corpus_df['text'] = dataset.corpus_df.text.astype(str)

    # Store list of timestamps to reuse later
    dataset.timestamp = dataset.dates_df.to_list()

  def clean_text(dataset,text):
    '''
    Make text lowercase, remove text in square brackets, 
    remove punctuation and remove words containing numbers.
    '''
    # todo a minúsculas
    text = text.lower() 
    # remover caracteres especiales
    text = re.sub('\[.*\]\%;,"“”', ' ', text)
    # remover puntuación
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\w*\d\w*', ' ', text)
    text = text.replace('\xa0',' ')
    text = re.sub('[‘’“”…«»]', ' ', text)
    text = re.sub('\n', ' ', text)
    text = text.replace('?',' ')
    text = text.replace('¿',' ')
    # Eliminamos los caracteres especiales
    text = re.sub(r'\W', ' ', str(text))
    # Eliminado las palabras que tengo un solo caracter
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    # Sustituir los espacios en blanco en uno solo
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    return text

  def clean_corpus_df(dataset):
    print('Cleaning Corpus')
    dataset.corpus_df['text'] = dataset.corpus_df.text.apply(dataset.clean_text)

  def remove_stopwords(dataset,text):
    dataset.stripped_text = [w for w in text.split() if w.lower() not in dataset.stopwords_es and not w.isdigit()]
    return ' '.join(word for word in dataset.stripped_text)

  def remove_stopwords_corpus_df(dataset):
    print('Removing Stopwords')
    dataset.stopwords_es = pd.read_csv(dataset.stopwords_url,header=None)[0].to_list()
    dataset.corpus_df.text = dataset.corpus_df.text.apply(dataset.remove_stopwords)

  def build_final_dataset_df(dataset):
    print('Building Final Dataset')
    dataset.final_dataset_df = pd.merge(dataset.corpus_df, dataset.sources_df, left_index=True, right_index=True)

  def create_topic_model(dataset):
    # from flair.embeddings import TransformerDocumentEmbeddings
    # embeddings = TransformerDocumentEmbeddings('dccuchile/bert-base-spanish-wwm-cased')
    # from transformers.pipelines import pipeline
    # embedding_model = pipeline("feature-extraction", model="LeoCordoba/mt5-small-cc-news-es-titles")
    print('Creating Topic Model')
    dataset.topic_model = BERTopic(language="multilingual",
                       calculate_probabilities=True,
                       embedding_model=dataset.embedding_model,
                       min_topic_size=10,
                       #embedding_model="all-distilroberta-v1",
                       nr_topics= "auto")
                       #umap_model=umap_model)
    dataset.topics, dataset.probs = dataset.topic_model.fit_transform(dataset.final_dataset_df.text.to_list())

  def run_full_process(dataset):
    dataset.download()
    dataset.prepare()
    dataset.create_corpus_df()
    dataset.clean_corpus_df()
    dataset.remove_stopwords_corpus_df()
    dataset.build_final_dataset_df()
    dataset.create_topic_model()

  def plot_bars(dataset,q,figsize,save=False):
    dataset.tabla = dataset.df[dataset.df.category == dataset.section].reset_index(drop=True)
    dataset.labels = pd.DataFrame()
    dataset.labels['num'] = pd.DataFrame(dataset.topic_model.generate_topic_labels()[1:])[0].str.split('_').apply(lambda x: x[0])
    dataset.labels['tema'] = pd.DataFrame(dataset.topic_model.generate_topic_labels()[1:])[0].str.split('_').apply(lambda x: "-".join(x[1:]))
    dataset.labels['num'] = dataset.labels['num'].astype(str)
    dataset.temas = pd.DataFrame(pd.DataFrame(dataset.probs).idxmax(axis=1)).rename({0:'num'},axis=1)
    dataset.temas['num'] = dataset.temas['num'].astype(str)
    dataset.temas = pd.merge(dataset.temas,dataset.labels,on='num',how='left')['tema']
    dataset.tabla = pd.merge(dataset.tabla,dataset.temas,left_index=True, right_index=True)

    # pd.merge( tabla , tabla.groupby('source').count()['tema'].reset_index().rename({'tema':'total'},axis=1) , on='source' , how='left')

    dataset.agrupado = dataset.tabla.groupby(['source','tema','yyyymm']).count()['text'].reset_index().rename({'text':'porc'},axis=1)
    dataset.agrupado['porc'] /= dataset.agrupado.groupby(['source','yyyymm'])['porc'].transform('max').div(100)
    dataset.agrupado['porc'] = dataset.agrupado.porc.round(2)

    dataset.top = dataset.tabla.groupby(['tema']).count()['source'].reset_index().sort_values(by='source',ascending=False).head(q)['tema'].to_list()

    from matplotlib import pyplot as plt
    for pos, tema in enumerate(dataset.top):
        dataset.final = dataset.agrupado[dataset.agrupado.tema == tema ].drop(columns=['tema'])
        if save:
          name = 'plo_'+ dataset.section + '_' + str(pos) + '.png'
          dataset.final.pivot('yyyymm','source','porc').plot.bar(figsize=figsize,title=tema,alpha=0.9,rot=0,color=['#EB172B', '#F68E1E', '#006998', '#32937f'],xlabel='').get_figure().savefig(name)
          print(name,'stored')
        else:
          dataset.final.pivot('yyyymm','source','porc').plot.bar(figsize=figsize,title=tema,alpha=0.9,rot=0,color=['#EB172B', '#F68E1E', '#006998', '#32937f'],xlabel='')

  def plot_bars2(dataset,q,figsize):
    dataset.tabla = dataset.df[dataset.df.category == dataset.section].reset_index(drop=True)
    dataset.labels = pd.DataFrame()
    dataset.labels['num'] = pd.DataFrame(dataset.topic_model.generate_topic_labels()[1:])[0].str.split('_').apply(lambda x: x[0])
    dataset.labels['tema'] = pd.DataFrame(dataset.topic_model.generate_topic_labels()[1:])[0].str.split('_').apply(lambda x: "-".join(x[1:]))
    dataset.labels['num'] = dataset.labels['num'].astype(str)
    dataset.temas = pd.DataFrame(pd.DataFrame(dataset.probs).idxmax(axis=1)).rename({0:'num'},axis=1)
    dataset.temas['num'] = dataset.temas['num'].astype(str)
    dataset.temas = pd.merge(dataset.temas,dataset.labels,on='num',how='left')['tema']
    dataset.tabla = pd.merge(dataset.tabla,dataset.temas,left_index=True, right_index=True)
    # pd.merge( tabla , tabla.groupby('source').count()['tema'].reset_index().rename({'tema':'total'},axis=1) , on='source' , how='left')
    dataset.agrupado = dataset.tabla.groupby(['source','tema','yyyymm']).count()['text'].reset_index().rename({'text':'porc'},axis=1)
    dataset.agrupado['porc'] /= dataset.agrupado.groupby(['source','yyyymm'])['porc'].transform('max').div(100)
    dataset.agrupado['porc'] = dataset.agrupado.porc.round(2)
    dataset.top = dataset.tabla.groupby(['tema']).count()['source'].reset_index().sort_values(by='source',ascending=False).head(q)['tema'].to_list()
    from matplotlib import pyplot as plt
    for tema in dataset.top:
        dataset.final = dataset.agrupado[dataset.agrupado.tema == tema ].drop(columns=['tema'])
        from matplotlib import pyplot as plt
        dataset.final.pivot('yyyymm','source','porc').plot.bar(figsize=(10,3),title=tema)

  def plot_topics_over_time(dataset,bins,n=10,save=False):
    ovt_df = dataset.topic_model.topics_over_time(dataset.final_dataset_df.text.to_list() , pd.to_datetime(pd.to_datetime(dataset.timestamp).date) , nr_bins=bins)
    ovt_df = ovt_df[ovt_df.Topic.isin(np.arange(0,n))]
    topic_names = dataset.topic_model.get_topic_info()[['Topic','Name']]
    ovt_df = pd.merge(ovt_df,topic_names,how='left',left_on='Topic',right_on='Topic')
    ovt_df.Timestamp = pd.to_datetime(ovt_df.Timestamp).dt.date
    import plotly.express as px
    fig = px.line(ovt_df, x="Timestamp", y="Frequency", color='Name',hover_name=None, hover_data=["Timestamp", "Frequency", "Words"])
    if save:
      name = 'ovt_'+ dataset.section + '.html'
      fig.write_html(name)
      print(name,'stored')
    else:
      fig.show()

  def plot_topics_over_time2(dataset,topics):
    if hasattr(dataset, 'topics_over_time'):
      display(dataset.topic_model.visualize_topics_over_time(dataset.topics_over_time, topics=topics))
    else:
      dataset.topics_over_time = dataset.topic_model.topics_over_time( dataset.final_dataset_df.text.to_list() , dataset.timestamp , nr_bins=30)
      display(dataset.topic_model.visualize_topics_over_time(dataset.topics_over_time, topics=topics))

section_range = ['deportes','politica','economia']

for sect in section_range:
  data = news_processor(data_url='https://github.com/fermasia/news-base/blob/main/files/',
                        months = 'auto', #['202303','202304','202305'],
                        section = sect,   # Sección para filtrar, si queda en blanco se usan todas las noticias
                        content = 2,  # 1 solo titulos 2 solo cuerpo 3 ambos (default)
                        stopwords_url = 'https://raw.githubusercontent.com/jbagnato/machine-learning/master/nlp/spanish.txt',
                        embedding_model='distiluse-base-multilingual-cased-v1')

  data.download()
  data.prepare()
  data.create_corpus_df()
  #data.clean_corpus_df()
  data.remove_stopwords_corpus_df()
  data.build_final_dataset_df()
  data.create_topic_model()

  data.plot_bars(q=10,figsize=(18,3),save=True)
  data.plot_topics_over_time(bins=12,n=10,save=True)