from sklearn import cluster
import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from wordcloud import WordCloud
import bz2

plt.style.use("ggplot")
plt.rcParams['figure.figsize'] = [10, 4]
showPyplotGlobalUse = False


# App layout
st.set_page_config(layout="centered", initial_sidebar_state="collapsed")
siteHeader = st.container()
Clust = st.container()
WordCl = st.container()
dataExploration = st.container()
FindWords = st.container()
WordCount = st.container()
Simil = st.container()

@st.cache(allow_output_mutation=True)
def read_data():
  with bz2.BZ2File('data.pbz2', 'rb') as f:
      df = pickle.load(f)
  df["cluster"] = df["cluster"].astype("str")

  return df

def find(df, word, times=100):

    ind = []
    counts = []
    for i, row in df.iterrows():
        c = row["text"].count(word)
        if c >= times:
            ind.append(i)
            counts.append(c)

    to_ret = df.iloc[ind].copy()
    to_ret["counts"] = counts
    
    return to_ret

def and_top_words(df, n=20):
    
    flg = True
    for _, row in df.iterrows():

        tops = pd.DataFrame(pd.Series(row["text"]).value_counts(normalize=True))
        tops = tops.rename(columns={0: row["title"]})

        if flg:
            all_tops = tops.copy()
            flg = False

        else:
            all_tops = all_tops.merge(tops,left_index=True, right_index=True)       

    return all_tops[:n]

def plot_top_words(df, selected, num=20):

  rows = df[df["title"].isin(selected)]
  return and_top_words(rows, num).plot.bar().get_figure()

def get_closest(title, n):

    inp = np.array(df[df["title"] == title][["tsne1", "tsne2"]])
    dists = []

    for _,row in df.iterrows():
        
        dists.append(np.linalg.norm(np.array(row[["tsne1", "tsne2"]]) - inp))

    dd = df.loc[:,["title"]]
    dd["dist"] = dists
    dd.index = dd["title"]
    dd = dd.sort_values(by="dist")
    dd = dd.drop(title)

    #return dd.head(n).plot.barh().get_figure()
    fig = px.bar(dd.head(n), x="title", y="dist", text="title")
    fig.update_layout(xaxis_visible=False, showlegend=True)
    return fig

def plot_cloud(df, selected):
  
  txt = " ".join(df[df["title"]==selected]["text"].item())
  wc = WordCloud(width = 800, height = 800).generate(txt)

  fig = plt.figure(figsize = (8, 8), facecolor = None)
  plt.imshow(wc)
  plt.axis("off")
  plt.tight_layout(pad = 0)
  
  return fig 

### APP CODE ###
df = read_data()  # load data

# Podcast analytics page
def data_analysis():

  # Title
  with siteHeader:
    st.title('Lex Fridman Podcast NLP')
    st.markdown('''An app to explore the [Lex Fridman Podcast](https://lexfridman.com/podcast/).''')
    st.markdown('''Some time ago, I was wondering if I could choose an episode based on its
    similarity with others that I had already listened to. So I decided to 
    create a clustering of the episodes based on the text transcripts.''')
    st.markdown('''After downloading the data, I started doing some analysis and realised
    that there were a few interesting statistics about the episodes.
    Thus, I decided to create this set of widgets to have a look at the
    insights of each episode.''')
    st.markdown('''Note: The list will not be updated to the latest episode since
    the encoding phase isn't automated yet. Also some episodes are
    misssing since there was no available transcription.''')
    
  # Cluster
  with Clust:
    st.header("Clustering of the episodes")
    st.markdown(
    '''The clustering was created by encoding the text from the transcriptions with the 
[Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder-large/5) model by Google and by applying the [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) algorithm
to the results.


Note: the transcription were parsed, tokenized and the most stop words removed to 
improve performance.'''
    )
    fig = px.scatter(df, x="tsne1", y="tsne2", color="cluster", hover_name="title")
    fig.update_layout(xaxis_visible=False, yaxis_visible=False, showlegend=False)
    st.plotly_chart(fig)

  # Word search & count
  with FindWords:
    st.header("Word Search")
    st.markdown("Here we can search for a term and the number of occurences.")

    w = st.text_input('Which word would you like to search for?', 'ai')
    rep = st.slider('Number of times that the word has been said?', min_value=1, max_value=1000, value=20, step=10)

    st.write(find(df, w, rep)[["title", "person", "counts"]])

  # Most common words
  with WordCount:
    st.header("Common words")
    st.markdown("Find most common words for each podcast.")

    sel_pod = st.multiselect("Choose one or more podcasts", df["title"].tolist())
    n_words = st.slider("Number of words to plot", min_value=5, max_value=100, step=1, value=20)
    but_count = st.checkbox("Show")
    
    if but_count and len(sel_pod) > 0:
      st.pyplot(plot_top_words(df, sel_pod, n_words))

  # word cloud
  with WordCl:
    st.header("Word Cloud")
    st.markdown('''Create [word cloud](https://pypi.org/project/wordcloud/) by choosing the podcast title and click on "show image."''')

    selection_col, display_col = st.columns(2)

    sel_pod_cloud = selection_col.selectbox("Choose the podcast...", df["title"].tolist())
    
    but_cloud = selection_col.checkbox("Show Image")
    if but_cloud:
      display_col.write(plot_cloud(df, sel_pod_cloud))

  # Similarity
  with Simil:
    st.header("Find similar podcasts")
    st.markdown("In this section, it is possible to find the most similar podcasts to the one in input. The metric used for similarity is the cosine distance between the text embeddings.")
    sel_pod2 = st.selectbox("Choose an episode", df["title"].tolist())
    n_words2 = st.slider("Number of similar episodes to plot", min_value=2, max_value=20, step=1, value=5)
    but_count2 = st.checkbox("Show", key="similcheck")
    
    if but_count2 and len(sel_pod2) > 0:
      #st.pyplot(get_closest(sel_pod2, n_words2))
      st.plotly_chart(get_closest(sel_pod2, n_words2))

# About page
#def about():

#  st.title("The project")

# Other info
#def future():
#  st.title("Next steps...")

# pages = {
#          "Podcast data analysis": data_analysis,
#          #"About the project": about,
#          #"Next steps": future 
#        }

# st.sidebar.title("Menu")
# page = st.sidebar.radio("Select your page", tuple(pages.keys()))

# pages[page]()
data_analysis()


