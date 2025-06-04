# netflix_titles.csv
Netflix and other OTT platforms heavily rely on personalized recommendations to retain users. This project mimics a real-world scenario where a data analyst or BI developer helps product teams build insights and personalization features for improving content discovery and customer satisfaction.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/netflix_titles.csv')
df.head()

# Null values
df.isnull().sum()

# Top genres
df['listed_in'].value_counts().head(10).plot(kind='barh', title='Top Genres')

# Content count by country
top_countries = df['country'].value_counts().head(10)
top_countries.plot(kind='bar', title='Top Producing Countries')

# Yearly content trend
df['date_added'] = pd.to_datetime(df['date_added'])
df['year_added'] = df['date_added'].dt.year
df['year_added'].value_counts().sort_index().plot(title='Netflix Additions Over Years')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

df['description'] = df['description'].fillna('')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]
