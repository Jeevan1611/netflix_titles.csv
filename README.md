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
