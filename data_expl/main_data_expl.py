import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import re
import string
from nltk.tokenize import word_tokenize
import pickle
import seaborn as sns
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 100)

df = pd.read_csv("/san2/data/websci/mcc/all_docs.csv")
print(df.columns)
df['rating_avg'] = df[['rating_1', 'rating_2', 'rating_3', 'rating_39', 'rating_7']].mean(axis=1)
print(df[df['ratings']>0][:10])
print(df[(df['rating_avg']>0) & (df['rating_avg']<1)])
df = df[['PY', 'content', 'majority_rating', 'rating_avg', 'wosarticle__wc']]
df_tagged = df.dropna()
#df_tagged['PY'] = df['PY'].astype(int)
print(df_tagged.shape)
print(df.shape)
df = df.dropna(subset=['content', 'PY'])
df['PY'] = df['PY'].astype(int)
print(df.shape)

########## Plotting year ~ abstracts
print("Plotting year against abstracts")
print(df['PY'].value_counts())
fig, ax = plt.subplots(1,1)
sns.distplot(df['PY'].value_counts(), kde=False, ax=ax)
plt.xlabel('Publishing Year')
plt.ylabel('Number of Abstracts')
plt.savefig('./figures/year_abstracts.pdf')
##########

#text = df['content'].values.astype('U').tolist()

#df['community'] = df['wosarticle__wc'].apply(len)#.str.strip('\]\[').str.replace('[\'"]', '').str.split(',|;')
df['community'] = df['wosarticle__wc'].str.lower()
df['community'] = df['community'].str.replace(r'green & sustainable science & technology,?;? ?', '')
df['community'] = df['community'].str.replace(r'sciences', '')
#df['community'] = df['community'].str.replace(r'environmental studies,?;? ?', '')
df['community'] = df['community'].str.replace(r'multidisciplinary,?;? ?', '')
df['community'] = df['community'].str.replace(r'engineering, environmental', 'environmental engineering')
df['community'] = df['community'].str.replace(r'\[|\]|\'|\"', '')
df['community'] = df['community'].str.strip()
df['community'] = df['community'].str.replace(r'[,|;]+', '')
df['community'] = df['community'].where(~df['community'].str.contains('environment'), 'environmental')
df['community'] = df['community'].where(~df['community'].str.contains('energ'), 'energy')
df['community'] = df['community'].where(~df['community'].str.contains('chemi'), 'chemical')
df['community'] = df['community'].where(~df['community'].str.contains('forest'), 'forestry')
df['community'] = df['community'].where(~df['community'].str.contains('agri'), 'agricultural')
df['community'] = df['community'].where(~df['community'].str.contains('ecol'), 'ecological')
df['community'] = df['community'].where(~df['community'].str.contains('agro'), 'agronomical')
df['community'] = df['community'].where(~df['community'].str.contains('econ'), 'economical')
df['community'] = df['community'].where(~df['community'].str.contains('water'), 'water')
df['community'] = df['community'].where(~df['community'].str.contains('fish'), 'fisheries')
df['community'] = df['community'].where(~df['community'].str.contains('biol'), 'biological')
df['community'] = df['community'].where(~df['community'].str.contains('engin'), 'engineering')
df['community'] = df['community'].where(~df['community'].str.contains('medic|health'), 'medicine & health care')
print(df.groupby(['community']).count().sort_values(['content'], ascending=False))
df.groupby(['community']).count().sort_values(['content'], ascending=False).to_csv("communities.csv")
#print(df['community'])
#processed_text = []
tqdm.pandas()
df['content'] = df['content'].str.lower()
df['content'] = df['content'].str.replace(r'\d+', '')
regexPart1 = r"\s"
regexPart2 = r"(?:s|'s|!+|,|\.|-|;|:|\(|\)|\"|\?+)?\s"  
#df['content'] = df['content'].str.replace(regexPart1 + re.escape(df['content']) + regexPart2, re.IGNORECASE)
df['content'] = df['content'].str.replace(regexPart1 + regexPart2, '')
df['content'] = df['content'].str.strip()
df['wc'] = df.progress_apply(lambda row: len(word_tokenize(row['content'])), axis=1)
print(df['content'][:10])
print(df['wc'][:10])

df.to_csv("preprocessed_data.csv")
#cv = CountVectorizer()
#tfidf = TfidfVectorizer(lowercase=False)
#wc_vector = tfidf.fit_transform(df['content'])

#pickle.dump(wc_vector, open("wc_vector.dat", "wb"))
#print(wc_vector.shape)
#print(wc_vector[:20])
#print(wc_vector[:-20])
