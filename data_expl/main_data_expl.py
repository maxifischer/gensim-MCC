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
from collections import Counter, OrderedDict


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
#print("Plotting year against abstracts")
#print(df['PY'].value_counts())
#count_data=df['PY'].value_counts()
#ax = sns.barplot(count_data.index, count_data.values)
#plt.xlabel('Publishing Year')
#plt.ylabel('Number of Abstracts')
#for ind, label in enumerate(ax.get_xticklabels()):
#    if ind % 3 == 0:  # every 10th label is kept
#        label.set_visible(True)
#    else:
#        label.set_visible(False)
#plt.savefig('./figures/year_abstracts.pdf')
##########

#text = df['content'].values.astype('U').tolist()

############### community preprocessing
df['community'] = df['wosarticle__wc'].str.lower()
df['community'] = df['community'].str.replace(r'\[|\]|\'|\"|&', '')
df['community'] = df['community'].str.strip()
df['community'] = df['community'].str.replace(r'[,|;]+', '')

#################### filters for 8 communities
#df['community'] = df['community'].where(~df['community'].str.contains('environmental|ecology'), 'environmental')
#df['community'] = df['community'].where(~df['community'].str.contains('energy|fuels'), 'energy')
#df['community'] = df['community'].where(~df['community'].str.contains('chemistry|chemical'), 'chemical')
#df['community'] = df['community'].where(~df['community'].str.contains('water'), 'water')
#df['community'] = df['community'].where(~df['community'].str.contains('management|economics'), 'economics')
#df['community'] = df['community'].where(~df['community'].str.contains('agriculture|agronomy'), 'agriculture agronomy')
#df['community'] = df['community'].where(~df['community'].str.contains('materials'), 'materials')
#df['community'] = df['community'].where(~df['community'].str.contains('forestry'), 'forestry')
#df['community'] = df['community'].where(df['community'].str.contains('environmental|energy|chemical|water|economics|agriculture|materials|forestry'), 'forestry')

############### community data exploration
df['community'] = df['community'].str.split()
communities = df['community'].tolist()
communities = [item for sublist in communities for item in sublist]
print(communities[:10])
counted = Counter(communities)
sorted_comm = sorted(counted.items(), key=lambda kv: kv[1], reverse=True)
print(sorted_comm)
###############
############## plotting word frequency
print("Plotting word frequency of word categories")
count_data = sorted_comm[:50]
print(count_data)
#ax = sns.barplot([el[0] for el in count_data], [el[1] for el in count_data])
#plt.xlabel('Most Frequently Appearing Categories')
#plt.ylabel('Number of Occurrences')
#plt.savefig('./figures/word_category_frequencies_before.pdf')
#############





#print(df['community'].value_counts().sort_values(['content'], ascending=False))
#df.groupby(['community']).count().sort_values(['content'], ascending=False).to_csv("communities.csv")
#print(df['community'])
#processed_text = []

########## preprocessing
#tqdm.pandas()
#df['content'] = df['content'].str.lower()
#df['content'] = df['content'].str.replace(r'\d+', '')
#regexPart1 = r"\s"
#regexPart2 = r"(?:s|'s|!+|,|\.|-|;|:|\(|\)|\"|\?+)?\s"  
#df['content'] = df['content'].str.replace(regexPart1 + re.escape(df['content']) + regexPart2, re.IGNORECASE)
#df['content'] = df['content'].str.replace(regexPart1 + regexPart2, '')
#df['content'] = df['content'].str.strip()
#df['wc'] = df.progress_apply(lambda row: len(word_tokenize(row['content'])), axis=1)
#print(df['content'][:10])
#print(df['wc'][:10])

#df.to_csv("preprocessed_data.csv")
#cv = CountVectorizer()
#tfidf = TfidfVectorizer(lowercase=False)
#wc_vector = tfidf.fit_transform(df['content'])

#pickle.dump(wc_vector, open("wc_vector.dat", "wb"))
#print(wc_vector.shape)
#print(wc_vector[:20])
#print(wc_vector[:-20])
