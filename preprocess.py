import pandas as pd
import string
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pickle
from tqdm import tqdm

df = pd.read_csv('/san2/data/websci/mcc/all_docs.csv')
df = df[['PY', 'content']]
print(df.shape)

df = df.dropna()
print(df.shape)

df['PY'] = df['PY'].astype(int)
print(df.groupby('PY').count())

# binning decided on evening out bucket sizes
df['binned'] = pd.cut(df['PY'], [1985, 2009, 2013, 2016, 2019])
print(df)

for name, group in df.groupby('binned'):
    print("Training on Bucket {0}-{1}...".format(name.left, name.right))
    print(group.shape)

    text = group['content'].tolist()
    #print(text[:2])

    processed_text = []
    for abstract in tqdm(text):
        # 1. remove uppercase
        #print("Removing uppercase...")
        tmp = abstract.lower()
        #print(tmp)

        # 2. remove numbers
        #print("Removing numbers...")
        tmp = re.sub(r'\d+', '', tmp)
        #print(tmp)

        # 3. remove punctuation
        #print("Removing punctuation...")
        tmp = tmp.translate(str.maketrans('','', string.punctuation))
        #print(tmp)

        # 4. remove whitespaces
        #print("Removing whitespaces...")
        tmp = tmp.strip()
        #print(tmp)

        # 5. tokenization
        #print("Tokenize...")
        tmp = word_tokenize(tmp)
        #print(tmp)

        # 6. remove stop words
        #print("Removing stopwords...")
        stop_words = set(stopwords.words('english'))
        tmp = [i for i in tmp if not i in stop_words]
        #print(tmp)

        # 7. stem words
        ### not sure if useful
        #stemmer = PorterStemmer()
        #tmp = [stemmer.stem(word) for word in tmp]
        #print(tmp)

        # 8. lemmatize
        ### not sure if useful
        #lemmatizer = WordNetLemmatizer()
        #tmp = [lemmatizer.lemmatize(word) for word in tmp]
        #print(tmp)
    
        processed_text.append(tmp)
    print(len(processed_text))
    num_tokens = sum([len(abstract) for abstract in processed_text])
    print(num_tokens)
    print(num_tokens/len(processed_text))
    #print(processed_text[:200])

    with open("data/mcc_{0}_{1}.list".format(name.left, name.right), "wb") as f:
        pickle.dump(processed_text, f)
