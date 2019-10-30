import pandas as pd
import string
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pickle
from tqdm import tqdm
from collections import OrderedDict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter
from scipy import sparse
import numpy as np

df = pd.read_csv('/san2/data/websci/mcc/all_docs.csv')
df = df[['PY', 'content']]
print(df.shape)

df = df.dropna()
print(df.shape)

prepr_abstracts = []
all_abstracts = df['content'].tolist()
for prepr in tqdm(all_abstracts):
    prepr = prepr.lower()
    prepr = re.sub(r'\d+', '', prepr)
    prepr = prepr.translate(str.maketrans('','', string.punctuation))
    prepr = prepr.strip()
    prepr = word_tokenize(prepr)
    prepr_abstracts.extend(prepr)
freq_count = Counter(prepr_abstracts)
print("counted")
freq = [key for (key, value) in freq_count.most_common(21000)]

#freq = freq_count[freq_count.values < 2]
print(len(freq))
#freq = list(freq.index)
print(freq[:10])

stop_words = set(stopwords.words('english'))
more_prepr = [i for i in freq if not i in stop_words]

tok2indx = dict()
unigram_counts = Counter()
for ii, abstract in enumerate([more_prepr]):
    #if ii % 200000 == 0:
    #    print(f'finished {ii/len(more_prepr):.2%} of abstracts')
    for token in abstract:
        unigram_counts[token] += 1
        if token not in tok2indx:
            tok2indx[token] = len(tok2indx)
indx2tok = {indx:tok for tok,indx in tok2indx.items()}
print('done')
print('vocabulary size: {}'.format(len(unigram_counts)))
print('most common: {}'.format(unigram_counts.most_common(10)))
with open("data/tok2idx.pkl", "wb") as f:
    pickle.dump(tok2indx, f)

with open("data/idx2tok.pkl", "wb") as f:
    pickle.dump(indx2tok, f)

df['PY'] = df['PY'].astype(int)
print(df.groupby('PY').count())

# binning decided on evening out bucket sizes
df['binned'] = pd.cut(df['PY'], [1985, 2009, 2013, 2016, 2019])
print(df)
#for i in range(1): 
for name, group in df.groupby('binned'):
    print("Training on Bucket {0}-{1}...".format(name.left, name.right))
    print(group.shape)

    #text = df['content'].tolist()
    text = group['content'].tolist()
    print(text[:2])

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
        tmp = [i for i in tmp if i in freq]
        tmp = [i for i in tmp if not i in stop_words]
        #tmp = [i for i in tmp if not i in freq]
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

    #with open("data/mcc_{0}_{1}.list".format(name.left, name.right), "wb") as f:
    #with open("data/mcc_{0}.list".format("all_small"), "wb") as f:
    with open("data/mcc_small_{0}_{1}.list".format(name.left, name.right), "wb") as f:
        pickle.dump(processed_text, f)

    ################## Yao et al. wants co-occurrence matrix and PPMI
    
    window = 5
    skipgram_counts = Counter()
    for iabstract, abstract in enumerate(processed_text):
        for ifw, fw in enumerate(abstract):
            icw_min = max(0, ifw - window)
            icw_max = min(len(abstract) - 1, ifw + window)
            icws = [ii for ii in range(icw_min, icw_max + 1) if ii != ifw]
            for icw in icws:
                skipgram = (abstract[ifw], abstract[icw])
                skipgram_counts[skipgram] += 1    
        if iabstract % 20000 == 0:
            print(f'finished {iabstract/len(processed_text):.2%} of abstract')    
    print('done')
    print('number of skipgrams: {}'.format(len(skipgram_counts)))
    print('most common: {}'.format(skipgram_counts.most_common(10)))

    row_indxs = []
    col_indxs = []
    dat_values = []
    ii = 0
    for (tok1, tok2), sg_count in skipgram_counts.items():
        ii += 1
        if ii % 1000000 == 0:
            print(f'finished {ii/len(skipgram_counts):.2%} of skipgrams')
        tok1_indx = tok2indx[tok1]
        tok2_indx = tok2indx[tok2]
        
        row_indxs.append(tok1_indx)
        col_indxs.append(tok2_indx)
        dat_values.append(sg_count)
    
    wwcnt_mat = sparse.csr_matrix((dat_values, (row_indxs, col_indxs)))
    print(wwcnt_mat.shape)
    print('done')

    num_skipgrams = wwcnt_mat.sum()
    assert(sum(skipgram_counts.values())==num_skipgrams)

    # for creating sparce matrices
    row_indxs = []
    col_indxs = []

    #pmi_dat_values = []
    ppmi_dat_values = []
    #spmi_dat_values = []
    #sppmi_dat_values = []

    # smoothing
    alpha = 0.75
    nca_denom = np.sum(np.array(wwcnt_mat.sum(axis=0)).flatten()**alpha)
    sum_over_words = np.array(wwcnt_mat.sum(axis=0)).flatten()
    sum_over_words_alpha = sum_over_words**alpha
    sum_over_contexts = np.array(wwcnt_mat.sum(axis=1)).flatten()

    ii = 0
    for (tok1, tok2), sg_count in skipgram_counts.items():
        ii += 1
        if ii % 1000000 == 0:
            print(f'finished {ii/len(skipgram_counts):.2%} of skipgrams')
        tok1_indx = tok2indx[tok1]
        tok2_indx = tok2indx[tok2]
    
        nwc = sg_count
        Pwc = nwc / num_skipgrams
        nw = sum_over_contexts[tok1_indx]
        Pw = nw / num_skipgrams
        nc = sum_over_words[tok2_indx]
        Pc = nc / num_skipgrams
    
        #nca = sum_over_words_alpha[tok2_indx]
        #Pca = nca / nca_denom
    
        pmi = np.log2(Pwc/(Pw*Pc))
        ppmi = max(pmi, 0)
    
        #spmi = np.log2(Pwc/(Pw*Pca))
        #sppmi = max(spmi, 0)
    
        row_indxs.append(tok1_indx)
        col_indxs.append(tok2_indx)
        #pmi_dat_values.append(pmi)
        ppmi_dat_values.append(ppmi)
        #spmi_dat_values.append(spmi)
        #sppmi_dat_values.append(sppmi)
        
    #pmi_mat = sparse.csr_matrix((pmi_dat_values, (row_indxs, col_indxs)))
    ppmi_mat = sparse.csr_matrix((ppmi_dat_values, (row_indxs, col_indxs)))
    #spmi_mat = sparse.csr_matrix((spmi_dat_values, (row_indxs, col_indxs)))
    #sppmi_mat = sparse.csr_matrix((sppmi_dat_values, (row_indxs, col_indxs)))

    print(ppmi_mat.shape)

    sparse.save_npz("data/mcc_{0}_{1}_ppmi.npz".format(name.left, name.right), ppmi_mat)
    #sparse.save_npz("data/mcc_{0}_ppmi.npz".format("all"), ppmi_mat)
    #print(ppmi_mat[:10])

    print('done') 
