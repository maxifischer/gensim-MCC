import pandas as pd
import string
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle
from tqdm import tqdm
from collections import Counter
from scipy import sparse
import numpy as np
import argparse
import os
from helper import str2bool


stop_words = set(stopwords.words('english'))


def preprocess(df, remove_stopwords=False, limited_vocab=False, freq=None):
    """preprocesses dataframes according to well-known text data cleaning steps

        Performs the following steps: lowercase words, delete strings with numbers, punctuation and whitespace

        Parameters
        ----------
        df : pd.Dataframe
            dataframe to be preprocessed

        returns tokenized abstracts
    """
    print('Preprocess data...')
    prepr_abstracts = []
    all_abstracts = df['content'].tolist()
    for prepr in tqdm(all_abstracts):
        prepr = prepr.lower()
        prepr = re.sub(r'\d+', '', prepr)
        prepr = prepr.translate(str.maketrans('','', string.punctuation))
        prepr = prepr.strip()
        prepr = word_tokenize(prepr)
        if remove_stopwords:
            prepr = [i for i in prepr if not i in stop_words]
        if limited_vocab:
            prepr = [i for i in prepr if i in freq]
        prepr_abstracts.extend(prepr)
    return prepr_abstracts


def limit_vocabulary(abstracts, voc_size=21000):
    """limits vocabulary to most frequent words in corpus

        Parameters
        ----------
        abstracts: list
            list of preprocessed abstract texts
        voc_size: int
            number of words in the vocabulary

        returns
            reduced vocabulary abstracts
            dictionary of each word with its frequency
    """
    print('Limit vocabulary...')
    freq_count = Counter(abstracts)
    freq = [key for (key, value) in freq_count.most_common(voc_size)]

    more_prepr = [i for i in freq if not i in stop_words]
    return more_prepr, freq

def analyze_dataset(dataset, save_dicts=True):
    tok2indx = dict()
    unigram_counts = Counter()
    for ii, abstract in enumerate([dataset]):
        for token in abstract:
            unigram_counts[token] += 1
            if token not in tok2indx:
                tok2indx[token] = len(tok2indx)
    indx2tok = {indx:tok for tok,indx in tok2indx.items()}
    print('Vocabulary size: {}'.format(len(unigram_counts)))
    print('Most common: {}'.format(unigram_counts.most_common(10)))

    if save_dicts == True:
        with open("data/tok2idx.pkl", "wb") as f:
            pickle.dump(tok2indx, f)

        with open("data/idx2tok.pkl", "wb") as f:
            pickle.dump(indx2tok, f)
    return tok2indx


def ppmi_calc(dataset, tok2indx):
    print('Calculating PPMI...')
    window = 5
    skipgram_counts = Counter()
    for iabstract, abstract in enumerate(dataset):
        for ifw, fw in enumerate(abstract):
            icw_min = max(0, ifw - window)
            icw_max = min(len(abstract) - 1, ifw + window)
            icws = [ii for ii in range(icw_min, icw_max + 1) if ii != ifw]
            for icw in icws:
                skipgram = (abstract[ifw], abstract[icw])
                skipgram_counts[skipgram] += 1
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
    ppmi_dat_values = []

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

        pmi = np.log2(Pwc/(Pw*Pc))
        ppmi = max(pmi, 0)

        row_indxs.append(tok1_indx)
        col_indxs.append(tok2_indx)
        ppmi_dat_values.append(ppmi)

    ppmi_mat = sparse.csr_matrix((ppmi_dat_values, (row_indxs, col_indxs)))

    print(ppmi_mat.shape)

    return ppmi_mat


def main(args):
    dataset_path = args['input']
    save_folder = args['output']
    voc_size = args['vocabulary_size']
    ppmi_needed = args['ppmi']
    preprocess_all = args['preprocess_all']

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    df = pd.read_csv(dataset_path)
    # only select publishing year (PY) and abstracts (content)
    df = df[['PY', 'content']]
    df = df.dropna()

    prepr_abstracts = preprocess(df)
    more_prepr, freq = limit_vocabulary(prepr_abstracts, voc_size)

    tok2idx = analyze_dataset(more_prepr, False)

    df['PY'] = df['PY'].astype(int)
    print(df.groupby('PY').count())

    # binning decided on evening out bucket sizes
    bins = [1985, 2009, 2013, 2016, 2019]

    # generates groups decided by cut criterium
    df['binned'] = pd.cut(df['PY'], bins)
    print(df)

    if preprocess_all:
        processed_text = preprocess(df, remove_stopwords=True, limited_vocab=True, freq=freq)

        num_tokens = sum([len(abstract) for abstract in processed_text])
        print(num_tokens)
        print(num_tokens/len(processed_text))

        ### EMB-A preprocessing
        with open(os.path.join(save_folder, "mcc_all.list"), "wb") as f:
            pickle.dump(processed_text, f)

    for name, group in df.groupby('binned'):
        print("Training on Bucket {0}-{1}...".format(name.left, name.right))
        print(group.shape)

        processed_text = preprocess(group, remove_stopwords=True, limited_vocab=True, freq=freq)

        num_tokens = sum([len(abstract) for abstract in processed_text])
        print(num_tokens)
        print(num_tokens/len(processed_text))

        ### EMB-B and EMB-C preprocessing
        with open(os.path.join(save_folder, "mcc_{0}_{1}.list".format(name.left, name.right)), "wb") as f:
            pickle.dump(processed_text, f)

        ################## Yao et al. wants co-occurrence matrix and PPMI
        if ppmi_needed:
            ppmi_mat = ppmi_calc([processed_text], tok2idx)
            ### EMB-D preprocessing
            sparse.save_npz(os.path.join(save_folder, "mcc_ppmi_{0}_{1}.npz".format(name.left, name.right)), ppmi_mat)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', default='/san2/data/websci/mcc/all_docs.csv', help='input file')
    parser.add_argument('output', default='output', help='output folder')
    parser.add_argument('vocabulary_size', type=int, default=21000, help='restrictions on the vocabulary size')
    parser.add_argument('ppmi', type=str2bool, nargs='?', const=True, default=False, help='whether or not to perform PPMI processing')
    parser.add_argument('preprocess_all', type=str2bool, nargs='?', const=True, default=False, help='whether or not to preprocess the whole dataset')
    args = vars(parser.parse_args())
    main(args)
