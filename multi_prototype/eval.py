import numpy as np
import pickle
from gensim.models import KeyedVectors
import simple_multiprotoype as smp
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr


file_path = "./data"
def matrix_to_word2vecformat(mat, idx2tok, eval_word):
    print(mat.shape)
    f = open("{0}/word2vec_mat_{1}.bin".format(file_path, eval_word), "w")
    f.write("{0} {1}\n".format(mat.shape[0], mat.shape[1]))
    for idx, row in enumerate(mat):
        f.write("{0}".format(idx2tok[idx]))
        for val in row:
            f.write(" {0}".format(val))
        f.write("\n")
    f.close()
    print("Matrix transformed")


#v_s = np.load("./runs/v_s.npz")
v_g = np.load("./runs/v_g.npz")

data = pickle.load(open("../data/mcc_all_small.list", "rb"))
vocab = pickle.load(open("./data/vocab", 'rb')) #set([word for abstract in data for word in abstract])
vocab_size = len(vocab)
idx2tok = {k : v for k,v in enumerate(vocab)}
tok2idx = {v : k for k,v in enumerate(vocab)}
eval_idx = vocab.index("plant")
print(idx2tok[eval_idx])

#matrix_to_word2vecformat(v_g, idx2tok, eval_words[0])
v_g_mat = KeyedVectors.load_word2vec_format("./data/emb_a_mcc_all_small.list.bin")# "./data/word2vec_mat_clean.bin")


wordsim = pd.read_csv('../eval_data/wordsim353.tsv', sep="\t", header=None, names=['left', 'right', 'val'])
print(wordsim)
wordsim_rel = pd.read_csv('../eval_data/wordsim_relatedness_goldstandard.txt', sep="\t", header=None, names=['left', 'right', 'val'])
print(wordsim_rel)
wordsim_sim = pd.read_csv ('../eval_data/wordsim_similarity_goldstandard.txt', sep="\t", header=None, names=['left', 'right', 'val'])
print(wordsim_sim)
K = 3

for dataset in [wordsim, wordsim_rel, wordsim_sim]:
    eval_words = list(set(dataset['left'].tolist() + dataset['right'].tolist())) #['plant']
    print(len(eval_words))
    print(eval_words[:5])
    run_name = "run5"

    smp.reisinger_clustering(run_name, eval_words)
    my = np.load("./runs/my_{0}.npz".format(run_name))

    """
##### NNs
for k in range(3):
    #print(my[eval_idx, k])
    #print(v_s[eval_idx, k])
    nns = v_g_mat.similar_by_vector(my[eval_idx, k], 10)
    print([x[0] for x in nns])
    """

    actual = dataset['val'].tolist()
    skipped = []
    avg_pred = []
    max_pred = []
    for idx, row in dataset.iterrows():
        try:
            left_idx = tok2idx[row['left']]
            right_idx = tok2idx[row['right']]
            avg_sim = 0
            max_sim = 0
            for i in range(K):
                for j in range(K):
                    cos_sim =  cosine_similarity([my[left_idx, i]], [my[right_idx, j]])[0][0]
                    avg_sim += cos_sim
                    if max_sim < cos_sim:
                        max_sim = cos_sim
            avg_sim /= (K*K)
            avg_pred.append(avg_sim)
            max_pred.append(max_sim)
        except KeyError:
            skipped.append(idx)
            continue
    actual = [v for ix, v in enumerate(actual) if ix not in skipped]
    print(spearmanr(actual, avg_pred))
    print(spearmanr(actual, max_pred))
