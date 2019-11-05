from gensim.models import KeyedVectors
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
from collections import Counter
import math
from tqdm import tqdm

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


# input w1, w2, wT, d, K, N
d = 200
N = 5 # context window
K = 3 # number of senses
neg = 1

# load word vectors of EMB-A and its initial data
data = pickle.load(open("../data/mcc_all_small.list", "rb"))
v_w = KeyedVectors.load_word2vec_format("./data/emb_a_mcc_all_small.list.bin")
w = list(v_w.wv.vocab.keys())
print(w[:5])

# init v_s(w, k) randomly and v_g(w) and my(w,k) to 0
v_s = np.random.rand(len(w), K, d) # sense vectors
v_g = v_w.syn0 # global word embeddings from Word2Vec
my = np.random.rand(len(w), K, d) # cluster centers
s_t = dict()

# calculate unigram distribution
flattened = [a for abstract in data for a in abstract]
#print(len(flattened))
freq_count = Counter(flattened) # (word, freq)
count = 0
for _, v in freq_count.items():
    if v>200:
        count +=1
print(count)
print(len([filt for filt in data if filt != []]))


freq_count_norm = {k: v/len(flattened) for k,v in freq_count.items()}
#print(freq_count_norm['sustainable'])
#print(sum(freq_count_norm.values()))

alpha = 3/4 # mentioned by Word2Vec paper to be a good value
noise_dist = {key: val ** alpha for key, val in freq_count_norm.items()}
Z = sum(noise_dist.values())
P_nw = {key: val / Z for key, val in noise_dist.items()}
#print(noise_dist_normalized['sustainable'])
#print(sum(noise_dist_normalized.values()))

alpha = 0.025

# loop through all words in training set
for abstract in tqdm(data):
    for t, curr_word in enumerate(abstract):
        if curr_word == "clean":
            R_t = range(1, 5) # context window range
            word = abstract[t]
            w_i = w.index(word)
            c_t = []
            c_t.extend(abstract[max(t-N, 0):t] + abstract[min(t+1, len(abstract)) : min(t+N+1, len(abstract))])
            v_context = 1/(2*N) * np.sum([v_g[w.index(c)] for c in c_t], axis=0)
            
            max_sim = 0
            s = 0
            for k in range(K):
                cos_sim = 1 - spatial.distance.cosine(my[w_i, k, :], v_context)
                if cos_sim > max_sim:
                    max_sim = cos_sim
                    s = k
            s_t.setdefault((w_i, s), []).append(v_context)
            my[w_i, s] = np.mean(s_t[(w_i, s)], axis=0)
        
            # based on Stergiou et al. 2017
            # or this blog post https://aegis4048.github.io/optimize_computational_efficiency_of_skip-gram_with_negative_sampling

            # draw negative samples
            neg_samples = np.random.choice(list(P_nw.keys()), size=neg, p = list(P_nw.values()))
            c_t_neg = [v_g[w.index(n)] for n in neg_samples]
            sigmoid_v = np.vectorize(sigmoid)
        
            # compute gradients
            h = v_g[w_i]
            grad_V_output_pos = (sigmoid_v(v_context * h) - 1) * h
            grad_V_input = (sigmoid_v(v_context * h) - 1) * v_context
            grad_V_output_neg_list = []
            for c_neg in c_t_neg:
                grad_V_output_neg_list.append(sigmoid_v(c_neg * h) * h)
                grad_V_input += sigmoid_v(c_neg * h) * c_neg

            # use SGD to update w, c_pos, and c_neg_1, ... , c_neg_K
            v_s[w_i, s] -= alpha * grad_V_input
            for pw in c_t:
                v_g[w.index(pw)] -= alpha * grad_V_output_pos
            for i, nw in enumerate(neg_samples):
                v_g[w.index(nw)] -= alpha * grad_V_output_neg_list[i]

np.save(open("./runs/v_s.npz", 'wb'), v_s)
np.save(open("./runs/v_g.npz", 'wb'), v_g)
np.save(open("./runs/my.npz", 'wb'), my)
