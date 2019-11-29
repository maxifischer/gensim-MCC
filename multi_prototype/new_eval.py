import gensim_eval
from gensim.models import KeyedVectors
import numpy as np
from tqdm import tqdm
import pandas as pd
import re


def process_postfix_to_sense(v_s, v_g):
    all_words = v_g.vocab.keys()
    new_v_s = [[] for x in range(len(all_words))]
    for word in tqdm(all_words):
        try:
            vector = v_s.get_vector(word)
            new_v_s[v_g.vocab.get(word).index].append(vector)
        except KeyError:
            sense = 0
            try:
                while True:
                    vector = v_s.get_vector('{0}_{1}'.format(word, str(sense)))
                    new_v_s[v_g.vocab.get(word).index].append(vector)
                    sense += 1
            except KeyError:
                continue
        
    print(len(new_v_s))
    #print(v_g.get_vector('react'))
    #print(v_s.get_vector('react'))
    #print(v_s.get_vector('{0}_{1}'.format('react', str(0))))
    #print(new_v_s[6880])
    return new_v_s


def qualitative_eval(m, eval_words, nn=5):
    nns = {}
    for eval_word in eval_words:
        nns[eval_word] =[]
        try:
            ms = m.wv.most_similar(eval_word, topn=nn)
            print('Evaluating word {0}'.format(eval_word))
            print([w[0] for w in ms])
            nns[eval_word].append([w[0] for w in ms])
        except KeyError:
            regex = re.compile(('^(' + eval_word + '_\d)$'))
            #print([x for x in m.vocab.keys() if regex.match(x)])
            for sense in [s for s in m.vocab.keys() if regex.match(s)]:
                ms = m.wv.most_similar(sense, topn=nn)
                print('Evaluating word {0}'.format(sense))
                print([w[0] for w in ms])
                nns[eval_word].append([w[0] for w in ms])
    return nns


word_lists = ['plant', 'desert', 'close', 'spruce', 'branch', 'branches', 'action', 'war', 'poor', 'education', 'water', 'energy', 'renewable', 'sustainable', 'sustainability', 'community', 'communities', 'consumption', 'consume', 'peace', 'life', 'land', 'justice', 'partnership', 'partnerships', 'global', 'watch', 'duck', 'cellular', 'draw', 'bank', 'mail', 'distribution', 'depression', 'version', 'plane', 'phone', 'wave', 'cycle', 'rise', 'rising', 'level', 'drop', 'rehabilitation', 'instrument', 'instrumentation', 'operation', 'operations', 'remote', 'relation', 'relations', 'study', 'area']
thresholds = ['1.0', '0.65', '0.6']
for th in thresholds:
    v_g = KeyedVectors.load_word2vec_format("./data/emb_a_mcc_all_small.list.bin")
    sense_vectors = KeyedVectors.load_word2vec_format("./runs/emb_a_sense_{0}.bin".format(th))
    v_s = process_postfix_to_sense(sense_vectors, v_g)
    #wordsim_clas = gensim_eval.evaluate_word_pairs(v_g, v_s, '../eval_data/wordsim353.tsv')
    #wordsim_rel = gensim_eval.evaluate_word_pairs(v_g, v_s, '../eval_data/wordsim_relatedness_goldstandard.txt')
    #wordsim_sim = gensim_eval.evaluate_word_pairs(v_g, v_s, '../eval_data/wordsim_similarity_goldstandard.txt')
    #scws = gensim_eval.evaluate_word_pairs(v_g, v_s, '../eval_data/SCWS_ratings.txt', context_eval=True)
    qualitative_eval(sense_vectors, word_lists)
#model.wv.accuracy('eval_data/questions-words.txt')

