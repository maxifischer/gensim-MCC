from gensim.models import Word2Vec
import gensim.downloader as api
import pickle
from time import time
import logging
import os

logging.basicConfig(filename="./logs/log1", format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

#dataset = api.load("text8")
#basic_model = Word2Vec(dataset, size=200, window=5, min_count=1, workers=4)
#print('Time to build vocab: {} mins'.format(round((time() - t) / 60,
#                                                          2)))
#basic_model.save('runs/word2vec_further_{}.model'.format("base"))


prefixed = [filename for filename in os.listdir('./data/') if filename.startswith("mcc_")]
print(prefixed)
for time_bin in prefixed:
    with open('data/{}'.format(time_bin), 'rb') as f:
        text = pickle.load(f)
    print("Data loaded...")

    ########### EMB per Bucket (EMB-B)
    t = time()
    model = Word2Vec(text, size=200, window=5, min_count=1, workers=4)
    print('Time to build vocab: {} mins'.format(round((time() - t) / 60,
                                                          2)))
    model.save('runs/emb_b_{}.model'.format(time_bin))
    ###########





    ########### Evaluation
    #print("Vocabulary size: {0}".format(len(model.wv.vocab)))

    model.wv.accuracy('eval_data/questions-words.txt')
    wordsim_clas = model.wv.evaluate_word_pairs('eval_data/wordsim353.tsv')
    wordsim_rel = model.wv.evaluate_word_pairs('eval_data/wordsim_relatedness_goldstandard.txt')
    wordsim_sim = model.wv.evaluate_word_pairs('eval_data/wordsim_similarity_goldstandard.txt')
    #simlex = model.wv.evaluate_word_pairs('eval_data/SimLex-999.txt', dummy4unknown=True)

    #print("Analogy scores: {0}".format(analogy_scores[:-1]))
    print("Wordsim Classic: {0}".format(wordsim_clas))
    print("Wordsim Relatedness: {0}".format(wordsim_rel))
    print("Wordsim Similarity: {0}".format(wordsim_sim))
    #print("SimLex: {0}".format(simlex))
