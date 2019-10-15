from gensim.models import Word2Vec
import gensim.downloader as api
import pickle
from time import time
import logging
import os

logging.basicConfig(filename="./logs/log1", format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

#dataset = api.load("text8")
#t = time()
#basic_model = Word2Vec(dataset, size=200, window=5, min_count=1, workers=4)
#print('Time to train: {} mins'.format(round((time() - t) / 60,
#                                                          2)))
#basic_model.save('runs/word2vec_further_{}.model'.format("base"))


#prefixed = [filename for filename in os.listdir('./data/') if filename.startswith("mcc_")]
# for chronological training
prefixed = ['mcc_1985_2009.list', 'mcc_2009_2013.list', 'mcc_2013_2016.list', 'mcc_2016_2019.list']
print(prefixed)
for time_bin in prefixed:
    with open('data/{}'.format(time_bin), 'rb') as f:
        text = pickle.load(f)
    print("Data loaded...")
    print(time_bin)

    ########### EMB per Bucket (EMB-B)
    #t = time()
    #model = Word2Vec(text, size=200, window=5, min_count=1, workers=4)
    #print('Time to train: {} mins'.format(round((time() - t) / 60,
    #                                                      2)))
    #model.save('runs/emb_b_{}.model'.format(time_bin))

    ########### EMB per Bucket pretrained
    #t = time()
    #basic_model = Word2Vec.load("runs/word2vec_further_base.model")
    #basic_model.train(text, total_examples=len(text), epochs=basic_model.epochs)
    #print('Time to train: {} mins'.format(round((time() - t) / 60,
    #                                                      2)))
    #basic_model.save('runs/emb_b_{}_further.model'.format(time_bin))

    #model = basic_model

    ########### EMB chronologically trained (EMB-C)
    t = time()
    if time_bin == "mcc_1985_2009.list":
        model = Word2Vec(text, size=200, window=5, min_count=1, workers=4)
    else:
        model.train(text, total_examples=len(text), epochs=model.epochs)
    print('Time to train: {} mins'.format(round((time() - t) / 60, 2)))
    model.save('runs/emb_c_{}.model'.format(time_bin))

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
