import gensim.downloader as api
from gensim.models import Word2Vec
import pickle
from time import time
import logging
import os

t = time()
logging.basicConfig(filename="further_log1", format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
dataset = api.load("text8")
basic_model = Word2Vec(dataset, size=200, window=5, min_count=1, workers=4)
print('Time to build vocab: {} mins'.format(round((time() - t) / 60,
                                                          2)))
basic_model.save('runs/word2vec_further_{}.model'.format("base"))


prefixed = [filename for filename in os.listdir('./data/') if filename.startswith("processed_text")]
print(prefixed)
for time_bin in prefixed:
    with open('data/{}'.format(time_bin), 'rb') as f:
        text = pickle.load(f)
    print("Data loaded...")

    t = time()
    basic_model.train(text, total_examples=len(text), epochs=basic_model.epochs)
    print('Time to build vocab: {} mins'.format(round((time() - t) / 60,
                                                          2)))
    basic_model.save('runs/word2vec_further_{}.model'.format(time_bin))
    print("Vocabulary size: {0}".format(len(basic_model.wv.vocab)))

    basic_model.wv.accuracy('tmp/questions-words.txt')
