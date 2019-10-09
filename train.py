from gensim.models import Word2Vec
import pickle
from time import time
import logging
import os

logging.basicConfig(filename="log1", format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

prefixed = [filename for filename in os.listdir('./data/') if filename.startswith("processed_text.")]
print(prefixed)
for time_bin in prefixed:
    with open('data/{}'.format(time_bin), 'rb') as f:
        text = pickle.load(f)
    print("Data loaded...")

    t = time()
    model = Word2Vec(text, size=200, window=5, min_count=1, workers=4)
    print('Time to build vocab: {} mins'.format(round((time() - t) / 60,
                                                          2)))
    model.save('runs/word2vec_{}.model'.format(time_bin))
    print("Vocabulary size: {0}".format(len(model.wv.vocab)))

    model.wv.accuracy('tmp/questions-words.txt')
