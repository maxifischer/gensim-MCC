from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import string
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
from keras.models import Sequential
from keras import layers
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
import pickle
from gensim.models import KeyedVectors
import os


pd.set_option('display.max_colwidth', -1)
K.set_session(K.tf.Session(config=K.tf.ConfigProto(device_count={"CPU": 16}, intra_op_parallelism_threads=16, inter_op_parallelism_threads=16)))
os.environ['MKL_NUM_THREADS'] = '16'
os.environ['GOTO_NUM_THREADS'] = '16'
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['openmp'] = 'True'


def preprocess():
    df  = pd.read_csv('/san2/data/websci/mcc/all_docs.csv')
    df = df[['PY', 'content']]
    df = df.dropna()

    df['PY'] = df['PY'].astype(int)
    print(df.groupby('PY').count())

    # binning decided on evening out bucket sizes
    df['binned'] = pd.cut(df['PY'], [1985, 2009, 2013, 2016, 2019])
    #print(df)

    df = df[(df['PY']<2010) | (df['PY']>2016)]
    print(df.groupby('PY').count())

    df['y'] = np.where(df['PY']<2010, 0, 1)
    print(df.groupby(['y']).count())

    X = df['content']
    y = df['y']

    print(X[:10])

    # abstract preprocessing
    prepr_abstracts = []
    #copyright_stamps = ['c', 'elsevier', 'ltd', 'all', 'rights', 'reserved']
    stop_words = set(stopwords.words('english'))
    all_abstracts = X.tolist()
    for prepr in tqdm(all_abstracts):
        prepr = prepr.lower()
        prepr = re.sub(r'\d+', '', prepr)
        prepr = prepr.translate(str.maketrans('','', string.punctuation))
        prepr = prepr.strip()
        prepr = word_tokenize(prepr)
        prepr = [i for i in prepr if not i in stop_words]
        #prepr = [tok2idx[j] for j in prepr]
        prepr_abstracts.append(prepr)

    vocab = list(set([word for abstract in prepr_abstracts for word in abstract]))
    tok2idx = {k: v for v, k in enumerate(vocab)}
    idx2tok = {v: k for v, k in enumerate(vocab)}
    vocab_size = len(vocab)
    print(vocab_size)

    idx_abstracts = []
    for pre in tqdm(prepr_abstracts):
        idx_abstracts.append([tok2idx[j] for j in pre])

    X = pd.Series(idx_abstracts)
    print(X[:10])
    max_length = max([len(w) for w in vocab])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    X_train_pad = pad_sequences(X_train, maxlen=max_length)
    X_test_pad = pad_sequences(X_test, maxlen=max_length)

    print(X_train_pad[0])
    print(X_train_pad.shape)
    print(X_test_pad.shape)

    pickle.dump(X_train, open("data/X_train.npy", 'wb'))
    pickle.dump(X_train_pad, open("data/X_train_pad.npy", "wb"))
    pickle.dump(X_test, open("data/X_test.npy", 'wb'))
    pickle.dump(X_test_pad, open("data/X_test_pad.npy", "wb"))
    pickle.dump(y_train, open("data/y_train.npy", 'wb'))
    pickle.dump(y_test, open("data/y_test.npy", "wb"))

    pickle.dump(vocab, open("data/vocab.npy", "wb"))

#preprocess()
X_train_pad = pickle.load(open("data/X_train_pad.npy", "rb"))
X_test_pad = pickle.load(open("data/X_test_pad.npy", "rb"))
y_train = pickle.load(open("data/y_train.npy", "rb"))
y_test = pickle.load(open("data/X_test.npy", "rb"))
vocab = pickle.load(open("data/vocab.npy", "rb"))

tok2idx = {k: v for v, k in enumerate(vocab)}
idx2tok = {v: k for v, k in enumerate(vocab)}
vocab_size = len(vocab)
print(vocab_size)
max_length = max([len(w) for w in vocab])

emb_matrix = KeyedVectors.load_word2vec_format("classify_runs/emb_b_mcc_2016_2019.list.bin").syn0
print(emb_matrix.shape)

########## Classification model definition
embedding_dim = 200

model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim,
                           weights=[emb_matrix],
                           input_length=max_length,
                           trainable=True))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()


def save_history(history):
    import matplotlib.pyplot as plt

    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("results/classify_acc.png")

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("results/classify_loss.png")

############## Train classification task
history = model.fit(X_train_pad, y_train,
                    epochs=10,
                    verbose=2,
                    validation_data=(X_test_pad, y_test),
                    batch_size=300)
loss, accuracy = model.evaluate(X_train_pad, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test_pad, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
save_history(history)
