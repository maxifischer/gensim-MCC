from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import string
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.models import Model, Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras import layers, Input
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import Constant
from keras.utils import to_categorical
from keras import backend as K
import pickle
from gensim.models import KeyedVectors
import os
from sklearn.preprocessing import LabelEncoder


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
        prepr = ' '.join(prepr)
        prepr_abstracts.append(prepr)

    #vocab = list(set([word for abstract in prepr_abstracts for word in abstract]))
    #tok2idx = {k: v for v, k in enumerate(vocab)}
    #idx2tok = {v: k for v, k in enumerate(vocab)}
    #vocab_size = len(vocab)
    #print(vocab_size)

    #idx_abstracts = []
    #for pre in tqdm(prepr_abstracts):
    #    idx_abstracts.append([tok2idx[j] for j in pre])

    #X = pd.Series(idx_abstracts)
    #print(X[:10])
    #max_length = max([len(w) for w in vocab])

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(prepr_abstracts)
    sequences = tokenizer.texts_to_sequences(prepr_abstracts)

    vocab = tokenizer.word_index
    print('Found %s unique tokens.' % len(vocab))

    max_length = max([len(w) for w in vocab])

    X = pad_sequences(sequences, maxlen=max_length, padding='post')
    
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    y = to_categorical(np.asarray(y))

    print('Shape of data tensor:', X.shape)
    print('Shape of label tensor:', y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    
    #tokenizer = Tokenizer()
    #X_train_seq = tokenizer.texts_to_sequences(X_train)
    #X_test_seq = tokenizer.texts_to_sequences(X_test)

    #X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
    #X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

    #print(X_train_pad[0])
    #print(X_train_pad.shape)
    #print(X_test_pad.shape)

    pickle.dump(X_train, open("data/X_train.npy", 'wb'))
    #pickle.dump(X_train_pad, open("data/X_train_pad.npy", "wb"))
    pickle.dump(X_test, open("data/X_test.npy", 'wb'))
    #pickle.dump(X_test_pad, open("data/X_test_pad.npy", "wb"))
    pickle.dump(y_train, open("data/y_train.npy", 'wb'))
    pickle.dump(y_test, open("data/y_test.npy", "wb"))

    pickle.dump(vocab, open("data/vocab.npy", "wb"))

#preprocess()
#X_train_pad = pickle.load(open("data/X_train_pad.npy", "rb"))
#X_test_pad = pickle.load(open("data/X_test_pad.npy", "rb"))
X_train = pickle.load(open("data/X_train.npy", "rb"))
X_test = pickle.load(open("data/X_test.npy", "rb"))
y_train = pickle.load(open("data/y_train.npy", "rb"))
y_test = pickle.load(open("data/y_test.npy", "rb"))
vocab = pickle.load(open("data/vocab.npy", "rb"))

#y_train = to_categorical(np.asarray(y_train), 2)
#y_test = to_categorical(np.asarray(y_test), 2)

print(X_train.shape)
print(y_train.shape)
print(X_train[:5])
print(y_train[:5])

y_train = np.array([i[1] for i in y_train]).astype(int)
y_test = np.array([j[1] for j in y_test]).astype(int)
print(y_train[:5])
print(y_test[:5])

tok2idx = {k: v for v, k in enumerate(vocab)}
idx2tok = {v: k for v, k in enumerate(vocab)}
vocab_size = len(vocab)+1
print(vocab_size)
max_length = max([len(w) for w in vocab])

emb_matrix = KeyedVectors.load_word2vec_format("classify_runs/emb_b_mcc_2016_2019.list.bin").syn0
print(emb_matrix.shape)
print(emb_matrix[:5])
embedding_dim = 200

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for i, word in enumerate(vocab):
    embedding_vector = emb_matrix[i]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
#embedding_matrix = emb_matrix

########## Classification model definition
model = Sequential()
embedding_layer = Embedding(
        vocab_size,
        embedding_dim,
        weights=[embedding_matrix],
        input_length=max_length,
        trainable=False)
model.add(embedding_layer)
#model.add(Flatten())
#model.add(Dense(1, activation='sigmoid'))
#seq_input = Input(shape=(max_length,), dtype='int32')
#embed = embedding_layer(seq_input)
#x = layers.Conv1D(128, 5, activation='relu')(embedded_sequences)
#x = layers.MaxPooling1D(5)(x)
#x = layers.Conv1D(128, 5, activation='relu')(x)
#x = layers.MaxPooling1D(5)(x)
#x = layers.Conv1D(128, 5, activation='relu')(x)
#x = layers.MaxPooling1D(35)(x)  # global max pooling
#x = layers.Flatten()(x)
#x = layers.Dense(128, activation='relu')(x)
#preds = layers.Dense(len(labels_index), activation='softmax')(x)
#model = Model(sequence_input, preds)

model.add(layers.GlobalMaxPooling1D())
#model.add(layers.GlobalMaxPooling1D())
#model.add(layers.Flatten())
#x = layers.Flatten()(x)
#x = layers.Dense(10, activation='relu')(x)
model.add(layers.Dense(1, activation='sigmoid'))
#model = Model(seq_input, preds)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()


def save_history(history):
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
history = model.fit(X_train, y_train,
                    epochs=10,
                    verbose=2,
                    validation_data=(X_test, y_test),
                    batch_size=300)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
save_history(history)
