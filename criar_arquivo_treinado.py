import nltk
from nltk.stem.lancaster import LancasterStemmer
nltk.download('punkt')
stemmer = LancasterStemmer()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy
import numpy as np
import tflearn
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.style as sty

import matplotlib.pyplot as plt

import random
import json
import pickle

#with open("arquivo.json") as file:
#dados = '/var/www/chatbot_flask/json/arquivo.json'
dados = "/home/jardelsewo.seed/Documentos/arquivos_chatbot_seed/json/arquivo.json"

def criar_arquivo_treinado():

    with open(dados) as file:
#    with open("arquivo_criado_do_bd.json") as file:

        data = json.load(file)

    #try:
    #    with open("data.pickle", "rb") as f:
    #        words, labels, training, output = pickle.load(f)
    #except:

    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["arquivo"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)
        f.close()


    #tensorflow.reset_default_graph()
    tf.compat.v1.reset_default_graph()

    net = tflearn.input_data(shape=[None, len(training[0])])
    #net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 16)
    #net = tflearn.fully_connected(net, 32)
    #net = tflearn.fully_connected(net, 64)
    #net = tflearn.fully_connected(net, 128)
    #net = tflearn.fully_connected(net, 256)
    #net = tflearn.fully_connected(net, 128)
    #net = tflearn.fully_connected(net, 64)
    #net = tflearn.fully_connected(net, 32)
    net = tflearn.fully_connected(net, 16)
    #net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)
    #net = regression(net, optimizer='adam', loss='categorical_crossentropy')

    model = tflearn.DNN(net)

    # simple early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)


    #try:
    #    model.load("model.tflearn")
    #except:
    model.fit(training, output, n_epoch=1050, batch_size=8, show_metric=True)
    #model.fit(training, output, n_epoch=1050, batch_size=8, show_metric=True, callbacks=early_stopping_cb)

    #model.fit(training, output, n_epoch=1000, batch_size=100, show_metric=True)
    #model.save("/var/www/chatbot_flask/modelos_tflearn_salvos/model.tflearn")
    model.save("/home/jardelsewo.seed/Documentos/arquivos_chatbot_seed/modelos_tflearn_salvos/model.tflearn")


criar_arquivo_treinado()
