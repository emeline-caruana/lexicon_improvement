#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import allocine
import regex as re
import nltk
nltk.download("movie_reviews") ##corpus anglais de critiques de films

import torch
import numpy as np
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from keras import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Embedding, Lambda

from data import *
from retrofitting import *

embed_size = len(embeddings_dict["and"])  ## récupération du nombre de features d'un mot du vocabulaire pour créer la matrice d'embeddings

def get_data_eng_sentiment(typ="train"):
    if typ == "train" :
        file = "corpus_retrofitting_algo/datasets/stanford_sentiment_analysis/stanford_raw_train.txt"
    elif typ == "dev":
        file = "corpus_retrofitting_algo/datasets/stanford_sentiment_analysis/stanford_raw_dev.txt"
    else:
        file = "corpus_retrofitting_algo/datasets/stanford_sentiment_analysis/stanford_raw_test.txt"
    critics = []

    with open(file, encoding='utf-8') as f:
        corpus_vocab = []
        for line in f:
            ## chaque ligne de type "1 | -1" + "text"
            #print(line)
            if re.match(r'\d\s[A-Za-z]+', line):
                l = line.split(" ")
                #print(l)
                sentiment = l[0]
                text = []
                for i in range(1,len(l)):
                    text.append(l[i])
                    if l[i] not in corpus_vocab :
                        corpus_vocab.append(l[i])
                critics.append((sentiment," ".join(text)))
    return critics,corpus_vocab

#print(critics_eng)

def fit_data(critics):
    X,Y = [],[]
    for sent,text in critics :
        X.append(text)
        Y.append(sent)

    tokenizer = Tokenizer(lower=True,split=' ')
    tokenizer.fit_on_texts(X)
    X = tokenizer.texts_to_sequences(X)
    X = pad_sequences(X)
    return X,Y

def get_embedding_mat(embed_dict,corpus_vocab):
    ## Fonction de création de matrice d'embeddings
    matrix = np.zeros((len(corpus_vocab),embed_size))
    #print(matrix.shape) # (10538, 99)
    for i,word in enumerate(corpus_vocab):
        vector = [0]*embed_size
        if (word in embed_dict.keys()) and (len(embed_dict[word]) == embed_size):
            ## On ne récupère que les vecteurs des mots qui ont la bonne taille et qui font partie du dictionnaire des valeurs pré entrainées
            vector = embed_dict[word]
            #print(len(vector))
        matrix[i] = vector
    return matrix

#train_critics_eng, train_corpus_vocab = get_data_eng_sentiment("train")
#X_train, Y_train = fit_data(train_critics_eng)

#embedding_matrix = get_embedding_mat(embed_dict,vocabulary)


MLP_model = MLPClassifier(hidden_layer_sizes=(100,),activation='tanh',alpha=0.001,solver='adam',max_iter=5000,n_iter_no_change=5)
#MLP_model.fit(X_train,Y_train)

def get_info_model(X,Y):
    print("preds : ",MLP_model.predict(X))
    print("preds probas: ",MLP_model.predict_proba(X))
    print("score : ",MLP_model.score(X,Y))
    print("accuracy : ",accuracy_score(Y,MLP_model.predict()))
    print("loss : ",MLP_model.loss_)
"""
model = Sequential()
model.add(Embedding(len(corpus_vocab), embed_size, weights=[embedding_matrix])) 
model.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(embed_size,)))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
"""