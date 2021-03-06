# File movies_critics.py

import re
import csv
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
from keras.layers import Dense, Embedding, Lambda, LSTM, Dropout, Activation

embed_size = len(embeddings_dict["and"])  ## récupération du nombre de features d'un mot du vocabulaire pour créer la matrice d'embeddings

def get_data_eng_sentiment(typ):
    ## récupération des données pour l'analyse de sentiments en anglais
    ## fichiers de stanford
    if typ == "train" :
        file = "stanford_raw_train.txt"
    elif typ == "dev":
        file = "stanford_raw_dev.txt"
    else:
        file = "stanford_raw_test.txt"
    critics = []

    with open(file, encoding='utf-8') as f:
        corpus_vocab = []
        for line in f:
            ## chaque ligne de type "1 | -1" + "text"
            if re.match(r'\d\s[A-Za-z]+', line):
                l = line.split(" ")
                #print(l)
                sentiment = l[0]
                text = []
                for i in range(1,len(l)):
                    text.append(l[i])
                critics.append((sentiment," ".join(text)))
    return critics

def get_data_fra_sentiment(typ="train"):
    ## récupération des données pour l'analyse de sentiments en français
    ## fichiers de allociné, corpus créé par Théophile Blard
    file_table = []
    if typ == "train" :
        file = "train.csv"
    elif typ == "dev":
        file = "valid.csv"
    else:
        file = "test.csv"
    critics = []

    with open(file, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            file_table.append(row)
        f.close()

    for i in range(len(file_table)):
        critics.append((file_table[i][3],file_table[i][2]))
    return critics


i2w = list(vocab)
w2i = {w: i for i, w in enumerate(i2w)}


def fit_data(critics):
    ## fonction pour que le modèle soit entraîné sur les données
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

embedding_matrix = get_embedding_mat(embeddings_dict,vocabulary)

train_critics_eng = get_data_eng_sentiment("train")
X_train, Y_train = fit_data(train_critics_eng) 

dev_critics_eng = get_data_eng_sentiment("dev")
X_dev, Y_dev = fit_data(dev_critics_eng) 

train_critics_fra = get_data_fra_sentiment("train")
X_train_f, Y_train_f = fit_data(train_critics_fra) 

# MLP avec sklearn
MLP_model = MLPClassifier(hidden_layer_sizes=(100,),activation='tanh',alpha=0.001,solver='adam',max_iter=5000,n_iter_no_change=5)


def fit_and_predict(X,Y):
    MLP_model.fit(X,Y)
    print("preds : ",MLP_model.predict(X))
    print("preds probas: ",MLP_model.predict_proba(X))
    print("score : ",MLP_model.score(X,Y))
    print("accuracy : ",accuracy_score(Y,MLP_model.predict(X)))
    print("loss : ",MLP_model.loss_)

fit_and_predict(X_train,Y_train)
fit_and_predict(X_dev,Y_dev)

# MLP avec Keras
model = Sequential()

def get_model(embedding_matrix, trainable) :
    model.add(Embedding(len(vocabulary), embed_size, weights=[embedding_matrix], trainable=trainable))
    model.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(embed_size,len(vocabulary))))
    model.add(Dropout(0.2))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    print(model.summary())

def train_and_fit(X1,Y1,X2,Y2) : 
    model.fit(X1, Y2, batch_size=30, epochs = 5,  verbose = 5)
    score,acc = model.evaluate(X2, Y2, verbose=2, batch_size=30)
    print("Score : ",score)
    print("Accuracy : ",acc)

get_model(embedding_matrix,True)
train_and_fit(np.array(X_train),np.array(Y_train),np.array(X_dev),np.array(Y_dev))
