import allocine
from data import *
from retrofitting import *

import nltk
nltk.download("movie_reviews") ##corpus anglais de critiques de films

import torch
import numpy as np
import sklearn 

from keras import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Embedding, Lambda

embed_size = len(embed_dict["and"])  ## récupération du nombre de features d'un mot du vocabulaire pour créer la matrice d'embeddings

def get_data_eng_sentiment():
  file = "stanford_raw_train.txt"
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

critics_eng, corpus_vocab = get_data_eng_sentiment()
#print(critics_eng)

X,Y = [],[]
for sent,text in critics_eng :
  X.append(text)
  Y.append(sent)

#print("X\n",X)
#print("Y\n",len(Y))

tokenizer = Tokenizer(lower=True,split=' ')
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)
X = pad_sequences(X)
print(X)

def get_embedding_mat(embed_dict,corpus_vocab):
  ## Fonction de création de matrice d'embeddings
  matrix = np.zeros((len(corpus_vocab),embed_size))
  #print(matrix.shape) # (10538, 99)
  for i,word in enumerate(corpus_vocab):
    if (word in embed_dict.keys()) and (len(embed_dict[word]) == embed_dim):
      ## On ne récupère que les vecteurs des mots qui ont la bonne taille et qui font partie du dictionnaire des valeurs pré entrainées
      vector = embed_dict[word]
      #print(len(vector))
    matrix[i] = vector
  return matrix

embedding_matrix = get_embedding_mat(embed_dict,corpus_vocab)


model = Sequential()
model.add(Embedding(len(corpus_vocab), embed_size, weights=[embedding_matrix])) 
model.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(embed_size,)))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

"""
Pour l'analyse de sentiments, voir s'il est intéressant de supprimer les stopswords, d'enlever la ponctuation et de tout mettre en minuscule
Si oui, alors, faire le processing de données de cette manière pour fra et eng
Pour l'eng on a déjà le corpus mais pas d'embedding pré-entraînés ou alors dans le fichier gensim ?
Pour le fra, pas de corpus mais on peut utiliser le movie_reviews version allociné
Création d'un petit réseau de neurones :
 - améliorer les embeddings des mots et catégoriser quel mot va dans quelle sentiment (positif, négatif ou neutre)
"""
