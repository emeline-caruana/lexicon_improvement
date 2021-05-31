#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import re
import torch
from nltk.corpus import wordnet as wn

## variables globales
vocab, vect_dic, embeds_dic = [], {}, {}

def read_data(file):
    vectors = {}
    with open(file, encoding='utf-8') as f:
        for line in f:
            l = line.split(" ")
            if (len(l) == 3):                                          ## retrait du saut de ligne '\n' pour le fichier de similarité
                n = l[2].split('\n')
                l[2] = n[0]
            else :
                del l[len(l)-1]
            vector = [val for val in l[1:len(l)]]                      ## récupération des vecteurs du type [mot 2, valeur] ou du type [valeur1, valeur2,..., valeurX] pour les fichiers d'embeddings

            if l[0] not in vectors.keys():                             ## ajout des vecteurs dans un dictionnaire avec le mot 1 en clé et le vecteur en valeur
                vectors[l[0]] = [vector]
            else :
                vectors[l[0]].append(vector)
    return(vectors)

vect_dic = read_data("corpus_retrofitting_algo/datasets/rg65_french.txt")
#print(vect_dic['corde'])

embed_dic = read_data("corpus_retrofitting_algo/word_embeddings/vecs100-linear-frwiki")
#print("corde : ", embed_dic['corde'])
#print("la : ", embed_dic['la'])

def find_vector(word,dic):
    for key, value in dic.items():
        if word == key:
            return value
    return("Le mot n'a pas été trouvé dans le lexique.")

#print(find_vector("corde", vect_dic))
#print(find_vector("idk", vect_dic))
