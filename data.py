#!/usr/bin/env python
# -*- coding: UTF-8 -*-

#import re
#import torch

## variables globales
vocab, vect_dic = [], {}

def read_data(file):
    vectors = {}
    with open(file, encoding='utf-8') as f :
        for line in f:
            l = line.split(" ")
            num = l[2].split()                          ## retrait du saut de ligne '\n'
            vector = [l[1],num[0]]                      ## récupération des vecteurs du type [mot 2, valeur]

            if l[0] not in vectors.keys():              ## ajout des vecteurs dans un dictionnaire avec le mot 1 en clé
                vectors[l[0]] = [vector]
            else :
                vectors[l[0]].append(vector)
    return(vectors)

v_dic = read_data("corpus_retrofitting_algo/datasets/rg65_french.txt")
print(v_dic)

def find_vector(word,dic):
    for key,value in dic.items() :
        if word == key :
            return value
    return("Le mot n'a pas été trouvé dans le lexique.")

print(find_vector("corde",v_dic))
print(find_vector("idk",v_dic))

