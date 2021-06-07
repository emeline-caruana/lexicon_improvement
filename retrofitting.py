#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import re
import torch
import nltk
from nltk.corpus import wordnet as wn
#nltk.download('omw')

from data import vect_dict, embed_dict

VOCABULARY = [w for w in vect_dict.keys()]
VOCAB = set(VOCABULARY)

DICT_NEIGBHORS = {}
DICT_HYPERNYMS = {}
DICT_HYPONYMS = {}

def get_synsets(word,lang):
    ## Méthode pour récupérer tous les mots en relation avec celui donnée en argument
    return wn.synsets(word,lang=lang)

print("CHIEN",wn.synsets('chien',lang='fra'))

def get_hypernyms(word,lang):
    ## Méthode pour récupérer tous les mots en relation d'hypernymie avec celui donnée en argument
    synsets = wn.synsets(word,lang=lang)
    hyp = []
    if synsets != [] :
        for synset in synsets:
            hyp += synset.hypernyms()
    return hyp

print("HYPERNYMS",get_hypernyms('chien','fra'))

def get_hyponyms(word,lang):
    ## Méthode pour récupérer tous les mots en relation d'hyponymie avec celui donnée en argument
    synsets = wn.synsets(word,lang=lang)
    hyp = []
    if synsets != [] :
        for synset in synsets:
            hyp += synset.hyponyms()
    return hyp

print("HYPONYMS",get_hyponyms('chien','fra'))

def get_lemma(synsets):
    ## Méthode pour récupérer uniquement les mots des synstes et pas 'word.n.01' par exemple
    lemmas,list_lemmas = [],[]
    for synset in synsets:
        lemmas = synset.lemma_names()
        for lemma in lemmas:
            list_lemmas.append(lemma)
    return list_lemmas

print("LEMMAS",get_lemma(get_hypernyms('chien','fra')))

def neighbors(word,lang,dict_neighb,rel='neighb'):
    ## Méthode pour récupérer les voisins d'un certain mot
    if rel == 'hypo':
        synsets = get_hyponyms(word,lang)
    elif rel == 'hyper':
        synsets = get_hypernyms(word,lang)
    else:
        synsets = get_synsets(word,lang)

    if synsets != [] :
        if word not in dict_neighb.keys():
            dict_neighb[word] = list(set(get_lemma(synsets)))
        else :
            dict_neighb[word].append(list(set(get_lemma(synsets))))
    return dict_neighb


print("NEIGHB",neighbors('chien','fra',DICT_NEIGBHORS))
print("NEIGHB HYPONYM",neighbors('chien','fra',DICT_HYPONYMS,'hypo'))
