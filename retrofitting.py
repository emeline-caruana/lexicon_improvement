#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import re
import torch
import nltk
from nltk.corpus import wordnet as wn
nltk.download('omw')

from data import vect_dict, embed_dict


print(wn.synsets('dog'))
print(wn.synset('dog.n.01').hypernyms())
print(wn.synset('dog.n.01').hyponyms())

"""
def get_synsets(word,lang):
    ## Méthode pour récupérer tous les mots en relation avec celui donnée en argument
    return wn.synsets(word,lang=lang)"""

print("CHIEN",wn.synsets('chien',lang='fra'))

def get_hypernyms(word,lang):
    synsets = wn.synsets(word,lang=lang)
    if synsets != [] :
        for synset in synsets:
            return synset.hypernyms()

print("HYPERNYMS",get_hypernyms('dog','eng'))

def get_hyponyms(word,lang):
    synsets = wn.synsets(word,lang=lang)
    if synsets != [] :
        for synset in synsets:
            return synset.hyponyms()

print("HYPONYMS",get_hyponyms('dog','eng'))

def get_lemma(synsets):
    lemmas = []
    for synset in synsets:
        lemmas.append(synset.lemma_names())
    return lemmas

print("LEMMAS",get_lemma(get_hypernyms('dog','eng')))
