#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import re
import torch
import nltk
from nltk.corpus import wordnet as wn
nltk.download('omw')
nltk.download('wordnet')  # utilisation de WOLF via NLTK wordnet

from data import vect_dict, embed_dict, vocab

DICT_NEIGBHORS = {}
DICT_HYPERNYMS = {}
DICT_HYPONYMS = {}

def get_synsets(word,lang):
    ## Méthode pour récupérer tous les mots en relation avec celui donnée en argument
    return wn.synsets(word,lang=lang)

#print("CHIEN",get_synsets('chien',lang='fra'))

def lemma(synsets,lang):
    ## Méthode pour récupérer uniquement les mots des synstes et pas 'word.n.01' par exemple
    lemmas,list_lemmas = [],[]
    for synset in synsets:
      lemmas = synset.lemma_names(lang)
      for lemma in lemmas:
        list_lemmas.append(lemma)
    return list_lemmas

#print("LEMMAS",lemma(get_synsets('chien','fra'),'fra'))

def get_hypernyms(word,lang):
    ## Méthode pour récupérer tous les mots en relation d'hypernymie avec celui donnée en argument
    synsets = get_synsets(word,lang=lang)
    hyp = []
    if synsets != [] :
        for synset in synsets:
            hyp += synset.hypernyms()
    return hyp

#print("HYPERNYMS",get_hypernyms('chien','fra'))

def get_hyponyms(word,lang):
    ## Méthode pour récupérer tous les mots en relation d'hyponymie avec celui donnée en argument
    synsets = get_synsets(word,lang=lang)
    hyp = []
    if synsets != [] :
        for synset in synsets:
            hyp += synset.hyponyms()
    return hyp

#print("HYPONYMS",get_hyponyms('chien','fra'))

def neighbors(word,lang,rel='neighb',dict_neighb={}):
    ## Méthode pour récupérer les voisins d'un certain mot en fonction du type de relation donnée en argument ou

    # Récupération des synsets avec le type de relation précisé ou non
    if rel == 'hyponym':
        synsets = get_hyponyms(word,lang)
    elif rel == 'hypernym':
        synsets = get_hypernyms(word,lang)
    else:
        synsets = get_synsets(word,lang)

    # Ajout de la liste de tous les mots appartenant au synset à un dictionnaire dont la clé est le mot donné en argument et la valeur est une liste de mots voisins
    if synsets != [] :
        if (word not in dict_neighb.keys()) or (dict_neighb == {}):
            dict_neighb[word] = list(set(lemma(synsets,lang)))
        else :
            dict_neighb[word].append(list(set(lemma(synsets,lang))))
    return dict_neighb


#print("NEIGHB",neighbors('chien','fra',DICT_NEIGBHORS))
#print("NEIGHB HYPONYM",neighbors('chien','fra','hyponym',DICT_HYPONYMS))
#print("NEIGHB HYPERNYM",neighbors('chien','fra','hypernym',DICT_HYPERNYMS))

def retrofit(num_iter,vocab,word_dict,lang,relation='neighb'):
  #vocab_size = len(vocab)
  vocabulary = vocab.intersection(set(word_dict.keys()))
  vectors_dict = word_dict

  for iter in range(num_iter):
    for word in vocabulary:
      sum = 0

      if word in word_dict.keys():
        word_vect = word_dict[word]
        #print("word vect",word_vect)
      else : word_vect = []
      list_neighb = neighbors(word,lang,relation)
      num_neighb = len(list_neighb)

      if list_neighb != [] and num_neighb != 0:
        word_vect += word_dict[word]*num_neighb
        for neighb in list_neighb:
          word_vect += vectors_dict[neighb]
          #print("word vect 2",word_vect)
        vectors_dict[word] = [ word_vect[i]/(2*num_neighb) for i in range(len(word_vect)) ]

  return vectors_dict


new_vectors = retrofit(1,vocab,embed_dict,'fra')
print("TEST RETROFIT",new_vectors['chien'])
