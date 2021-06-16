#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import nltk
from nltk.corpus import wordnet as wn
nltk.download('omw')
nltk.download('wordnet')  # utilisation de WOLF via NLTK wordnet

#from data import similarity_dict, embeddings_dict, vocabulary

def get_synsets(word,lang):
    ## Méthode pour récupérer tous les mots en relation avec celui donné en argument
    return wn.synsets(word,lang=lang)


def lemma(synsets,lang):
    ## Méthode pour récupérer uniquement les mots des synsets et pas 'word.n.01' par exemple
    lemmas,list_lemmas = [],[]
    for synset in synsets:
        lemmas = synset.lemma_names(lang)
        for lemma in lemmas:
            list_lemmas.append(lemma)
    return list_lemmas


def get_hypernyms(word,lang):
    ## Méthode pour récupérer tous les mots en relation d'hypernymie avec celui donnée en argument
    synsets = get_synsets(word,lang=lang)
    hyp = []
    if synsets != [] :
        for synset in synsets:
            hyp += synset.hypernyms()
    return hyp


def get_hyponyms(word,lang):
    ## Méthode pour récupérer tous les mots en relation d'hyponymie avec celui donnée en argument
    synsets = get_synsets(word,lang=lang)
    hyp = []
    if synsets != [] :
        for synset in synsets:
            hyp += synset.hyponyms()
    return hyp


def neighbors(word,lang,rel='neighb',list_neighb=[]):
    ## Méthode pour récupérer les voisins d'un certain mot en fonction du type de relation donnée en argument ou
    list_neighb = []
    # Récupération des synsets avec le type de relation précisé ou non
    if rel == 'hyponym':
        synsets = get_hyponyms(word,lang)
    elif rel == 'hypernym':
        synsets = get_hypernyms(word,lang)
    elif rel == 'synonym':
        synsets = get_synonyms(word,lang)
    else:
        synsets = get_synsets(word,lang)

    # Ajout de la liste de tous les mots appartenant au synset à un dictionnaire dont la clé est le mot donné en argument et la valeur est une liste de mots voisins
    for synset in synsets:
        if (synset not in list_neighb) or (list_neighb == []):
            if synset in vocab:
                list_neighb.append(synset.lemma_names(lang))
    return list_neighb


def retrofit(num_iter,vocab,word_dict,lang,relation='neighb'):
    ## Fonction de retrofitting
    ## D'après l'algorithme de Faruqui
    vocabulary = vocab.intersection(set(word_dict.keys()))
    vectors_dict = word_dict

    for iter in range(num_iter):

        for word in vocabulary:
            if word in vectors_dict.keys():
                word_vect = vectors_dict[word]
            else : word_vect = []

            list_neighb = neighbors(word,lang,relation)
            num_neighb = len(list_neighb)

            if list_neighb != []:
                word_vect = word_dict[word] * num_neighb
                for neighb in list_neighb:
                    if neighb in vectors_dict.keys():
                        word_vect += vectors_dict[neighb]
                        #print("word vect 2",word_vect)
                vectors_dict[word] = word_vect/(2*num_neighb)

    print("DONE")
    return vectors_dict

"""
BATCH_SIZE = 5
vocab_list = list(vocab)
new_vectors = {}

for i in range(5):
    batch = vocab_list[i:BATCH_SIZE]
    print(batch)
    set_batch = set(batch)
    new_vectors.update(retrofit(5,set_batch,embed_dict,'fra'))
print("TEST AVANT RETROFIT",batch[0],embed_dict[batch[0]])
print("TEST APRES RETROFIT",batch[0],new_vectors[batch[0]])
"""