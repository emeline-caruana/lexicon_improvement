#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import nltk
from nltk.corpus import wordnet as wn
nltk.download('omw')
nltk.download('wordnet')  # utilisation de WOLF via NLTK wordnet

from data import similarity_dict, embeddings_dict, vocabulary

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


def get_hypernyms(synsets,lang):
    ## Méthode pour récupérer tous les mots en relation d'hypernymie avec celui donnée en argument
    syn_list = []
    if synsets != [] :
        for syn in synsets:
            syn_list += syn.hypernyms()
    return syn_list


def get_hyponyms(synsets,lang):
    ## Méthode pour récupérer tous les mots en relation d'hyponymie avec celui donnée en argument
    syn_list = []
    if synsets != [] :
        for syn in synsets:
            syn_list += syn.hyponyms()
    return syn_list


def neighbors(word,lang,voc,rel='neighb',list_neighb=[]):
    ## Méthode pour récupérer les voisins d'un certain mot en fonction du type de relation donnée en argument ou
    list_neighb = []
    synsets = get_synsets(word,lang)

    # Ajout de la liste de tous les mots appartenant au synset à un dictionnaire dont la clé est le mot donné en argument et la valeur est une liste de mots voisins
    if rel == "neighb":
        list_lemmas = lemma(synsets,lang)
        for lem in list_lemmas:
            if lem not in list_neighb and lem in vocabulary:
                list_neighb.append(lem)
    else:
        for synset in synsets:
            # Récupération des synsets avec le type de relation précisé ou non
            if rel == 'hyponym':
                list_neighb = lemma(get_hyponyms(synset,lang),lang)
            elif rel == 'hypernym':
                list_neighb = lemma(get_hypernyms(synset,lang),lang)

    return list_neighb


def retrofit(num_iter,vocab,word_dict,lang,relation='neighb'):
    ## Fonction de retrofitting
    ## D'après l'algorithme de Faruqui
    list_vocab = set(vocab).intersection(set(word_dict.keys()))
    new_word_dict = word_dict

    for iter in range(num_iter):

        for word in list_vocab:
            if word in new_word_dict.keys():
                word_vect = new_word_dict[word]
            else : word_vect = []
            #print("wv1",word_vect)
            list_neighb = neighbors(word,lang,vocab,relation)
            num_neighb = len(list_neighb)

            if list_neighb != []:
                #word_vect = word_dict[word] * num_neighb
                #print("wv2",word_vect)
                for neighb in list_neighb:
                    if neighb in new_word_dict.keys():
                        len_vect = len(new_word_dict[neighb])
                        word_vect = [(word_vect[i]+new_word_dict[neighb][i]) for i in range(len_vect)]
                        #print("wv3",word_vect)
                new_word_dict[word] = [ word_vect[i]/(2*num_neighb) for i in range(len(word_vect))]
                #print("new wv",word,new_word_dict[word])

    print("DONE")
    return new_word_dict

#new_embeddigns_dict = retrofit(1,vocabulary,embeddings_dict,"eng")
#print("new\n",len(new))

"""
BATCH_SIZE = 15
new_vectors = {}

for i in range(5):
    batch = vocabulary[i:BATCH_SIZE]
    print(batch)
    set_batch = set(batch)
    new_vectors = retrofit(5,set_batch,embeddings_dict,'fra')
print("TEST AVANT RETROFIT","chien",embeddings_dict["chien"])
print("TEST APRES RETROFIT","chien",new["chien"])"""
