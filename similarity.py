#!/usr/bin/env python
# -*- coding: UTF-8 -*-

## File similarity.py

import numpy as np
import scipy
from scipy import stats, spatial
from operator import itemgetter, attrgetter

import sklearn
from sklearn.metrics.pairwise import cosine_similarity

def indice(liste_sim, autre_liste):
    ## Pour chaque élement de la liste de similarité, on met son indice+1 (= son rang) dans la liste de rang
    rang = []
    liste_tri = sorted(autre_liste)
    for i in range(len(liste_sim)) :
        if liste_sim[i] in liste_tri:
            indice = 1 + liste_tri.index(liste_sim[i])
            rang.append(indice)
    return rang


def paires(embed_dict, simil_dict):
  paires = []
  liste_mots = []
  
  ## pour chaque paire de mots dont on a les embeddings
  for i in simil_dict :
    for j in range(len(simil_dict[i])):
      paires.append((i,simil_dict[i][j][0]))
  return paires

simil_paires = paires(embeddings_dict,similarity_dict)


def corr_spearman(embed_dict, simil_dict):
    ## Calcule la corrélation de Spearman

    cos_rang = []
    humain_rang = []
    cos_score = []
    score_humain = []

    paires_mots = paires(embed_dict, simil_dict)
    liste_mots = []

    ## pour chaque paire de mots dont on a les embeddings
    for i in range(len(paires_mots)) :

        if paires_mots[i][0] in embed_dict and paires_mots[i][1] in embed_dict :
            ## récupération du score donné par les humains
            for j in range(len(simil_dict[paires_mots[i][0]])):
                if simil_dict[paires_mots[i][0]][j][0] == paires_mots[i][1]:
                    score_humain.append(simil_dict[paires_mots[i][0]][j][1])
                    mot1, mot2 = paires_mots[i][0], paires_mots[i][1]

                    ## similarité cosinus
                    cos_score.append(1 - spatial.distance.cosine(embed_dict[mot1], embed_dict[mot2]))

    ## On calcule le rang de chaque élément de la liste tout en triant le rang des éléments.
    ## Puisque le coefficient de Spearman calcule le coefficient de corrélation sur les valeurs de rang des données
    cos_rang = indice(cos_score, cos_score)
    humain_rang = indice(score_humain, score_humain)

    ## corrélation de Spearman
    #spearman = np.corrcoef(humain_rang,cos_rang)
    spearman = scipy.stats.spearmanr(humain_rang, cos_rang,nan_policy='omit')

    return spearman

  
sp = corr_spearman(embeddings_dict,similarity_dict)
print("Corrélation de Spearman :",sp)

def cosine_similarity(embed_dict,simil_dict,word_pairs):
  new_simil_dict = {}
  cos_list = []
  for word1,word2 in word_pairs:
    if (word1 in embed_dict.keys()) and (word2 in embed_dict.keys()):
      cos = cosine_similarity(embed_dict[word1],embed_dict[word2])
    cos_list.append(cos)
