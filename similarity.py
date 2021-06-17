import numpy as np
import spacy
from spacy import stats
from operator import itemgetter, attrgetter

from data import *
from retrofitting import *

def indice(liste_sim, autre_liste):
    ## Pour chaque élement de la liste de similarité, on met son indice+1 (= son rang) dans la liste de rang
    rang = []
    liste_tri = sorted(autre_liste)
    for i in range(len(liste_sim)) :
        if liste_sim[i] in liste_tri:
            indice = 1 + list_tri.index(liste_sim[i])
            rang.append(indice)
    return rang

print(indice(similarity_dict,similarity_dict))


def corr_spearman(embed_dict, simil_dict):
    ## Calcule la corrélation de Spearman

    cos_rang = []
    humain_rang = []
    cos_score = []
    score_humain = []

    paires = [ (mot,simil_dict[mot][0]) for mot in simil_dict.keys() ]

    ## pour chaque paire de mots dont on a les embeddings
    for mot in paires :

        if mot[0] in embed_dict and mot[1] in embed_dict :
            ## récupération du score donné par les humains
            score_humain.append(simil_dict[mot][1])

            mot1, mot2 = mot[0], mot[1]
            embdg1 = embed_dict[mot1]
            embdg2 = embed_dict[mot2]

            ## similarité cosinus
            cos_score.append(1 - spatial.distance.cosine(embdg_1, embdg_2))

        ## On calcule le rang de chaque élément de la liste tout en triant le rang des éléments.
        ## Puisque le coefficient de Spearman calcule le coefficient de corrélation sur les valeurs de rang des données
        cos_rang = indice(cos_score, cos_score)
        humain_rang = indice(score_humain, score_humain)

        ## corrélation de Spearman
        spearman = spcipy.stats.spearmanr(humain_rang, cos_rang)

        return(spearman)
