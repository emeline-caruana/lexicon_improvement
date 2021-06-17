import numpy as np
import spacy
from spacy import stats



def indice_similarité(liste_sim, tri) :
	""" Pour chaque élement de la liste de similarité, on met son indice+1 (= son rang) dans la liste de rang """
	rang = []
	for score in liste_sim :
		index = 1 + tri.index(score)
		rang.append(index)

	return rang
    
def coef_spearman(embeddings, data):
    """Calcule le coefficient de spearman
    Entrée :
        - emdeddings : dictionnaire des embeddings à tester
        - data : dictionnaire des paires de mots et de leur similarité évaluée par les humains
    Sortie :
        coefficient de spearman """

    cos_rang = []
    humain_rang = []
    cos_score = []
    score_humain = []
    paires = list(data.keys())
    liste_mots = []


    """ Pour chaque paire de mots dont on a les embeddings"""
    for elt in paires :
        if elt[0] in embedd and elt[1] in embeddings :
            """ On récupère le score humain """
            score_humain.append(data[elt])

            mot_1, mot_2 = elt[0], elt[1]
            embdg_1 = embeddings[mot_1]
            embdg_2 = embeddings[mot_2]

            """ On calcule la similarité cosinus """
            produit_scalaire = np.dot(embdg_1, embdg_2)
            norme_1 = np.linalg.norm(embdg_1)
            norme_2 = np.linalg.norm(embdg_2)
            cos = produit_scalaire / (norme_1 * norme_2)
            score_cos.append(cos)

        """ On trie les listes de similarité cos et de évalution humaine """
        cos_trie = sorted(cos_score, reverse = True)
        humain_trie = sorted(score_humain, reverse = True)

        """ On calcule le rang de chaque élément de la liste
        Puisque le coefficient de Spearman calcule le coefficient de corrélation sur les valeurs de rang des données """
        cos_rang = indice_similarité(cos_score, cos_trie)
        humain_rang = indice_similarité(score_humain, humain_trie)

        """ On calcule le coefficient de spearman """
        spearman = scipy.stats.spearmanr(humain_rang, cos_rang)

        return(spearman)
