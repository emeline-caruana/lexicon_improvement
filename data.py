#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# File data.py

import nltk
from nltk.corpus import wordnet as wn

## variables globales
vocabulary, similarity_dict, embeddings_dict = [], {}, {}

def read_data(lang,type='embeds'):
    vectors = {}

    if lang == 'fra':
        if type != 'embeds':
            similarity_dict.clear()
            file = "/corpus_retrofitting_algo/datasets/rg65_french.txt"
        else:
            embeddings_dict.clear()
            file = "/corpus_retrofitting_algo/word_embeddings/vecs100-linear-frwiki/vecs100-linear-frwiki"
    else:
        if type != 'embeds':
            similarity_dict.clear()
            file = "c/orpus_retrofitting_algo/datasets/ws353.txt"
        else:
            embeddings_dict.clear()
            file = "/corpus_retrofitting_algo/word_embeddings/vectors_datatxt_250_sg_w10_i5_c500_gensim_clean/vectors_datatxt_250_sg_w10_i5_c500_gensim_clean"
    with open(file, encoding='utf-8') as f:
        for line in f:
            l = line.split(" ")
            if (len(l) == 3):                                                   ## retrait du saut de ligne '\n' pour le fichier de similarité
                n = l[2].split('\n')
                l[2] = float(n[0])
                vector = [l[1],l[2]]
                if l[0] not in vectors.keys():                                   ## ajout des vecteurs dans un dictionnaire avec le mot 1 en clé et le vecteur en valeur
                    vectors[l[0]] = [vector]
                    if l[1] not in vectors.keys():                                ## ajout des vecteurs dans un dictionnaire avec le mot 2 en clé et le vecteur [mot 1, valeur] en valeur
                        vectors[l[1]] = [[l[0],float(l[2])]]
                    else :
                        vectors[l[1]].append([l[0],float(l[2])])
                else :
                    vectors[l[0]].append(vector)
                    if l[1] not in vectors.keys():                                ## ajout des vecteurs dans un dictionnaire avec le mot 2 en clé et le vecteur [mot 1, valeur] en valeur
                        vectors[l[1]] = [[l[0],float(l[2])]]
                    else :
                        vectors[l[1]].append([l[0],float(l[2])])


            else :
                del l[len(l)-1]
                vector = [float(val) for val in l[2:len(l)]]                    ## récupération des vecteurs du type [mot 2, valeur] ou du type [valeur1, valeur2,..., valeurX] pour les fichiers d'embeddings

                if l[0] not in vectors.keys():                                  ## ajout des vecteurs dans un dictionnaire avec le mot 1 en clé et le vecteur en valeur
                    vectors[l[0]] = vector
                else :
                    vectors[l[0]].append(vector)

    return vectors


def find_vector(word,dic):
    ## Fonction qui permet de récupérer le vecteur d'un certain mot
    ## return [mot 2, value] pour le cas du dictionnaire vect_dict
    ## return [embeddings] pour le cas du dictionnaire embed_dict
    for key, value in dic.items():
        if word == key:
            return value
    return("Le mot n'a pas été trouvé dans le lexique.")

#print(find_vector("corde", embed_dict))
#print(find_vector("idk", embed_dict))
"""
similarity_dict = read_data("fra","vect")                     ## dictionnaire pour la similarité cosinus

#print(vect_dict['corde'])
#print(similarity_dict)
#print("VECTS",len(vect_dict))

embeddings_dict = read_data("fra")                            ## dictionnaire pour le retrofitting et la tâche d'analyse de sentiments
#print("corde : ", embed_dict['corde'])
#print(embeddings_dict)
#print("EMBEDS",len(embed_dict))


embed_vocab = [el for el in embeddings_dict.keys()]
simil_vocab = [el for el in similarity_dict.keys()]
vocabulary = set(embed_vocab + simil_vocab)
vocabulary = list(vocabulary)
#print("len vocabulary",len(vocabulary))"""
