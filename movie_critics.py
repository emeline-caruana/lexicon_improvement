import allocine
from data import *
from retrofitting import *

import allocine

import nltk
nltk.download("movie_reviews") ##corpus anglais de critiques de films


"""
Pour l'analyse de sentiments, voir s'il est intéressant de supprimer les stopswords, d'enlever la ponctuation et de tout mettre en minuscule
Si oui, alors, faire le processing de données de cette manière pour fra et eng

Pour l'eng on a déjà le corpus mais pas d'embedding pré-entraînés ou alors dans le fichier gensim ?

Pour le fra, pas de corpus mais on peut utiliser le movie_reviews version allociné


Création d'un petit réseau de neurones :
 - améliorer les embeddings des mots et catégoriser quel mot va dans quelle sentiment (positif, négatif ou neutre)

"""
