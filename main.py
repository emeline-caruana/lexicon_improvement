#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import data
from data import *
from retrofitting import *
from movie_critics import *

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('lang', default=None, help='Langue à tester.')
    args = parser.parse_args()

    # récupération des données des fichiers du corpus

    embeddings_dict, embeddings_vocab = data.read_data(args.lang)
    similiraty_dict, similarity_vocab = data.read_data(args.lang,"vect")

    vocabulary = set(embeddings_vocab + similarity_vocab)
    vocabulary = list(vocabulary)

    # tâche de similarité avant retrofitting

    # analyse de sentiments avant retrofitting
    embedding_matrix = get_embedding_mat(embeddings_dict,vocabulary)
