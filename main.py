#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import data
from data import *
from retrofitting import *
from movie_critics import *

import sys
import argparse


if __name__ == "__main__":

    usage = """ Bienvenue dans le projet de retrofitting Juline et Emeline. Avant de commencer, veuillez choisir une langue {'fra'|'eng'}
    """+sys.argv[0]+""" Tout le projet va se faire dans cette langue.\nSi le choix n'est pas fait (ou possible), la langue par défaut est l'anglais.
    """

    parser = argparse.ArgumentParser(usage = usage)
    parser.add_argument('lang', default="eng", help='Langue utilisée')
    args = parser.parse_args()

    # récupération des données des fichiers du corpus

    embeddings_dict = data.read_data(args.lang)
    similiraty_dict = data.read_data(args.lang,"vect")

    embed_vocab = [el for el in embeddings_dict.keys()]
    simil_vocab = [el for el in similarity_dict.keys()]
    vocabulary = set(embeddings_vocab + similarity_vocab)
    vocabulary = list(vocabulary)

    # tâche de similarité avant retrofitting

    sp = corr_spearman(embeddings_dict,similarity_dict)

    # analyse de sentiments avant retrofitting

    embedding_matrix = get_embedding_mat(embeddings_dict,vocabulary)
    
    if args.lang == "eng":
        train_critics = get_data_eng_sentiment("train")
        X_train, Y_train = fit_data(train_critics) 

        dev_critics = get_data_eng_sentiment("dev")
        X_dev, Y_dev = fit_data(dev_critics) 
    else:
        train_critics = get_data_fra_sentiment("train")
        X_train, Y_train = fit_data(train_critics) 

        dev_critics = get_data_fra_sentiment("dev")
        X_dev, Y_dev= fit_data(dev_critics) 

    
    MLP_model = MLPClassifier(hidden_layer_sizes=(100,),activation='tanh',alpha=0.001,solver='adam',max_iter=5000,n_iter_no_change=5)

    
    fit_and_predict(X_train,Y_train)
    fit_and_predict(X_dev,Y_dev)
    # Retrofitting

    new_embeddings_dict = retrofit(1,vocabulary,embeddings_dict,"eng")

    # tâche de similarité après retrofitting

    sp = corr_spearman(new_embeddings_dict,similarity_dict)
    
    
    # analyse de sentiments après retrofitting
    embedding_matrix = get_embedding_mat(new_embeddings_dict,vocabulary)

        if args.lang == "eng":
            train_critics = get_data_eng_sentiment("train")
            X_train, Y_train = fit_data(train_critics) 

            dev_critics = get_data_eng_sentiment("dev")
            X_dev, Y_dev = fit_data(dev_critics) 
        else:
            train_critics = get_data_fra_sentiment("train")
            X_train, Y_train = fit_data(train_critics) 

            dev_critics = get_data_fra_sentiment("dev")
            X_dev, Y_dev= fit_data(dev_critics) 


        MLP_model = MLPClassifier(hidden_layer_sizes=(100,),activation='tanh',alpha=0.001,solver='adam',max_iter=5000,n_iter_no_change=5)


        fit_and_predict(X_train,Y_train)
        fit_and_predict(X_dev,Y_dev)
