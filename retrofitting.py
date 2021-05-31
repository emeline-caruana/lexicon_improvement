#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import re
import torch
from nltk.corpus import wordnet as wn
from data import vect_dic, embed_dic

print(wn.synsets('dog'))
print(wn.synset('dog.n.01').hypernyms())
print(wn.synset('dog.n.01').hyponyms())

def get_synset(word,lang):
    pass;

def get_word(synsets,lang):
    pass;
