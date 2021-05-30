####################################
Pour le français:
####################################
#------ word-embeddings pré-entraînés ---------#

fichier vecs100-linear-frwiki.zip

- entraînement sur un dump de wikipedia
  (frwiki-20140804-corpus.xml.bz2 téléchargé [ici](http://linguatools.org/tools/corpora/wikipedia-monolingual-corpora/)) 
- preprocessing:
    -`xml2txt.pl` pour enlever les balises xml
    - sxpipe-light pour tokeniser   ( `perl ~/installation/melt-2.0b7/sxpipe-melt/segmenteur.pl < frwiki_raw.txt > frwiki_tokenized.txt` )
    - 650,353,499 tokens (3.6 Gb en texte brut)
- vecs100:  `./word2vec -train frwiki_tokenized.txt -output vecs50 -threads 2 -min-count 100 -cbow 0 -negative 10`
- on utilise la taille de fenêtre par défaut (window=5),ce qui correspond à une taille de fenêtre maximale de 11 (5 mots de chaque côté de la cible); nombre d'itérations par défaut (5)

Vous pouvez aussi utiliser d'autres word embeddings, comme 
# embeddings fasttext
cc.fr.300.vec (1.2G)
voir https://fasttext.cc/docs/en/crawl-vectors.html

#------ ressources sémantiques ---------#
WOLF : https://gforge.inria.fr/frs/download.php/file/33496/wolf-1.0b4.xml.bz2


#------ datasets d'évaluation ---------#
- similarité lexical : fichier rg65_french.txt



#####################
Pour l'anglais: 
#####################

#------ word-embeddings pré-entraînés ---------#
fichier vectors_datatxt_250_sg_w10_i5_c500_gensim_clean.tar.bz2 

- data: concaténation de plusieurs corpus
    - (générés par le script dans le package word2vec original)
    - cf: https://github.com/imsky/word2vec/blob/master/demo-train-big-model-v1.sh

- `gensim.models.Word2Vec(size = 250, min_count = 500, window=8, sample=1e-3, workers = 8, sg=1, hs=0, negative = 10, iter=5)`

#------ lexique sémantique ---------#
- PPDB : http://paraphrase.org/#/download (utilisez ppdb lexical xl size)
- wordnet 

#------ datasets d'évaluation ---------#
- similarité lexical : fichier ws353.txt
- analyse de sentiments : fichier stanford_sentiment_analysis.tar.gz
  https://nlp.stanford.edu/sentiment/
