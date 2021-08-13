import os
import numpy as np
from .file_tools import *
from .sanity_check_tools import *
from .nsp_score_tools import *

# returns total number of valid sentence pairs, determined by given predictor
def count_valid(pairs, predictor, predictions=None):
    preds = predictions[0] if predictions else predictor.predict(pairs)[0]
    preds = np.array(preds)
    total = preds.size - np.count_nonzero(preds)
    return total

# returns sum of valid transition weights of input pairs with given predictor
def count_valid_with_weights(pairs, predictor, predictions=None):
    _, weights = predictions if predictions else predictor.predict(pairs)
    total = sum([w[0] for w in weights])
    return total

# scores simplification by comparing the average weighted valid pairs
# of the original and simplified input sentences.
# Assumes inputs are lists of pairs of sentences.
def score_simplification(orig, simple, predictor):
    simple_score = count_valid_with_weights(simple, predictor) / len(simple)
    complex_score = count_valid_with_weights(orig, predictor) / len(orig)
    score = 100 * simple_score / complex_score
    return score
    
# returns pairs of neighboring input sentences (or None if fewer than 2 sentences)
# there will always be n-1 pairs for n sentences
def create_pairs(sentences):
    if len(sentences) < 2:
        return None
    pairs = [(sentences[i-1], sentences[i]) for i in range(1,len(sentences))]
    return pairs

# returns a list of sentences read from a file.
# assumes file seperates sentences by line already
def load_sentences_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = f.read().splitlines()
    return sentences

# PURPOSE
# Create a csv containing confidences for all pairs in the given input directory.
# SIGNATURE
# score_directory :: String, String => None
def score_directory(input_dir, output_dir):
    files = os.listdir(input_dir)
    for file in files:
        print('**************************SCORING ' + str(file) + ' **************************')
        fpath = os.path.join(input_dir, file)
        sentences = load_sentences_from_file(fpath)
        pairs = create_pairs(sentences)
        scores = get_nsp_confidence(pairs, predictor)
        pairs_numeric = get_pairs_numeric(sentences)
        fields = ['PairID', 'Pair' 'Confidence']
        out_name = file[:-4] + '.csv'
        out_path = os.path.join(output_dir, out_name)
        write_to_csv(out_path, fields, zip(pairs_numeric, pairs, scores))
        



        
        
