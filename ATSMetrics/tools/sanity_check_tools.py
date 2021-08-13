# Tools for use in experiments with control text.

import os
from .nsp_score_tools import *

# PURPOSE
# Returns a list of tuples [(from_sentences1, from_sentences2)...].
# This creates a garbled text that can be used as a negative control
# in next sentence prediction experiements.
# SIGNATURE
# interleave_sentences :: List[String], List[String] => List[Tuple]
def interleave_sentences(sentences1, sentences2):
    if (len(sentences1) > len(sentences2)):
        return interleave_sentences(sentences2, sentences1)
    sentences2_cut = sentences2[:len(sentences1)]
    combined = list(zip(sentences1, sentences2_cut))
    return combined

# PURPSOE
# Calculates the weighted total for a set of predictions.
# SIGNATURE
# calculated_weighted_total :: List[Tuple], NSPredictor, Tensor => Float
def calculated_weighted_total(pairs, predictor, predictions):
    weighted_total = count_valid_with_weights(pairs, predictor, predictions)
    weighted_total = 100 * weighted_total / len(pairs)
    return weighted_total

# PURPOSE
# Create a dictionary with relevant information about performance on an 
# input set of known coherent sentence paris.
# SIGNATURE
# coherent_text_report :: List[Tuple], NSPredictor, Tensor => Dictionary
def coherent_text_report(pairs, predictor, predictions=None):
    results = {}
    predictions = predictor.predict(pairs) if not predictions else predictions
    results['correct'] = count_valid(pairs, predictor, predictions)
    results['total'] = len(pairs)
    results['percent_correct'] = 100 * results['correct'] / len(pairs)
    results['weighted_total'] = calculated_weighted_total(pairs, predictor, predictions)
    return results

# PURPOSE
# Count the number of times the system correctly predicted a sentence pair was
# incoherent.
# SIGNATURE
# count_valid_for_incoherent :: List[Tuple], NSPredictor, Tensor => Float
def count_valid_for_incoherent(pairs, predictor, predictions):
    total = len(pairs)
    false_pos = count_valid(pairs, predictor, predictions)
    return total - false_pos

# PURPOSE
# Create a dictionary with relevant information about performance on an 
# input set of known incoherent sentence paris.
# SIGNATURE
# coherent_text_report :: List[Tuple], NSPredictor, Tensor => Dictionary
def incoherent_text_report(pairs, predictor, predictions=None):
    results = {}
    predictions = predictor.predict(pairs) if not predictions else predictions
    results['correct'] = count_valid_for_incoherent(pairs, predictor, predictions)
    results['total'] = len(pairs)
    results['percent_correct'] = 100 * results['correct'] / len(pairs)
    results['weighted_total'] = 100 - calculated_weighted_total(pairs, \
        predictor, predictions)
    return results

# PURPOSE
# Get the predictor confidence for each sentence pair in a set of pairs.
# SIGNATURE
# get_nsp_confidence :: List[List], NSPredictor, Tensor
def get_nsp_confidence(pairs, predictor, predictions=None):
    _, weights = predictions if predictions else predictor.predict(pairs)
    confidence = [100 * w[0] for w in weights]
    return confidence

# PURPOSE
# Interleave sentences from two files and write them to a new scrambled file.
# SIGNATURE
# interleave_files :: String, String, String => None
def interleave_files(fpath1, fpath2, out_path):
    if os.path.exists(out_path):
        print("File already exists. Move or delete the file.")
        return
    with open(fpath1, 'r', encoding='utf-8') as f1:
        with open(fpath2, 'r', encoding='utf-8') as f2:
            lines1 = f1.read().splitlines()
            lines2 = f2.read().splitlines()
            n1 = len(lines1)
            n2 = len(lines2)
            for i in range(min(n1, n2)):
                with open(out_path, 'a', encoding='utf-8') as o:
                    o.write(lines1[i])
                    o.write('\n')
                    o.write(lines2[i])
                    o.write('\n')

# PURPOSE
# Given a list of pairs, and a predictor or a list of predictions, 
# return a list of confidences on each next sentence pair.
# SIGNATURE
# get_pairs_confidences :: List[Tuple(String)], NSPredictor, Tensor => List
def get_pairs_confidences(pairs, predictor, predictions=None):
    return get_nsp_confidence(pairs, predictor, predictions)

# PURPOSE
# Get numeric tuples representing the pairs genterated by
# nsp_score_tools.createpairs().
# SIGNATURE
# get_pairs_numeric :: List[String] => List[Tuple]
def get_pairs_numeric(sentences):
    return [(x - 1, x) for x in range(1, len(sentences))]


