import math
import numpy as np

from .file_tools import load_sentences_from_file, write_to_csv

# returns pairs of neighboring input sentences (or None if fewer than 2 sentences)
# there will always be n-1 pairs for n sentences
def create_pairs(sentences):
    if len(sentences) < 2:
        return None
    pairs = [(sentences[i-1], sentences[i]) for i in range(1,len(sentences))]
    return pairs

# PURPOSE
# Get numeric tuples representing the pairs genterated by
# nsp_score_tools.createpairs().
# SIGNATURE
# get_pairs_numeric :: List[String] => List[Tuple]
def get_pairs_numeric(sentences):
    return [(x - 1, x) for x in range(1, len(sentences))]

# PURPOSE
# Get the predictor confidence for each sentence pair in a set of pairs.
# SIGNATURE
# get_nsp_confidence :: List[List], NSPredictor, Tensor
def get_nsp_confidence(pairs, predictor, predictions=None):
    _, weights = predictions if predictions else predictor.predict(pairs)
    confidences = [100 * w[0] for w in weights]
    return confidences

# PURPOSE
# Given the confidences for all pairs in a document, return the average confidence
# for all pairs in the document.
# SIGNATURE
# get_avg_nsp_confidence :: List<Float> => Float
def get_avg_nsp_confidence(confidences):
    return np.mean(confidences)

# returns total number of valid sentence pairs, determined by given predictor
def count_valid(pairs, predictor, predictions=None):
    preds = predictions[0] if predictions else predictor.predict(pairs)[0]
    preds = np.array(preds)
    total = len(preds) - np.count_nonzero(preds)
    return total

# PURPOSE
# Given a list of pairs, a predictor, and optionally pregenerated predictions,
# return the number of pairs predicted to be invalid.
# SIGNATURE
# count_invalid :: List[List], NSPredictor, Tensor => Integer
def count_invalid(pairs, predictor, predictions=None):
    if not predictions:
        predictions = predictor.predict(pairs)
    preds = predictions[0]
    return len(preds) - count_valid(pairs, predictor, predictions)

# PURPOSE
# Given the mean NSP confidence for the simplified and original documents,
# the number of incoherent pairs in the simplified and original documents,
# and the total pairs in the original document, return the BERNICE score.
# SIGNATURE
# calculate_bernice :: Float, Float, Integer, Integer, Integer => Float
def calculate_bernice(mean_simp, mean_orig, inc_simp, inc_orig, total_pairs):
    pair_weight = 100
    doc_weight = 100
    pair_score = calc_pair_score(mean_simp, mean_orig, pair_weight)
    doc_score = calc_doc_score(inc_simp, inc_orig, doc_weight, total_pairs)
    return pair_score + doc_score

# PURPOSE
# Given the mean NSP confidence for the simplified and original documents
# and the weight for the NSP confidence, return the pair score.
# SIGNATURE
# calc_pair_score :: Float, Float, Float => Float
def calc_pair_score(mean_simp, mean_orig, pair_weight):
    return (mean_simp / mean_orig) * pair_weight

# PURPOSE
# Given the number of incoherent pairs in the simplified and original documents
# the weight for the document score, and the total number of pairs in the original
# document, return the document score.
# SIGNATURE
# calc_doc_score :: Integer, Integer, Float, Integer => Float
def calc_doc_score(inc_simp, inc_orig, doc_weight, total_pairs):
    x = calc_x(inc_simp, inc_orig, total_pairs)
    stretch = .16
    return (2 / (1 + math.exp(-x / stretch)) - 1) * doc_weight

# PURPOSE
# Given the number of incoherent pairs in the simplified and original documents
# and the weight for the document score, return the x value to use
# in the modified sigmoid function that calculates the doc score.
# SIGNATURE
# calc_x :: Integer, Integer, Integer => Float
def calc_x(inc_simp, inc_orig, total_pairs):
    return (inc_orig - inc_simp) / (total_pairs - inc_orig + 1)



