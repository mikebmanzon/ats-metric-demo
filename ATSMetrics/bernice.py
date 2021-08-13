from .toolsv2.bernice_tools import *
from .predictors.NSPredictor import NSPredictor

# PURPOSE
# Given a list of original sentences and a list of simplified sentences,
# calculate and return the BERNICE score for document cohesion.
# The lists may be of different lengths and should be the
# original and simplified version of the same document.
# SIGNATURE
# bernice_score :: List[String], List[String] => Float
def bernice_score(orig_sentences, simplified_sentences):
    orig_pairs = create_pairs(orig_sentences)
    if orig_pairs == None or len(orig_pairs) < 1:
        raise ValueError('No pairs were created. orig_sentences must contain at least two sentences.')
    num_orig_pairs = len(orig_pairs)
    simp_pairs = create_pairs(simplified_sentences)
    predictor = NSPredictor('bert-base-cased')
    orig_predictions = predictor.predict(orig_pairs)
    simp_predictions = predictor.predict(simp_pairs)
    orig_confidences = get_nsp_confidence(orig_pairs, predictor, orig_predictions)
    simp_confidences = get_nsp_confidence(simp_pairs, predictor, simp_predictions)
    orig_mean = get_avg_nsp_confidence(orig_confidences)
    simp_mean = get_avg_nsp_confidence(simp_confidences)
    orig_incoherent = count_invalid(orig_pairs, predictor, orig_predictions)
    simp_incoherent = count_invalid(simp_pairs, predictor, simp_predictions)
    return calculate_bernice(simp_mean, orig_mean, simp_incoherent, orig_incoherent, num_orig_pairs)
