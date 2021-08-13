import numpy as np
import pandas as pd
import os
from itertools import permutations
from .file_tools import load_sentences_from_file, write_csv_row, write_lines_to_file, write_to_csv
from .bernice_tools import *

# PURPOSE
# Create a csv for each file in the given directory containing confidences 
# for all pairs in the file. Files are assumed to have one sentence
# per line.
# SIGNATURE
# score_directory :: String, String, NSPredictor => None
def score_directory(input_dir, output_dir, predictor):
    files = os.listdir(input_dir)
    for file in files:
        print('**************************SCORING ' + str(file) + ' **************************')
        fpath = os.path.join(input_dir, file)
        sentences = load_sentences_from_file(fpath)
        pairs = create_pairs(sentences)
        if pairs == None or len(pairs) < 2:
            print('************** One or fewer Sentences **********************')
            continue
        scores = get_nsp_confidence(pairs, predictor)
        pairs_numeric = get_pairs_numeric(sentences)
        fields = ['PairID', 'Pair', 'Confidence']
        out_name = file[:-4] + '.csv'
        out_path = os.path.join(output_dir, out_name)
        write_to_csv(out_path, fields, zip(pairs_numeric, pairs, scores))

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
            interleaved = interleave_sentences(lines1, lines2)
            interleaved = flatten(interleaved)
            write_lines_to_file(interleaved, out_path)

# PURPOSE
# Helper function to flatten a list of tuples.
# SIGNATURE
# flatten :: List[Tuple] => List
def flatten(list_tup):
    return list(sum(list_tup, ()))

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

# returns sum of valid transition weights of input pairs with given predictor
def count_valid_with_weights(pairs, predictor, predictions=None):
    _, weights = predictions if predictions else predictor.predict(pairs)
    total = sum([w[0] for w in weights])
    return total

# DEPRECATED -- First idea for BERNICE # 
# scores simplification by comparing the average weighted valid pairs
# of the original and simplified input sentences.
# Assumes inputs are lists of pairs of sentences.
def score_simplification(orig, simple, predictor):
    simple_score = count_valid_with_weights(simple, predictor) / len(simple)
    complex_score = count_valid_with_weights(orig, predictor) / len(orig)
    score = 100 * simple_score / complex_score
    return score

# PURPOSE
# Given an original document path, an output directory, start and end indices
# (inclusive), and a predictor, write the pairwise nsp confidence of sentence 
# permutations to the directory, one csv per permutation. 
# The original document is assumed to be one sentence per line.
# The 0th entry represents results for the original text.
# SIGNATURE
# pairwise_nsp_confidence :: String, String, Integer, Interger, NSPredictor,
# Tensor => None
def r_permutations_confidence_to_file(orig_path, out_path, predictor):
    if os.path.exists(out_path):
        print("File already exists. Move or delete the file to do a new "
        + "analysis.")
        return
    sents = load_sentences_from_file(orig_path)
    r_perms = generate_r_permutations_map(len(sents), 2)
    fields = ['pair_id', 'sent1', 'sent2', 'confidence']
    write_csv_row(out_path, fields)
    for r_p in r_perms:
        row = [r_p]
        r_p_sents = [(sents[r_p[0]], sents[r_p[1]])]
        confidence = get_nsp_confidence(r_p_sents, predictor, None)
        pair = (sents[r_p[0]], sents[r_p[1]])
        row += pair
        row += [confidence[0]]
        write_csv_row(out_path, row)

# PURPOSE
# Given a number of items, n, and the size of the desired permutations, r,
# return a list of permutations of n or size r.
# SIGNATURE
# generate_r_permutations_map(n, r) :: Integer, Integer => List[Tuple]
# EXAMPLE
# generate_r_permutations(2, 2) => [(0, 1), [1, 0]]
def generate_r_permutations_map(n, r):
    return list(permutations(range(n), r))

# PURPOSE
# Given precalculated csvs containing confidences for sentence pairs,
# calculate BERNICE.
# SIGNATURE
# calc_bernice_from_csv :: String, String => Float
def calc_bernice_from_csv(orig_fpath, simp_fpath):
    orig_df = pd.read_csv(orig_fpath)
    simp_df = pd.read_csv(simp_fpath)
    orig_confidences = orig_df['Confidence'].values
    simp_confidences = simp_df['Confidence'].values

    num_orig_pairs = len(orig_confidences)
    orig_incoherent = len([x for x in orig_df['Confidence'].values if x < 50])
    simp_incoherent = len([x for x in simp_df['Confidence'].values if x < 50])
    orig_mean = np.mean(orig_confidences)
    simp_mean = np.mean(simp_confidences)
    return calculate_bernice(simp_mean, orig_mean, simp_incoherent, orig_incoherent, num_orig_pairs)

# PURPOSE
# Given precalculated csvs containing confidences for sentences pairs,
# Calculate BERNICE for all files in the orig_dir.
# Output the result to a CSV at output_path.
# SIGNATURE
# calc_bernice_for_dir :: String, String, String => None
def calc_bernice_for_dir(orig_dir, simp_dir, output_path):
    fnames = os.listdir(orig_dir)
    scores = []
    orig_means = []
    simp_means = []
    orig_incs = []
    simp_incs = []
    num_orig_pairs_lst = []
    for file in fnames:
        orig_fpath = os.path.join(orig_dir, file)
        simp_fpath = os.path.join(simp_dir, file)
        orig_df = pd.read_csv(orig_fpath)
        simp_df = pd.read_csv(simp_fpath)
        orig_confidences = orig_df['Confidence'].values
        simp_confidences = simp_df['Confidence'].values

        num_orig_pairs = len(orig_confidences)
        orig_incoherent = len([x for x in orig_df['Confidence'].values if x < 50])
        simp_incoherent = len([x for x in simp_df['Confidence'].values if x < 50])
        orig_mean = np.mean(orig_confidences)
        simp_mean = np.mean(simp_confidences)
        bernice_score = calculate_bernice(simp_mean, orig_mean, simp_incoherent, orig_incoherent, num_orig_pairs)


        scores.append(bernice_score)
        orig_means.append(orig_mean)
        simp_means.append(simp_mean)
        orig_incs.append(orig_incoherent)
        simp_incs.append(simp_incoherent)
        num_orig_pairs_lst.append(num_orig_pairs)
    fields = ['FileName', 'BERNICE_Score', 'orig_mean', 'simp_mean', 'orig_incoherent', 'simp_incoherent', 'num_orig_pairs']
    write_to_csv(output_path, fields, zip(fnames, scores, orig_means, simp_means, orig_incs, simp_incs, num_orig_pairs_lst))

################ DEPRECATED ## This section contains functions used only to calculate confidences of
################ possible permutations of sentence pairs for a given document.
################ The code should be refactored if it is desired to run this experiment again,
################ as the permutations do not need to be recalculated from scratch every time.
################ Instead, use r_permutations to calculate confidences for all possible pairs,
################ then permute the confidences.


# PURPOSE
# Given an original document path, an output directory, start and end indices
# (inclusive), and a predictor, write the pairwise nsp confidence of sentence 
# permutations to the directory, one csv per permutation. 
# The original document is assumed to be one sentence per line.
# The 0th entry represents results for the original text.
# SIGNATURE
# pairwise_nsp_confidence :: String, String, Integer, Interger, NSPredictor,
# Tensor => None
def pairwise_nsp_confidence_to_file(orig_path, out_path, headings, start, \
    end, predictor):
    n_fact = end - start + 1
    perm_map = generate_perm_map(n_fact)
    orig_sentences = load_sentences_from_file(orig_path)
    sent_selection = get_sentence_range(orig_sentences, start, end)
    if os.path.exists(out_path):
        print("File already exists. Move or delete the file to do a new "
        + "analysis.")
        return
    write_csv_row(out_path, headings)
    for perm in perm_map:
        row = [perm]
        confidences = get_perm_confidence(sent_selection, perm, predictor)
        row += confidences
        write_csv_row(out_path, row)

# PURPOSE
# Generate headings to write file as CSV.
# SIGNATURE
# generate_headings :: Integer, String => List[String]
def generate_fields(num_fields, label):
    return [label + str(x) for x in range(num_fields)]

# PURPOSE
# Given an original document path, an output directory, start and end indices
# (inclusive), and a predictor, write the average nsp confidence of sentence 
# permutations to the directory, one csv per permutation.
# The original document is assumed to be one sentence per line.
# The 0th entry represents results for the original text.
# It is recommended to use pairwise_nsp_confidence and take the averages
# rather than using this function unless only averages are desired.
# SIGNATURE
# pairwise_nsp_confidence :: String, String, Integer, Interger, NSPredictor,
# Tensor => None
def doc_mean_nsp_confidence_to_file(orig_path, out_path, headings, start, \
    end, predictor):
    n_fact = end - start + 1
    perm_map = generate_perm_map(n_fact)
    orig_sentences = load_sentences_from_file(orig_path)
    sent_selection = get_sentence_range(orig_sentences, start, end)
    if os.path.exists(out_path):
        print("File already exists. Move or delete the file to do a new "
        + "analysis.")
        return
    write_csv_row(out_path, headings)
    for perm in perm_map:
        row = [perm]
        confidence = get_doc_confidence(sent_selection, perm, predictor)
        row += [confidence]
        write_csv_row(out_path, row)

# PURPOSE
# Given the path to an original document with one sentence per line,
# permute sentences in the given range and return confidence in pairwise
# comparisons between the sentences.
# SIGNATURE
# permute_and_analyze_pairwise :: String, Integer, Integer, NSPredictor, 
# Tensor => List[List]
def permute_and_analyze_pairwise(orig_path, start, end, predictor):
    n_fact = end - start + 1
    perm_map = generate_perm_map(n_fact)
    orig_sentences = load_sentences_from_file(orig_path)
    sent_selection = get_sentence_range(orig_sentences, start, end)
    pairwise_analysis = []
    for perm in perm_map:
        row = [perm]
        confidences = get_perm_confidence(sent_selection, perm, predictor)
        pairwise_analysis.append(row + confidences)
    return pairwise_analysis

# PURPOSE
# Given the path to an original document with one sentence per line,
# permute sentences in the given range and return average confidence from BERT 
# nsp for the full document.
# SIGNATURE
# permute_and_analyze_doc_means :: String, Integer, Integer, NSPredictor, 
# Tensor => List
def permute_and_analyze_doc_means(orig_path, start, end, predictor):
    n_fact = end - start + 1
    perm_map = generate_perm_map(n_fact)
    orig_sentences = load_sentences_from_file(orig_path)
    sent_selection = get_sentence_range(orig_sentences, start, end)
    doc_analysis = []
    for perm in perm_map:
        row = [perm]
        confidence = get_doc_confidence(sent_selection, perm, predictor)
        doc_analysis.append(row + [confidence])
    return doc_analysis

# PURPOSE
# Generates a list of n_fact! permutations.
# SIGNATURE
# generate_perm_map :: Integer => List[Tuple]
# EXAMPLE
# generate_perm_map(2) => [[0, 1], [1, 0]]
def generate_perm_map(n_fact):
    return list(permutations(range(n_fact)))

# PURPOSE
# Get sentences in a given range, [start-end] including
# both bounds.
# SIGNATURE
# get_sentence_range :: List, Integer, Integer => List
def get_sentence_range(orig_sentences, start, end):
    if ((len(orig_sentences) - 1) < end):
        print("end var set past end of sentences.")
        return
    return orig_sentences[start : end + 1]

# PURPOSE
# Given a list of sentences and a list of ints defining the permutation,
# permute the list and return based on the permuted list.
# SIGNATURE
# get_perm_pairs :: List, Tuple => List[List]
def get_perm_pairs(sents, perm):
    sentences = [sents[i] for i in perm]
    return create_pairs(sentences)

# PURPOSE
# Given a list of sentences, a list of ints defining the permutation,
# a predictor, return a list of confidences on each next sentence pair.
# SIGNATURE
# get_perm_confidence :: List, List, NSPredictor, Tensor => List
def get_perm_confidence(sent_selection, perm, predictor, predictions=None):
    pairs = get_perm_pairs(sent_selection, perm)
    return get_nsp_confidence(pairs, predictor, predictions)

# PURPOSE
# Given a list of sentences, a list of ints defining the permutation,
# a predictor, return the average confidence for all pairs in the list of 
# sentences.
# SIGNATURE
# get_doc_confidence :: List, List, NSPredictor, Tensor => Float
def get_doc_confidence(sent_selection, perm, predictor):
    pairs = get_perm_pairs(sent_selection, perm)
    return calculated_weighted_total(pairs, predictor, None)

# PURPOSE
# Permute a list of sentences and save the permutations to the
# o_dir directory, one permutation per file and one sentence per
# line.
# SIGNATURE
# write_perms_to_files :: List, String, List, String, String => None
def write_perms_to_files(sents, o_dir, perm_map,  f_head='perm', f_end='.txt'):
    count = 0
    for perm in perm_map:
        file_name = f_head + str(count) + f_end
        count += 1
        sentences = [sents[i] for i in perm]
        file_path = os.path.join(o_dir, file_name)
        with open(file_path, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                f.write(sentence)
                f.write('\n')