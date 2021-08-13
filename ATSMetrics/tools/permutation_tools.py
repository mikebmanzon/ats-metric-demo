import numpy as np
import os
from itertools import permutations
import sys
import csv
sys.path.append("..")
from tools.nsp_score_tools import *
from tools.sanity_check_tools import *
import importlib.machinery
import importlib.util
from pathlib import Path

## pairwise_nsp_confidence_to_file and oc_mean_nsp_confidence_to_file can be
## dramatically more efficient by using precalculated pairs. Implement this
## if the tools are needed further.


curr_dir = Path('permutation_tools.py').parent

predictor_module_path = os.path.join(curr_dir, '..', 'BERNICE_exp_1', 'NSPredictor.py')
loader = importlib.machinery.SourceFileLoader( 'NSPredictor.py', predictor_module_path)
spec = importlib.util.spec_from_loader('NSPredictor.py', loader)
NSPredictor = importlib.util.module_from_spec(spec)
loader.exec_module(NSPredictor)

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
# Given a csv file path, open the file and append a given row in csv format.
# SIGNATURE
# write_csv_row :: String, List => None
def write_csv_row(fpath, row):
    with open(fpath, 'a', newline='') as f:
        write = csv.writer(f)
        write.writerow(row)

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
# Given an original document path, an output directory, start and end indices
# (inclusive), and a predictor, write the pairwise nsp confidence of sentence 
# permutations to the directory, one csv per permutation. 
# The original document is assumed to be one sentence per line.
# The 0th entry represents results for the original text.
# SIGNATURE
# pairwise_nsp_confidence :: String, String, Integer, Interger, NSPredictor,
# Tensor => None
def r_permutations_confidence_to_file(orig_path, out_path, headings, predictor):
    if os.path.exists(out_path):
        print("File already exists. Move or delete the file to do a new "
        + "analysis.")
        return
    sents = load_sentences_from_file(orig_path)
    r_perms = generate_r_permutations_map(len(sents), 2)
    write_csv_row(out_path, headings)
    for r_p in r_perms:
        row = [r_p]
        r_p_sents = [(sents[r_p[0]], sents[r_p[1]])]
        confidence = get_nsp_confidence(r_p_sents, predictor, None)
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
# Given a file path, a list of fields, and a list of lists of rows,
# write the information to a csv file.
# SIGNATURE
# write_to_csv :: String, List, List[List] => None
def write_to_csv(fpath, fields, rows):
    with open(fpath, 'w', newline='') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(rows)

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

# PURPOSE
# Get numeric tuples representing the pairs genterated by
# nsp_score_tools.createpairs().
# SIGNATURE
# get_pairs_numeric :: List[String] => List[Tuple]
def get_pairs_numeric(sentences):
    return [(x - 1, x) for x in range(1, len(sentences))]