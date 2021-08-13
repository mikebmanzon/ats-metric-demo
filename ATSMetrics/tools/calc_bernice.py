import math

def calculate_bernice(mean_simp, mean_orig, inc_simp, inc_orig, total_pairs):
    pair_weight = 100
    doc_weight = 100
    pair_score = calc_pair_score(mean_simp, mean_orig, pair_weight)
    doc_score = calc_doc_score(inc_simp, inc_orig, doc_weight, total_pairs)
    return pair_score + doc_score

def calc_pair_score(mean_simp, mean_orig, pair_weight):
    return (mean_simp / mean_orig) * pair_weight

def calc_doc_score(inc_simp, inc_orig, doc_weight, total_pairs):
    x = calc_x(inc_simp, inc_orig, total_pairs)
    stretch = .16
    return (2 / (1 + math.exp(-x / stretch)) - 1) * doc_weight

def calc_x(inc_simp, inc_orig, total_pairs):
    return (inc_orig - inc_simp) / (total_pairs - inc_orig + 1)

# print(calculate_bernice(93.72618915047366,  88.28116433089285, 2, 4, 28)) # Banksy
# print(calculate_bernice(74.34062, 88.4292, 9, 3, 30)) # Denmark