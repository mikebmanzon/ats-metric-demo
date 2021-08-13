import os
from .file_tools import load_sentences_from_file, write_lines_to_formatted_file
from tqdm import tqdm
# PURPOSE
# Given a file path, an output path, and a simplifier,
# simplify the text in the file using the simplifier and
# write the simplified text to the output path, one
# sentence per line.
# SIGNATURE
# simplify_file_to_file :: String, String, Simplifier => None
def simplify_file_to_file(file_path, out_path, simplifier):
    sentences = load_sentences_from_file(file_path)
    simplified = simplifier.simplify(sentences)
    write_lines_to_formatted_file(simplified, out_path)

# PURPOSE
# Given an input directory, an output directory, and
# a simplifier, simplify the text in each file in
# the input directory and write the simplification to
# the output directory, one sentence per line.
# SIGNATURE
# simplify_dir :: String, String, Simplifier => None
def simplify_dir(input_dir, output_dir, simplifier):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for file in tqdm(os.listdir(input_dir)):
        fpath = os.path.join(input_dir, file)
        out_path = os.path.join(output_dir, file)
        print("**************************************************** SIMPLIFYING " + file + "****************************************************")
        simplify_file_to_file(fpath, out_path, simplifier)