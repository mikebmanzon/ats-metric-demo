import os
from .nsp_score_tools import load_sentences_from_file
from .file_tools import write_lines_to_file

def simplify_file_to_file(file_path, out_path, simplifier):
    sentences = load_sentences_from_file(file_path)
    simplified = simplifier.simplify(sentences)
    write_lines_to_file(simplified, out_path)

def simplify_dir(input_dir, output_dir, simplifier):
    for file in os.listdir(input_dir):
        fpath = os.path.join(input_dir, file)
        out_path = os.path.join(output_dir, file)
        print("**************************************************** SIMPLIFYING " + file + "****************************************************")
        with open(fpath, 'r', encoding='utf-8') as f:
            sentences = f.read().splitlines()
            simp_sentences = simplifier.simplify(sentences)
            with open(out_path, 'w', encoding = 'utf-8') as g:
                for sentence in simp_sentences:
                    g.write(sentence)
                    g.write('\n')