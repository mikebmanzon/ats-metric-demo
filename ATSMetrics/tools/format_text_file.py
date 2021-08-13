# This script can be used to make text files containing one
# sentence per line.

import re
from .file_tools import *

# PURPOSE
# Given an original file name and a desired output file name,
# Format the text from the original file such that the output file
# contains the sentences from the original file, one sentence per line.
# SIGNATURE
# write_formatted_file :: String, String => None
def write_formatted_file(orig_fp, output_fp):
    text = convert_to_single_string(orig_fp)
    sentences = sent_tokenize_text(text)
    write_lines_to_file(sentences, output_fp)

# PURPOSE
# Given a single string, split it into a list of sentences.
# SIGNATURE
# sent_tokenize_text :: String => List[String]
def sent_tokenize_text(text):
    split_sentences = re.compile(r'(?<!\w\.\w.)(?<!\s[A-Z][a-z]\.)(?<!\s[A-Z]\.)(?<!\s[a-z]\.)(?<=\.|\?|\!)\s(?=[A-Z0-9]|\s|\“[A-Z0-9])|(?<!\w\.\w.)(?<!\s[A-Z][a-z]\.)(?<!\s[A-Z]\.)(?<!\s[a-z]\.)(?<=\."|\?"|\!"|\.”|\?”|\!”|\.\'|\?\'|\!\')\s(?=[A-Z0-9]|\s|\“[A-Z0-9])')
    sentences = re.split(split_sentences, text)
    sentences = [sent.strip() for sent in sentences]
    return sentences

# PURPOSE
# Function to convert a text file into a single string.
def convert_to_single_string(orig_fn):
    with open(orig_fn, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    full_string = ''
    for line in lines:
        full_string += line
        full_string += ' '
    return full_string

# Run this function to see how the process works on a test document.
def demo():
    ifn = 'gatsby_short_unprocessed.txt'
    ofn = 'gatsby_short_processed.txt'
    write_formatted_file(ifn, ofn)