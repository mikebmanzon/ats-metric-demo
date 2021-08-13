import csv
import os
import json

from .nlp_tools import *

# PURPOSE
# Write lines to a file.
# SIGNATURE
# write_lines_to_files :: List[String], String => None
def write_lines_to_file(lines, fpath):
    with open(fpath, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line)
            f.write('\n')

# PURPOSE
# Given a file path, a list of fields, and a list of lists of rows,
# write the information to a csv file.
# SIGNATURE
# write_to_csv :: String, List, List[List] => None
def write_to_csv(fpath, fields, rows):
    with open(fpath, 'w', newline='', encoding='utf-8') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(rows)

# PURPOSE
# Given a csv file path, open the file and append a given row in csv format.
# SIGNATURE
# write_csv_row :: String, List => None
def write_csv_row(fpath, row):
    with open(fpath, 'a', newline='', encoding='utf-8') as f:
        write = csv.writer(f)
        write.writerow(row)

# returns a list of sentences read from a file.
# assumes file seperates sentences by line already
def load_sentences_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        sentences = f.read().splitlines()
    return sentences

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
# Given an original file name and a desired output file name,
# format the text from the original file such that the output file
# contains the sentences from the original file, one sentence per line.
# SIGNATURE
# write_formatted_file :: String, String => None
def write_lines_to_formatted_file(lines, output_fp):
    text = convert_lines_to_single_string(lines)
    sentences = sent_tokenize_text(text)
    write_lines_to_file(sentences, output_fp)

# PURPOSE
# Function to convert a text file into a single string.
def convert_to_single_string(orig_fp):
    with open(orig_fp, 'r', encoding='utf-8-sig') as f:
        lines = f.read().splitlines()
    return convert_lines_to_single_string(lines)

# PURPOSE
# Format all files in the input directory to files of the
# same name with one sentence per line in the output directory.
# SIGNATURE
# format_dir :: String, String => None
def format_dir(input_dir, output_dir):
    for file in os.listdir(input_dir):
        in_path = os.path.join(input_dir, file)
        out_path = os.path.join(output_dir, file)
        write_formatted_file(in_path, out_path)

# PURPOSE
# Read a json file into memory.
# SIGNATURE
# read_json :: String => Dictionary
def read_json(fpath):
    with open(fpath, 'r') as f:
        data = json.load(f)
    return data

# PURPOSE
# Write a dictionary to a JSON file at the specified output path.
# Format the JSON file to be human-readable.
# SIGNATURE
# write_json :: Dictionary, String => None
def write_json(write_dict, output_path):
    with open(output_path, 'w') as f:
        json.dump(write_dict, f, indent=4)