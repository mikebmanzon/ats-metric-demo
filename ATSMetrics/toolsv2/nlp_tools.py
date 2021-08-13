import re

# PURPOSE
# Given a single string, split it into a list of sentences.
# SIGNATURE
# sent_tokenize_text :: String => List[String]
def sent_tokenize_text(text):
    split_sentences = re.compile(r'(?<!\w\.\w.)(?<!\s[A-Z][a-z]\.)(?<!\s[A-Z]\.)(?<!\s[a-z]\.)(?<=\.|\?|\!)\s(?=[A-Z0-9]|\s|\“[A-Z0-9]|\s|\"[A-Z0-9]|\s|\'[A-Z0-9]|\s|\‘[A-Z0-9]|\s|\`[A-Z0-9]|\s|\“\'[A-Z0-9])|(?<!\w\.\w.)(?<!\s[A-Z][a-z]\.)(?<!\s[A-Z]\.)(?<!\s[a-z]\.)(?<=\."|\?"|\!"|\.”|\?”|\!”|\.\'|\?\'|\!\'|\.’|\?’|\!’|\.`|\?`|\!`)\s(?=[A-Z0-9]|\s|\“[A-Z0-9]|\s|\"[A-Z0-9]|\s|\'[A-Z0-9]|\s|\‘[A-Z0-9]|\s|\`[A-Z0-9]|\s|\“\'[A-Z0-9])|(?<!\w\.\w.)(?<!\s[A-Z][a-z]\.)(?<!\s[A-Z]\.)(?<!\s[a-z]\.)(?<=\.\'”|\?\'”|\!\'”)\s(?=[A-Z0-9]|\s|\“[A-Z0-9]|\s|\"[A-Z0-9]|\s|\'[A-Z0-9]|\s|\‘[A-Z0-9]|\s|\`[A-Z0-9]|\s|\“\'[A-Z0-9])')
    sentences = re.split(split_sentences, text)
    sentences = [sent.strip() for sent in sentences]
    return sentences

# PURPOSE
# Function to convert a text file into a single string.
# SIGNATURE
# convert_lines_to_single_string :: List[String] => String
def convert_lines_to_single_string(lines):
    full_string = ''
    full_string += lines[0]
    for i in range(1, len(lines)):
        full_string += ' '
        full_string += lines[i]
    return full_string