from transformers import AutoTokenizer, T5ForConditionalGeneration
import csv
import json
import re
import tensorflow as tf

def normalize_text(text):
    """Lowercase and remove quotes from a TensorFlow string."""
    text = re.sub("'(.*)'", r"\1",text.lower())
    return text

def narrative_qa():
    import os
    if not os.path.exists("narrativeqa"):
        os.makedirs("narrativeqa")
    paragraphs = {}
    with open("./datasets/narrativeqa/third_party/wikipedia/summaries.csv") as f:
        spamreader = csv.reader(f)
        for i, line in enumerate(spamreader):
            print(line)
            if i == 0:
                continue
            paragraphs[line[0]] = line[2].replace("\n", "")

    fout_test = open(f"narrativeqa/test.tsv", "w+")
    fout_train = open(f"narrativeqa/train.tsv", "w+")
    fout_dev = open(f"narrativeqa/dev.tsv", "w+")
    counts = open(f"narrativeqa/counts.json", "w+")

    count_train = 0
    count_test = 0
    count_dev = 0
    with open("./datasets/narrativeqa/qaps.csv") as f:
        spamreader = csv.reader(f)
        for i, line in enumerate(spamreader):
            print(line)
            if i == 0:
                continue
            line1 = f"{line[2]} \\n {paragraphs[line[0]]} \t {line[3]} \n"
            line2 = f"{line[2]} \\n {paragraphs[line[0]]} \t {line[4]} \n"
            if line[1] == "train":
                fout_train.write(line1)
                fout_train.write(line2)
                count_train += 1
            elif line[1] == "test":
                fout_test.write(line1)
                fout_test.write(line2)
                count_test += 1
            elif line[1] == "valid":
                fout_dev.write(line1)
                fout_dev.write(line2)
                count_dev += 1
            else:
                print(" >>>> ERROR ")

    counts.write(json.dumps({"train": count_train, "dev": count_dev, "test": count_test}))




class QAPredictor():
    def __init__(self, model_name="allenai/unifiedqa-t5-base"):
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def predict(self, inputs, **generator_args):
        if isinstance(inputs, list):
            results = []
            for i in inputs:
                input_string = normalize_text(f"{i['question']} \\n {i['context']}")
                input_ids = self.tokenizer.encode(input_string, return_tensors="pt")
                res = self.model.generate(input_ids, **generator_args)
                results.append(res)
            return results

        else:
            input_string = normalize_text(f"{inputs['question']} \\n {inputs['context']}")
            input_ids = self.tokenizer.encode(input_string, return_tensors="pt")
            res = self.model.generate(input_ids, **generator_args)
            return self.tokenizer.batch_decode(res, skip_special_tokens=True)


if __name__ == "__main__":
    predictor = QAPredictor()
    inputs = {
    'context': "The strangest thing that has happened was when they were singing the Chinese National Anthem she was standing in front of the TV swaying and singing ... the words weren't really the words but it was kind of freaky ! I asked her is she knew the song and she said yes : ) She also is screamed fireworks a lot ! ! ! She did enjoy naming everyone she knows who is Chinese too ( that took a while LOL ) and she was so cute as the parade of countries happened .",
    'question': "What is probably true about this story ?"
}
    print(predictor.predict(inputs))