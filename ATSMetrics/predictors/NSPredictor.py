from transformers import AutoConfig, BertForNextSentencePrediction, AutoTokenizer
import torch
import os
if "JPY_PARENT_PID" in os.environ:
    print("Import tqdm for jupyter notebook")
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
from torch.multiprocessing import Pool, cpu_count, set_start_method
from functools import partial



class NSPredictor():
    def __init__(self, model_name_or_path, max_length=128, device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.device = device
        self.model = BertForNextSentencePrediction.from_pretrained(model_name_or_path, config=self.config)
        self.model.to(device)
        self.model.eval()
        self.max_length = max_length

    def predict(self, texts, threads=1):
        if self.device == "cuda" and threads>1:
            raise NotImplementedError
        results = []
        confidences = []
        if threads > 1:
            with Pool(min(cpu_count(), threads)) as p:
                # annotate_ = partial(_forward, model=self.model, tokenizer=self.tokenizer, max_length=self.max_length)
                nsp_results = list(
                    tqdm(
                        p.imap(self._forward, texts),
                        total=len(texts),
                        desc="Next Sentence Prediction",
                    )
                )
            results.extend(i[0] for i in nsp_results)
            confidences.extend([i[1] for i in nsp_results])
            return results, confidences
        else:
            for i in tqdm(texts, desc="Next Sentence Prediction"):
                opt = self._forward(i)
                results.append(opt[0])
                confidences.append(opt[1])
            return results, confidences
        # return confidence.max(-1).indices.detach().tolist(), confidence.tolist()


    def _forward(self, sentence_pair):
        encoding = self.tokenizer(sentence_pair[0], sentence_pair[1], max_length=self.max_length, padding="max_length",
                                  truncation=True, return_token_type_ids=True, return_tensors="pt").to(self.device)
        seq_relationship_logits = self.model(**encoding)[0]
        confidence = torch.nn.functional.softmax(seq_relationship_logits, dim=1)
        return confidence.max(-1).indices.detach().tolist()[0], confidence.tolist()[0]

if __name__ == "__main__":
    set_start_method('spawn')
    predictor = NSPredictor("bert-base-uncased")
    import sys
    sys.path.append("../")
    from tools import nsp_score_tools
    lit_sentences1 = nsp_score_tools.load_sentences_from_file('../BERNICE_sanity_check/text/informal/gatsby_ch1.txt')
    lit_pairs1 = nsp_score_tools.create_pairs(lit_sentences1)
    results = predictor.predict(lit_pairs1,8)
    print(results)
