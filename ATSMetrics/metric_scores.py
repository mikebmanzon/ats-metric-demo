from .predictors.lercPredictor import lercPredictor
from .predictors.QAPredictor import QAPredictor
from .predictors.question_generation.pipelines import pipeline
from .toolsv2.bernice_tools import *
from .predictors.NSPredictor import NSPredictor
import numpy as np
def sam_score(x):
    return 1 + 4/(1+np.exp(-1.5*x-2))

class Scorer:
    def __init__(self):
        # self.qa_generator = pipeline("question-generation")
        # self.lerc_predictor = lercPredictor()
        self.qa_predictor = QAPredictor()
        self.nspredictor = NSPredictor('bert-base-cased')
    def sam_score(self, orig_sentences, simplified_sentences, question=None, answer=None):
        # orig_sentence, simplified_sentence = orig_sentences[0], simplified_sentences[0]
        # if not question or not answer:
        #     try:
        #         generated_questions = self.qa_generator(orig_sentence)
        #     except:
        #         raise ValueError
        #     if len(generated_questions) == 0:
        #         raise ValueError
        #     question = generated_questions[0]['question']
        #     answer = generated_questions[0]['answer']

        # org_pred = self.qa_predictor.predict({"context": orig_sentence,
        #                                  "question": question,
        #                                  "reference": answer})[0]
        # simp_pred = self.qa_predictor.predict({"context": simplified_sentence,
        #                                  "question": question,
        #                                  "reference": answer})[0]
        # org_pred_score = self.lerc_predictor.predict({"candidate": org_pred,
        #           "context": orig_sentence,
        #           "question": question,
        #           "reference": answer})
        # simp_pred_score = self.lerc_predictor.predict({"candidate": simp_pred,
        #                                               "context": orig_sentence,
        #                                               "question": question,
        #                                               "reference": answer})
        # if org_pred_score > 5:
        #     org_pred_score = 5
        # elif org_pred_score < -5:
        #     org_pred_score = -5
        # if simp_pred_score > 5:
        #     simp_pred_score = 5
        # elif simp_pred_score < -5:
        #     simp_pred_score = -5
        # return sam_score(simp_pred_score - org_pred_score), question, simp_pred, org_pred, simp_pred_score, org_pred_score
        return 100
    # PURPOSE
    # Given a list of original sentences and a list of simplified sentences,
    # calculate and return the BERNICE score for document cohesion.
    # The lists may be of different lengths and should be the
    # original and simplified version of the same document.
    # SIGNATURE
    # bernice_score :: List[String], List[String] => Float
    def bernice_score(self, orig_sentences, simplified_sentences):
        orig_pairs = create_pairs(orig_sentences)
        num_orig_pairs = len(orig_pairs)
        simp_pairs = create_pairs(simplified_sentences)
        # predictor = NSPredictor('bert-base-cased')
        orig_predictions = self.nspredictor.predict(orig_pairs)
        simp_predictions = self.nspredictor.predict(simp_pairs)
        orig_confidences = get_nsp_confidence(orig_pairs, self.nspredictor, orig_predictions)
        simp_confidences = get_nsp_confidence(simp_pairs, self.nspredictor, simp_predictions)
        orig_mean = get_avg_nsp_confidence(orig_confidences)
        simp_mean = get_avg_nsp_confidence(simp_confidences)
        orig_incoherent = count_invalid(orig_pairs, self.nspredictor, orig_predictions)
        simp_incoherent = count_invalid(simp_pairs, self.nspredictor, simp_predictions)
        return calculate_bernice(simp_mean, orig_mean, simp_incoherent, orig_incoherent, num_orig_pairs)
