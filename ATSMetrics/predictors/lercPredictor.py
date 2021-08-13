# from allennlp.predictors import Predictor
import torch
from .lerc.lerc_predictor import LERCPredictor
class lercPredictor:
    def __init__(self):
        # Loads an AllenNLP Predictor that wraps our trained model
        self.predictor = LERCPredictor.from_path(
                archive_path='https://storage.googleapis.com/allennlp-public-models/lerc-2020-11-18.tar.gz',
                predictor_name='lerc',
                cuda_device=0 if torch.cuda.is_available() else -1
            )
        # The instance we want to get LERC score for in a JSON format
    def predict(self, inputs):
        if isinstance(inputs, list):
            scores = []
            for i in inputs:
                scores.append(self.predictor.predict_json(i)['pred_score'])
            print('Average predicted LERC Score:', sum(scores)/len(scores))
            return scores
        else:
            return self.predictor.predict_json(inputs)['pred_score']
