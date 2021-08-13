from lercPredictor import lercPredictor
import json
import pandas as pd
from tqdm import tqdm
from QAPredictor import QAPredictor
# predictor = lercPredictor()
# scores = predictor.predict([{
#     'context': "There is one area I want to work on . Breast - feeding . Right now , Luke's addicted to the bottle . We were so eager to wean him off his nose tube that when he started taking a bottle , we made it our only goal to re - enforce that .",
#     'question': "What may be your reason for wanting to work on Breast - feeding ?",
#     'reference': "It could help my son .",
#     'candidate': "I want to help Luke feed."
# },{
#     'context': "The strangest thing that has happened was when they were singing the Chinese National Anthem she was standing in front of the TV swaying and singing ... the words weren't really the words but it was kind of freaky ! I asked her is she knew the song and she said yes : ) She also is screamed fireworks a lot ! ! ! She did enjoy naming everyone she knows who is Chinese too ( that took a while LOL ) and she was so cute as the parade of countries happened .",
#     'question': "What is probably true about this story ?",
#     'reference': "They are watching the Olympics",
#     'candidate': "the Olympics are watching"
# }])

if __name__ == "__main__":
    predictor = lercPredictor()
    qa_predictor = QAPredictor()
    with open('MOCHA/data/mocha/dev.json') as f:
        dev_data = json.load(f)
    result_dict = {}
    df_list = []
    for i,d in tqdm(dev_data.items()):
        dataset = i
        for k,j in tqdm(d.items()):
            source = j['metadata']['source']
            inputs = {"candidate":j['candidate'],
                           "context":j['context'],
                           "question":j['question'],
                           "reference":j['reference']}
            score = predictor.predict(inputs)
            id = k
            human_score = j['score']
            pred = qa_predictor.predict(inputs)[0]
            pred_inputs = inputs.copy()
            pred_inputs['candidate'] = pred
            pred_score = predictor.predict(pred_inputs)
            df_list.append(pd.DataFrame({'dataset':[dataset],'id':[id],'model':[source],
                          'human_score':[human_score],'lerc_score':[score],'answer_by_t5':[pred],'t5_lerc_score':[pred_score]}))
    pd.concat(df_list,axis=0).to_csv("MOCHA_lerc_dev_predictions.csv")

