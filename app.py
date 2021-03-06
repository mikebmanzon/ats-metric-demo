import sys

import json
import streamlit as st

from ATSMetrics.bernice import bernice_score
from ATSMetrics.toolsv2.nlp_tools import *
from ATSMetrics.metric_scores import Scorer

with open('samples.json') as f:
    samples = json.load(f)

# caches already calculated bernice scores
@st.cache
def calculate_bernice(scorer, orig, simple):
    return scorer.bernice_score(orig, simple)

@st.cache(allow_output_mutation=True)
def instantiate_scorer():
    scorer = Scorer()
    return scorer
scorer = instantiate_scorer()

# sidebar menu to select sample or manual input
simplifiers = ['Manual', 'Sample']
simplifier_input = st.sidebar.selectbox('Select mode:', simplifiers)

manual_questions = [False, True]
manual_question_input = st.sidebar.selectbox('User generated question & answer:', manual_questions)
if manual_question_input:
    user_question = st.sidebar.text_input("Input your question:", '')
    user_answer = st.sidebar.text_input("Input your answer:", '')
else:
    user_question, user_answer = None, None

st.title('Coherence and Meaning Preservation')
st.write("Our metrics seek to address two aspects of text classification not represented in major metrics: coherence and meaning preservation. To measure coherence, BERNice uses BERT's Next Sentence Predictor to evaluate the cohesion of pairs of neighboring sentences in the original and simplified texts. To measure meaning preservation, we first generate a question and answer from the original text (or take input question and answer). We then use LERC to score that answer on the original and simplified texts. Our meaning preservation metric, SAM, is the bias of the simplified and original LERC scores.")
st.write('Use the sidebar menu to select manual or sample text mode, as well as user input or autogenerated question and answer.')
st.write('Mode: ', simplifier_input)
if simplifier_input == 'Manual':
    st.write('Input original and simplified texts in the corresponding boxes below.')
elif simplifier_input == 'Sample':
    st.write('Select sample original text and candidate simplified texts using the corresponding dropdown menus below.')

# puts original text input on left side
top_left_column, top_right_column = st.beta_columns(2)

# puts simple text on right side (input if manual mode, simplifies text if simplifier has been selected)
if simplifier_input == 'Manual':
    orig_text = top_left_column.text_area("Text to simplify:", '', height=200)
    simple_text = top_right_column.text_area("Simplified text:", '', height=200)

elif simplifier_input == 'Sample':
    # user selects original sample text
    orig_sample_choice = top_left_column.selectbox('Original Sample:', list(samples.keys()))
    orig_text = samples[orig_sample_choice]['orig']

    # user selects simple sample text
    simple_sample_choice = top_right_column.selectbox('Simple Sample:', list(samples[orig_sample_choice]['simple'].keys()))
    simple_text = samples[orig_sample_choice]['simple'][simple_sample_choice]

    # sample texts are displayed
    top_left_column.write('Original text:')
    top_left_column.write(orig_text)
    top_right_column.write("Simplified text:")
    top_right_column.write(simple_text)

# original and simple texts are split into lists of sentences
orig_split = sent_tokenize_text(orig_text)
simple_split = sent_tokenize_text(simple_text)

# evaluate button appears if both original and simple text are present
if orig_text and simple_text:
    eval_button = st.button("Evaluate")
    bottom_left_column, bottom_right_column = st.beta_columns(2)

    # pressing eval button calculates BERNice (and TODO meaning preservation) metrics
    if eval_button:
        bottom_left_column.write('Coherence: BERNice')
        bottom_right_column.write('Meaning preservation: SAM')
        if len(simple_split) > 1 and len(orig_split) > 1:
            bern_score = scorer.bernice_score(orig_split, simple_split)
            bottom_left_column.write('Coherence score = {num:.3f}'.format(num=bern_score))
        else:
            bottom_left_column.write('BERNice requires at least 2 sentences in original and simplified texts.')
        try:
            sam_score, question, simp_pred, org_pred, simp_pred_score, org_pred_score = scorer.sam_score(orig_split, simple_split, user_question, user_answer)
            bottom_right_column.write('Meaning Preservation score = {num:.3f}'.format(num=sam_score))
            bottom_right_column.write(f'Question is "{question}"')
            bottom_right_column.write(
                f'Answer to the original context is "{org_pred}", with a lerc score {round(org_pred_score, 3)}')
            bottom_right_column.write(
                f'Answer to the simplified context is "{simp_pred}", with a lerc score {round(simp_pred_score, 3)}')
        except Exception as e:
            bottom_right_column.write("Can not generate a question for the input context, maybe you can try to input manual questions")



