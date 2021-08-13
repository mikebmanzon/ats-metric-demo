# Coherence and Meaning Preservation Metric Demo App

## Description
This webapp is a demo for BERNice (our coherence metric) and [name TBD] (our meaning preservation metric). 

## Instructions
In this directory run the command 'streamlit run app.py' and go to the specified network URL. 
Use the dropdown menu in the sidebar to select 'Manual' or 'Sample' mode. 
In 'Manual' mode, you can input your own original and simplified texts for evaluation. 
In 'Sample' mode, first select your original sample text from the left dropdown menu, then select which candidate simplification you want to evaluate.
The sample texts can be modified by changing samples.json in this directory.

## Requirements
* streamlit (install with 'pip install streamlit')
* NSPredictor and whatever predictor we end up using for Meaning Preservation metric (run setup.sh in '../src/ATSMetrics/predictors/' )