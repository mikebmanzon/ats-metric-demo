# Predictors
To use the LERC predictor, first run ./setup.sh to install the dependencies.

## LERC Predictor

To feed the predictor a list of json Dict or one single json Dict as inputs. Every json Dict is a piece of QA data with context, question, reference and candidate. Input should be like:

```python
inputs = [{
    'context': "There is one area I want to work on . Breast - feeding . Right now , Luke's addicted to the bottle . We were so eager to wean him off his nose tube that when he started taking a bottle , we made it our only goal to re - enforce that .",
    'question': "What may be your reason for wanting to work on Breast - feeding ?",
    'reference': "It could help my son .",
    'candidate': "I want to help Luke feed."
},{
    'context': "The strangest thing that has happened was when they were singing the Chinese National Anthem she was standing in front of the TV swaying and singing ... the words weren't really the words but it was kind of freaky ! I asked her is she knew the song and she said yes : ) She also is screamed fireworks a lot ! ! ! She did enjoy naming everyone she knows who is Chinese too ( that took a while LOL ) and she was so cute as the parade of countries happened .",
    'question': "What is probably true about this story ?",
    'reference': "They are watching the Olympics",
    'candidate': "the Olympics are watching"
}]
```

And you can initialize the predictor as:

```python
from lercPredictor import lercPredictor
predictor = lercPredictor()
```

Then make LERC predictions:

```python
scores = predictor.predict(inputs)
```

as the [demo.py](./demo.py) 

## T5 QA Predictor

To use the QA predictor, first initialize them as 

```python
from QAPredictor import QAPredictor
predictor = QAPredictor(model_name)
```

The model_name will be default as "allenai/unifiedqa-t5-base", which means we use T5-base model as the predictor model. You can also change to any model listed on [this page](https://huggingface.co/allenai).

And the input of the predictor has the same format as the LERC predictor. 

```python
preds = predictor.predict(inputs)
```

The predictor will return the answers to your inputs.

## NSP Predictor

Just initialize the predictor with two arguments model_name_or_path and max_length. The model_name_or_path can be a model_path in [huggingface hub](https://huggingface.co/models) or a local path to a model directory which contains config.json, pytorch_model.bin and vocal.txt. 

For example: 

```python
from NSPredictor import NSPredictor
predictor = NSPredictor("bert-base-uncased", 128)
```

After initialization, you can use the predictor to predict if a tuple of sentence is neighbor. The arguments should be a list of tuple, every tuple contains two sentences. And the precitor will return the list of predictions(0 for yes 1 for no) and confidence.

For example:

```python
predictor.predict([("To tourists, Amsterdam still seems very liberal.","Recently the city’s Mayor told them that the coffee shops that sell marijuana would stay open, although there is a new national law to stop drug tourism.")])
```

### Use GPU

You can also use GPU to process your model and data by assigning device when you initialize your model.

```python
predictor = NSPredictor("bert-base-uncased", 128, device="cuda")
```

Using GPU can be much faster than using CPU

### Multiprocessing

To use multiprocessing to do predictions, just give how many threads you want to use as an argument

```python
predictor.predict([("Hi","Hello")], threads=4)
```

### Things to optimize

**GPU and threads can not be used at the same time as there is an incompatible issue for torch**

**And also, if you are using a notebook, you can not use multiprocessing**

