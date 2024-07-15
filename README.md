# AI Writing Detector

Data: https://www.kaggle.com/datasets/jdragonxherrera/augmented-data-for-llm-detect-ai-generated-text

## Overview of Data
Via kaggle: 
Some of those already compile the others inside them, so I first removed the duplicates comparing by full text. After that the data augmentation took place, with a process composed of 2 steps that were iterated over and over, first I tried correcting typos on the texts by using language_tool_python, then I introduced noise the way the organizators seem to have done it (see https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/452279), then I corrected typos again, and repeat. After repeating these steps a couple of times I removed duplicates again comparing by full text.

The result is this dataset, it's split in train and test because I wanted to prevent information leaking between the train and test sections, so I did the steps independently on each of them (I split them before doing the data augmentation). If you don't care about train and test you can just concatenate both into a single dataset for training purposes.

#### Dictionary

| text  |  label |
|---|---|
| text string data | response variable (0 = Not AI, 1 = AI)  |

#### Number of observations:

- Train: 346,977 (222,154 not AI, 124,823 AI)
- Test: 86,587 (55,845 not AI, 30,742 AI)

## Preprocessing

#### Downsampling & Rebalancing Data
To save on computation time down the road, I downsampled my data. While I would normally be against removing good training data, I have overestimated the capability of my machine and my patience before and do not want to make those mistakes again. The data will be restrucuted to 40,000 training samples (20k AI, 20k not AI) and 10,000 testing samples.

#### Cleaning Text Data
First, I wanted to remove any PII from the text. In the past, I haven't had much luck with the Scrub or Sanityze libraries so I resorted to building my own function to cover the basic PII. I used regex functions to remove emails, phone numbers, and social security numbers. Using Spacy's NER, I also removed any instances of names. Note: if you elect to use Spacy's small libary, I would use POS == "PROPN" rather than NER because it often does not identify names accurately. Within the function, I also performed some classic NLP tasks such as converting to lowercase, remove newline characters and punctuation, and lemmatization.

## Modeling
With the data cleaned and preprocessed, I began modeling. I elected to train both the Spacy and Bert embeddings using the same machine learning techniques without hyperparameter tuning: naive bayes, logistic regression, KNN, random forest, XGBoost, SVM, and a simple RNN.

#### Spacy
All models tended to follow the following trends: performing very well in identifying AI instances (high recall for class 1) but have moderate precision, meaning there are some false positives for AI instances. For non AI instances (class 0), the model has great precision, meaning it has very few false positives for non AI instances, but the recall is lower, indicating it misses quite a few actual non AI instances.

#### TF TextVectorization
Using generic TensorFlow text classification code, I was able to train the model with the highest accuracy.

#### Bert
Generally, not as great as Spacy.

#### FastText

## Error & Future Questions

* Use more data
* Adjust model hyperparameters/neural network architecture
* Test different embedding techniques
* Use an LLM to classify text
