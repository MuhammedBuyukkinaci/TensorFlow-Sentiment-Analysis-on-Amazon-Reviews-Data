# TensorFlow-Sentiment-Analysis-on-Amazon-Reviews-Data

Implementing different RNN models on a subset of Amazon Reviews data.

# Requirements

```
pip3 install requirements.txt
```

# Dataset

Dataset used in this project is a subset of [Amazon Reviews](https://www.kaggle.com/bittlingmayer/amazonreviews#train.ft.txt.bz2).

Train dataset has 150k rows.

Test dataset has 30k rows.

Output Classes are positive and negative (Binary Classification).

The models were trained on train dataset and validated on test dataset.

# Pre-trained Word Embeddings

I used 100-Dimensional [GloVe Word Embeddings](https://nlp.stanford.edu/projects/glove/) for this project.

You can download it from [here](https://www.kaggle.com/terenceliu4444/glove6b100dtxt#glove.6B.100d.txt).

After downloading, _glove.6B.100d.txt_ must be in where .py files are.

# Results

![results](https://github.com/MuhammedBuyukkinaci/TensorFlow-Sentiment-Analysis-on-Amazon-Reviews-Data/blob/master/results.png)

