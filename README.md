# TensorFlow-Sentiment-Analysis-on-Amazon-Reviews-Data

Implementing different RNN([GRU](https://arxiv.org/pdf/1412.3555.pdf) & [LSTM](https://www.bioinf.jku.at/publications/older/2604.pdf)) models on a subset of Amazon Reviews data.

# Requirements

```
pip3 install requirements.txt
```

# Pre-trained Word Embeddings

I used 100-Dimensional [GloVe Word Embeddings](https://nlp.stanford.edu/projects/glove/) for this project.

You can download it from [here](https://www.kaggle.com/terenceliu4444/glove6b100dtxt#glove.6B.100d.txt).

After downloading, _glove.6B.100d.txt_ must be in where .py files are.

# Default Version

If you have GPU, use ``` 02_CUDNN_GRU.py ``` or ``` 06_CUDNN_LSTM.py ``` as default .

If you don't have, use ``` 04_CPU_Optimized_GRU.py ``` or ``` 08_CPU_Optimized_LSTM.py ``` as default.

# Training

``` python3 01_Baseline_GPU.py ```

``` python3 02_CUDNN_GRU.py ```

``` python3 03_CUDNN_GRU_bidirectional.py ```

``` python3 04_CPU_Optimized_GRU.py ```

``` python3 05_Baseline_LSTM.py ```

``` python3 06_CUDNN_LSTM.py ```

``` python3 07_CUDNN_LSTM_bidirectional.py ```

``` python3 08_CPU_Optimized_LSTM.py ```

``` python3 09_CONV1D_CUDNNGRU.py ```

``` python3 10_CONV2D.py ```

``` python3 11_Attention_GRU.py ```

``` python3 12_Attention_CUDNNGRU.py ```

``` python3 13_Attention_CUDNNGRU_bidirectional.py ```


# Dataset

Dataset used in this project is a subset of [Amazon Reviews](https://www.kaggle.com/bittlingmayer/amazonreviews#train.ft.txt.bz2).

Train dataset has 150k rows.

Test dataset has 30k rows.

Output classes are positive and negative (Binary Classification).

The models were trained on train dataset and validated on test dataset.

After cloning the repository

# Models

``` 01_Baseline_GRU.py ``` --> Base GRU implementation.

``` 02_CUDNN_GRU.py ``` --> GPU optimized CUDNNGRU implementation.

``` 03_CUDNN_GRU_bidirectional.py ``` --> GPU optimized CUDNNGRU bidirectional implementation.

``` 04_CPU_Optimized_GRU.py ``` --> CPU optimized GRU implementation.

``` 05_Baseline_LSTM.py ``` --> Base LSTM implementation.

``` 06_CUDNN_LSTM.py ``` --> GPU optimized CUDNNLSTM implementation.

``` 07_CUDNN_LSTM_bidirectional.py ``` --> GPU optimized CUDNNLSTM bidirectional implementation.

``` 08_CPU_Optimized_LSTM.py ``` --> CPU optimized LSTM implementation.

``` 09_CONV1D_CUDNNGRU.py ``` --> CONV1D BEFORE CUDNNGRU implementation

``` 10_CONV2D.py ``` ---> CONV2D implementation before fully connected layers. 

``` python3 11_Attention_GRU.py ``` ---> Attention Layer including GRU implementation

``` python3 12_Attention_CUDNNGRU.py ``` ---> Attention Layer including CUDNNGRU implementation

``` python3 13_Attention_CUDNNGRU_bidirectional.py ``` --->  Attention Layer including Bidirectional CUDNNGRU implementation


# Early Stopping

I defined a customized function to check if loss is decreasing on test data.

If it isn't decreasing for a time period, the model stops to train.

``` Python
#Defining a function for early stopping
def early_stopping_check(x):
    if np.mean(x[-20:]) <= np.mean(x[-80:]):
        return True
    else:
        return False
```

# Results

![results](https://github.com/MuhammedBuyukkinaci/TensorFlow-Sentiment-Analysis-on-Amazon-Reviews-Data/blob/master/results.png)

