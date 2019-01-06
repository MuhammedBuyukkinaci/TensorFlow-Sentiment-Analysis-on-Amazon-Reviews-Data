"""
Created on Sun Jan  6 12:11:28 2019

@author: Muhammed Buyukkinaci
"""

import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
import os
from tqdm import tqdm
import re
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from datetime import datetime

#Always seed the randomness of this universe.
np.random.seed(51)

#Define HyperParameters
MAX_WORD_TO_USE = 100000 # how many words to use in training
MAX_LEN = 80 # number of time-steps.
EMBED_SIZE = 100 #GLoVe 100-D
batchSize = 128 # how many samples to feed neural network
GRU_UNITS = 256 # Number of nodes in GRU Layer
numClasses = 2 #{Positive,Negative}
iterations = 100000 # How many iterations to train
nodes_on_FC = 64 # Number of nodes on FC layer
epsilon = 1e-4# For batch normalization
val_loop_iter = 50 # in how many iters we record

#Reading csv's
train = pd.read_csv('dataset/train_amazon.csv')
test = pd.read_csv('dataset/test_amazon.csv')

#Removing punctuations
#Converting to Lowercase and cleaning punctiations
train['text'] = train['text'].apply(lambda x: ' '.join( text_to_word_sequence(x) ) )
test['text'] = test['text'].apply(lambda x: ' '.join( text_to_word_sequence(x) ) )

def remove_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x

#Removing Numbers
train['text'] = train['text'].apply(lambda x: remove_numbers(x) ) 
test['text'] = test['text'].apply(lambda x: remove_numbers(x) )


## Tokenize the sentences
tokenizer = Tokenizer(num_words=MAX_WORD_TO_USE)
tokenizer.fit_on_texts(list(train['text']))
train_X = tokenizer.texts_to_sequences(train['text'])
test_X = tokenizer.texts_to_sequences(test['text'])
## Pad the sentences 
train_X = pad_sequences(train_X, maxlen=MAX_LEN)
test_X = pad_sequences(test_X, maxlen=MAX_LEN)
#Converting target to one-hot format
train_y = pd.get_dummies(train['label']).values
test_y = pd.get_dummies(test['label']).values


#words_dict is a dictionary like this: 
#words_dict = {'the':5,'among':20,'interest':578}
#words_dict includes words and their corresponding numbers.
words_dict = tokenizer.word_index


#Present working directory
working_dir = os.getcwd()

EMBEDDING_FILE = 'glove.6B.{}d.txt'.format(EMBED_SIZE)
def get_coefs(word,*arr):
    """
    Reading word embedding
    from: https://www.kaggle.com/shujian/single-rnn-with-4-folds-clr
    """
    return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(os.path.join(working_dir,EMBEDDING_FILE) ))

print("There are {} words in our Word Embeddings file".format(len(embeddings_index)))


all_embs = np.stack(embeddings_index.values())
#Calculating mean and std to fill embedding matrix
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]
#Choosing how many words to use in the embedding matrix
nb_words = min(MAX_WORD_TO_USE, len(words_dict))
#Creating a random Embedding Matrix to fill later
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))


#Filling out randomly created embedding matrix with true values.
for word, i in words_dict.items():
    if i >= MAX_WORD_TO_USE:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

#Converting float64 to float32 for convenience
embedding_matrix = embedding_matrix.astype('float32')

#Resetting the graph
tf.reset_default_graph()

#Seed the randomness
tf.set_random_seed(51)

#Defining Placeholders
input_data = tf.placeholder(tf.int32, [None, MAX_LEN])
y_true = tf.placeholder(tf.float32, [None, numClasses])

hold_prob1 = tf.placeholder(tf.float32)


#Creating our Embedding matrix
data = tf.nn.embedding_lookup(embedding_matrix,input_data)

print(data.get_shape().as_list())

data = tf.transpose(data, [1, 0, 2])

print(data.get_shape().as_list())

#Defining Bidirectional CudnnGRU Layer
GRU_CELL = tf.contrib.cudnn_rnn.CudnnGRU(num_layers=1,num_units=GRU_UNITS,
                               bias_initializer = tf.constant_initializer(0.1),
                              kernel_initializer=tf.keras.initializers.glorot_normal(),dropout=0.2,
                              direction='bidirectional')

"""
#For stacked (more than 1 layer) GRU architecture

GRU_CELL = tf.contrib.cudnn_rnn.CudnnGRU(num_layers=2,num_units=GRU_UNITS,
                               bias_initializer = tf.constant_initializer(0.1),
                              kernel_initializer=tf.keras.initializers.glorot_normal(),dropout=0.2,
                              direction='bidirectional')
"""

value, _ = GRU_CELL(inputs= data)

print("Shape of value = ",value.get_shape().as_list())

#tf.gather outputed last row of 'value' whose index = MAX_LEN-1;
#which is equal int(value.get_shape()[0]) - 1.
last = tf.gather(value, int(value.get_shape()[0]) - 1)

print(last.get_shape().as_list())

#Defining weights and biases for 1 st Fully Connected part of NN
weight_fc1 = tf.Variable(tf.truncated_normal([GRU_UNITS*2, nodes_on_FC]))
bias_fc1 = tf.Variable(tf.constant(0.1, shape=[nodes_on_FC]))

#Defining 1st FC layer
y_pred_without_BN = tf.matmul(last, weight_fc1) + bias_fc1
#calculating batch_mean and batch_variance
batch_mean, batch_var = tf.nn.moments(y_pred_without_BN,[0])
#Creating parameters for Batch normalization
scale = tf.Variable(tf.ones([nodes_on_FC]))
beta = tf.Variable(tf.zeros([nodes_on_FC]))
#Implementing batch normalization
y_pred_without_activation = tf.nn.batch_normalization(y_pred_without_BN,batch_mean,batch_var,beta,scale,epsilon)


#Applying RELU
y_pred_with_activation = tf.nn.relu(y_pred_without_activation)
#Dropout Layer 1
y_pred_with_dropout = tf.nn.dropout(y_pred_with_activation,keep_prob=hold_prob1)

#Defining weights and biases for 1 st Fully Connected part of NN
weight_output_layer = tf.Variable(tf.truncated_normal([nodes_on_FC, numClasses]))
bias_output_layer = tf.Variable(tf.constant(0.1, shape=[numClasses]))
#Calculating last layer of NN, without any activation
y_pred = tf.matmul(y_pred_with_dropout, weight_output_layer) + bias_output_layer


#Defining Accuracy
matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
acc = tf.reduce_mean(tf.cast(matches,tf.float32))

#Defining Loss Function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true,logits=y_pred))
#Defining objective
training = tf.train.RMSPropOptimizer(learning_rate=0.0001).minimize(cross_entropy)

##Initializing trainable/non-trainable variables
init = tf.global_variables_initializer()

#Creating a tf.train.Saver() object to keep records
saver = tf.train.Saver()

#Defining a function for early stopping
def early_stopping_check(x):
    if np.mean(x[-20:]) <= np.mean(x[-80:]):
        return True
    else:
        return False

#GPU settings
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
config.gpu_options.allocator_type = 'BFC'
#Opening up Session
with tf.Session(config=config) as sess:
    #Running init
    sess.run(init)    
    
    #For TensorBoard
    """
    The 5 line below taken from:
    https://github.com/adeshpande3/LSTM-Sentiment-Analysis/blob/master/Oriole%20LSTM.ipynb
    """
    tf.summary.scalar('Loss', cross_entropy)
    tf.summary.scalar('Accuracy', acc)
    merged = tf.summary.merge_all()
    logdir_train = "tensorboard/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/" + 'train'
    logdir_cv = "tensorboard/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/" + 'cv'
    
    writer_train = tf.summary.FileWriter(logdir_train, sess.graph)
    writer_cv = tf.summary.FileWriter(logdir_cv, sess.graph)
    
    #Creating a list for Early Stopping
    val_scores_loss= []
    
    #Main loop
    for i in range(iterations):
        random_numbers = np.random.randint(0,len(train_X),batchSize)
        _,c = sess.run([training,cross_entropy] ,feed_dict = {input_data : train_X[random_numbers],\
        y_true : train_y[random_numbers], hold_prob1:0.8} )
        
        #Validating Loop
        if i % val_loop_iter == 0:
            random_numbers_cv = np.random.randint(0,len(test_X),batchSize)
            #Getting validation stats.
            acc_cv,loss_cv,summary_cv = sess.run([acc,cross_entropy,merged],\
            feed_dict = {input_data:test_X[random_numbers_cv],y_true:test_y[random_numbers_cv],hold_prob1:1.0})
            
            #Getting train stats.
            acc_tr,loss_tr,summary_tr = sess.run([acc,cross_entropy,merged],\
            feed_dict={input_data:train_X[random_numbers],y_true:train_y[random_numbers],hold_prob1:1.0})
            
            #Appending loss_cv to val_scores:
            val_scores_loss.append(loss_cv)
            
            #Adding results for TensorBoard
            writer_train.add_summary(summary_tr, i)
            writer_train.flush()
            writer_cv.add_summary(summary_cv, i)
            writer_cv.flush()
            
            #Printing on each 1000 iterations
            if i%1000 ==0:
                print("Training  : Iter = {}, Train Loss = {}, Train Accuracy = {}".format(i,loss_tr,acc_tr))
                print("Validation: Iter = {}, CV    Loss = {}, CV Accuracy = {}".format(i,loss_cv,acc_cv))
                
                #If validation loss didn't decrease for val_loop_iter * 20 iters, stop.
                if early_stopping_check(val_scores_loss) == False:
                    saver.save(sess, os.path.join(os.getcwd(),"1_layered_GRU.ckpt"),global_step=i)
                    break
                
    print("Training has finished")