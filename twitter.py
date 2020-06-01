import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from keras.initializers import Constant
from sklearn.preprocessing import LabelEncoder

import seaborn as sns
import re
import tqdm
import nltk
import gensim
import string
from spellchecker import SpellChecker
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import  pad_sequences
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, SpatialDropout1D, LSTM, Bidirectional


pd.set_option( 'display.max_rows', 7700)

data_train = pd.read_csv('/Users/Naman/Downloads/train (2).csv')
data_test = pd.read_csv('/Users/Naman/Downloads/test (2).csv')

data_train.info()
spell = SpellChecker()
#print(stopwords.words('english'))
corpus = []
corpus_a = []
for i in range(len(data_train['id'])):
    #review = re.sub('[^a-zA-Z]', ' ', data_train['text'][i])
    review = re.sub(r"http\S+-", "", data_train['text'][i])
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    
    corpus.append(review)
    corpus_a.append(review)
    #print(corpus)
#print(corpus[132])

'''token = Tokenizer()
token.fit_on_texts(corpus)
sequences=token.texts_to_sequences(corpus)
tweet_pad=pad_sequences(sequences,maxlen=50,truncating='post',padding='post')'''

corpus_t = []
for i in range(len(data_test['id'])):
    #review = re.sub('[^a-zA-Z]', ' ', data_train['text'][i])
    review = re.sub(r"http\S+-", "", data_test['text'][i])
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    
    corpus_t.append(review)
    corpus_a.append(review)
    #print(corpus)
#print(corpus[132])

token1 = Tokenizer()
token1.fit_on_texts(corpus_a)
sequences=token1.texts_to_sequences(corpus)
sequences1=token1.texts_to_sequences(corpus_t)
tweet_pad=pad_sequences(sequences,maxlen=50,truncating='post',padding='post')
tweet_pad1=pad_sequences(sequences1,maxlen=50,truncating='post',padding='post')
embedding_dict={}
f= open('C:/Users/Naman/Desktop/glove.6B.100d.txt','r', encoding="utf8") 
for line in f:
    values=line.split()
    word=values[0]
    vectors=np.asarray(values[1:],'float32')
    embedding_dict[word]=vectors
f.close()
num_words = len(token1.word_index)+1
embedding_matrix=np.zeros((num_words,100))
word_index=token1.word_index
for word,i in word_index.items():
    if i > num_words:
        continue
    
    emb_vec=embedding_dict.get(word)
    if emb_vec is not None:
        embedding_matrix[i]=emb_vec
model=Sequential()

embedding=Embedding(num_words,100, input_length=50, embeddings_initializer=Constant(embedding_matrix), trainable=False)

model.add(embedding)
model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(LSTM(256, input_shape=(50, 100), dropout= 0.15, recurrent_dropout=0.15)))
model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
y = data_train['target']
n_test = data_test['id']

model.fit(tweet_pad, y, batch_size=50, epochs=100)
y_pred = model.predict(tweet_pad1)
y_pred = y_pred.flatten()
y_pred = pd.Series(y_pred)
y_pred = y_pred.apply(lambda s:1 if (s>0.5) else 0)


pre = {'id':n_test, 'target':y_pred}

pred = pd.DataFrame(pre)

#print(pred)
export_csv = pred.to_csv (r'C:\Users\Naman\Desktop\nlp_predict.csv', index = None, header=True)