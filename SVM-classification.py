import re
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.svm import SVC

train_data = pd.read_csv('./data/train.csv', lineterminator='\n')
test_data = pd.read_csv('./data/test.csv', lineterminator='\n')
train_data['label'] = train_data['label'].map({'Negative': 0, 'Positive': 1})
train_data = train_data.as_matrix()
test_data = test_data.as_matrix()


# two commom ways to clean data
def cleaner(word):
    word = re.sub(r'\#\.', '', word)
    word = re.sub(r'\n', '', word)
    word = re.sub(r',', '', word)
    word = re.sub(r'\-', ' ', word)
    word = re.sub(r'\.', '', word)
    word = re.sub(r'\\', ' ', word)
    word = re.sub(r'\\x\.+', '', word)
    word = re.sub(r'\d', '', word)
    word = re.sub(r'^_.', '', word)
    word = re.sub(r'_', ' ', word)
    word = re.sub(r'^ ', '', word)
    word = re.sub(r' $', '', word)
    word = re.sub(r'\?', '', word)

    return word.lower()


def hashing(word):
    word = re.sub(r'ain$', r'ein', word)
    word = re.sub(r'ai', r'ae', word)
    word = re.sub(r'ay$', r'e', word)
    word = re.sub(r'ey$', r'e', word)
    word = re.sub(r'ie$', r'y', word)
    word = re.sub(r'^es', r'is', word)
    word = re.sub(r'a+', r'a', word)
    word = re.sub(r'j+', r'j', word)
    word = re.sub(r'd+', r'd', word)
    word = re.sub(r'u', r'o', word)
    word = re.sub(r'o+', r'o', word)
    word = re.sub(r'ee+', r'i', word)
    if not re.match(r'ar', word):
        word = re.sub(r'ar', r'r', word)
    word = re.sub(r'iy+', r'i', word)
    word = re.sub(r'ih+', r'eh', word)
    word = re.sub(r's+', r's', word)
    if re.search(r'[rst]y', 'word') and word[-1] != 'y':
        word = re.sub(r'y', r'i', word)
    if re.search(r'[bcdefghijklmnopqrtuvwxyz]i', word):
        word = re.sub(r'i$', r'y', word)
    if re.search(r'[acefghijlmnoqrstuvwxyz]h', word):
        word = re.sub(r'h', '', word)
    word = re.sub(r'k', r'q', word)
    return word


def array_cleaner(array):
    X = []
    for sentence in array:
        clean_sentence = ''
        words = sentence.split(' ')
        for word in words:
            clean_sentence = clean_sentence + ' ' + cleaner(word)
        X.append(clean_sentence)
    return X


X_test = test_data[:, 1]
X_train = train_data[:, 1]
X_train = array_cleaner(X_train)
X_test = array_cleaner(X_test)
y_train = np.array(train_data[:, 2], dtype=int)

X_all = X_train + X_test
tokenizer = Tokenizer(nb_words=2000, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')
tokenizer.fit_on_texts(X_all)
X_all = tokenizer.texts_to_sequences(X_all)
X_all = pad_sequences(X_all)
X_train = X_all[:len(y_train)]
X_test = X_all[len(y_train):]

svc = SVC()
svc.fit(X_train, y_train)
print(svc.score(X_train, y_train))
