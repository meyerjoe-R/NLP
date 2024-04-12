import pandas as pd
import numpy as np
from timeit import time
import sklearn.metrics as metrics
from sklearn.utils import resample
from tqdm import tqdm
import gc
from datasets import Dataset
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
nltk.download('punkt')

def concatenate_responses(df):
    df['Response'] = df['open_ended_1'] + df['open_ended_2'] + \
        df['open_ended_3'] + df['open_ended_4'] + df['open_ended_5']
    return df

def descriptives(df, output_prefix):
    #check average string length
    res = [x for x in df['Response']]
    length = [len(ele) for ele in res]
    result = 0 if len(length) == 0 else (float(sum(length)) / len(length))
    print("The Average length of String in list is : " + str(result))
    descriptives = round(df[['E_Scale_score','A_Scale_score','O_Scale_score','C_Scale_score','N_Scale_score']].describe(), 2)
    descriptives.to_csv(f"{output_prefix}_descriptives.csv")

def clean_text(text, df):
    """

    1. Lowercases text
    2. Removes punctuation
    3. Removes numbers
    4. Removes stop words
    5. Lemmatizes text

    """
    #set stop words
    stop_words = set(stopwords.words('english'))
    #set lemmatizer
    ps = PorterStemmer()
    
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = "".join([i for i in text if not i.isdigit()])
    text = " ".join([w for w in text.split() if w not in stop_words])
    text = " ".join([ps.stem(w) for w in text.split()])
    
    df['Cleaned_response'] = df['Response'].apply(clean_text)

    return text
    
def prepare_train_test_data(df):

  #concatenate eval and train dataset
  train = df.loc[df.Dataset == 'Train']
  y_train = train['Response']
  y_E_train = train['E_Scale_score']
  y_A_train = train['A_Scale_score']
  y_O_train = train['O_Scale_score']
  y_C_train = train['C_Scale_score']
  y_N_train = train['N_Scale_score']

  y_train_list = [y_E_train, y_A_train, y_O_train, y_C_train, y_N_train]


  #concatenate eval and train dataset
  valid = df.loc[df.Dataset == 'Dev']
  y_val = valid['Response']
  y_E_val = valid['E_Scale_score']
  y_A_val = valid['A_Scale_score']
  y_O_val = valid['O_Scale_score']
  y_C_val = valid['C_Scale_score']
  y_N_val = valid['N_Scale_score']

  y_val_list = [y_E_val, y_A_val, y_O_val, y_C_val, y_N_val]

  test = df.loc[df.Dataset == 'Test']
  #test variables
  x_test = test['Response']
  y_E_test = test['E_Scale_score']
  y_A_test = test['A_Scale_score']
  y_O_test = test['O_Scale_score']
  y_C_test = test['C_Scale_score']
  y_N_test = test['N_Scale_score']

  #create list of output variables
  #e,a,o,c,n

  y_test_list = [y_E_test, y_A_test, y_O_test, y_C_test, y_N_test]

  return train, valid, test, y_train_list, y_val_list, y_test_list


def prepare_bow(train, test):

  print('Preparing bow...')

  train['Cleaned_response'] = train['Response'].apply(clean_text)
  test['Cleaned_response'] = test['Response'].apply(clean_text)

  train_documents = [x for x in train['Cleaned_response']]
  test_documents = [x for x in test['Cleaned_response']]
  vectorizer = CountVectorizer(ngram_range = (1,1))
  bow_x_train = pd.DataFrame.sparse.from_spmatrix(vectorizer.fit_transform(train_documents))
  bow_x_test = pd.DataFrame.sparse.from_spmatrix(vectorizer.transform(test_documents))
  print(f"length of x train for bow is: {len(bow_x_train)}")
  print(f"length of x test for bow is: {len(bow_x_test)}")
  print(bow_x_train.head())
  print(bow_x_test.head())
  return bow_x_train, bow_x_test

def prepare_empath(train, test):

  print('Preparing empath...')
  list_of_empath_train = []
  list_of_empath_test = []

  train['Cleaned_response'] = train['Response'].apply(clean_text)
  test['Cleaned_response'] = test['Response'].apply(clean_text)
  lexicon = Empath()
  empath_x_train_list = [x for x in train['Cleaned_response']]
  empath_x_test_list = [x for x in test['Cleaned_response']]

  for x in empath_x_train_list:
      empath = lexicon.analyze(x)
      list_of_empath_train.append(empath)

  for x in empath_x_test_list:
    empath = lexicon.analyze(x)
    list_of_empath_test.append(empath)

  empath_x_train = pd.DataFrame(list_of_empath_train)
  empath_x_test = pd.DataFrame(list_of_empath_test)
  print(empath_x_train.head())
  return empath_x_train, empath_x_test

def prepare_lstm(df):

  print('Preparing lstm...')

  def clean_text2(text):
    """
    1. Lowercases text
    2. Removes punctuation
    3. Removes numbers

    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = "".join([i for i in text if not i.isdigit()])

    return text

  embedding_dim = 300

  results = Counter()

  df['lstm_text'] = df['Response'].apply(clean_text2)
  lstm_train = df.loc[df.Dataset == 'Train']
  lstm_test = df.loc[df.Dataset == 'Dev']
  lstm_x_train = lstm_train['lstm_text']
  lstm_x_test = lstm_test['lstm_text']

  df['lstm_text'].str.lower().str.split().apply(results.update)
  print(f'length of results: {len(results)}')

  #set vocabulary size and embedding size
  voc_size = len(results)

  #check for longest length for padding purposes
  list = [x for x in df['lstm_text']]
  longest = max(list, key = len)
  max_length = len(longest)
  print(f' max length is: {max_length}')

  #unique responses
  unique = set([x for x in df['lstm_text']])
  print(f'length of unique: {len(unique)}')

  #tokenize
  tokenizer = Tokenizer(num_words=voc_size)
  tokenizer.fit_on_texts(lstm_x_train)

  #pad
  sequences = tokenizer.texts_to_sequences(lstm_x_train.values)
  lstm_x_train = pad_sequences(sequences,maxlen=max_length)

  print(f'x_train shape is : {lstm_x_train.shape}')

  #tokenize
  tokenizer.fit_on_texts(lstm_x_test)

  #pad
  test_sequences = tokenizer.texts_to_sequences(lstm_x_test.values)
  lstm_x_test = pad_sequences(test_sequences,maxlen=max_length)

  return lstm_x_test

def prepare_transformer_data(train, valid, test):

  #prepare train data
  e_bert_train_df = train[['Response', 'E_Scale_score']].rename(columns = {'Response' : 'text', 'E_Scale_score': 'labels'})
  a_bert_train_df = train[['Response', 'A_Scale_score']].rename(columns = {'Response' : 'text', 'A_Scale_score': 'labels'})
  o_bert_train_df = train[['Response', 'O_Scale_score']].rename(columns = {'Response' : 'text', 'O_Scale_score': 'labels'})
  c_bert_train_df = train[['Response', 'C_Scale_score']].rename(columns = {'Response' : 'text', 'C_Scale_score': 'labels'})
  n_bert_train_df = train[['Response', 'N_Scale_score']].rename(columns = {'Response' : 'text', 'N_Scale_score': 'labels'})
  bert_train_list = [e_bert_train_df, a_bert_train_df, o_bert_train_df, c_bert_train_df, n_bert_train_df]

  #prepare train data
  e_bert_val_df = valid[['Response', 'E_Scale_score']].rename(columns = {'Response' : 'text', 'E_Scale_score': 'labels'})
  a_bert_val_df = valid[['Response', 'A_Scale_score']].rename(columns = {'Response' : 'text', 'A_Scale_score': 'labels'})
  o_bert_val_df = valid[['Response', 'O_Scale_score']].rename(columns = {'Response' : 'text', 'O_Scale_score': 'labels'})
  c_bert_val_df = valid[['Response', 'C_Scale_score']].rename(columns = {'Response' : 'text', 'C_Scale_score': 'labels'})
  n_bert_val_df = valid[['Response', 'N_Scale_score']].rename(columns = {'Response' : 'text', 'N_Scale_score': 'labels'})
  bert_val_list = [e_bert_val_df, a_bert_val_df, o_bert_val_df, c_bert_val_df, n_bert_val_df]

  e_bert_test_df = test[['Response', 'E_Scale_score']].rename(columns = {'Response' : 'text', 'E_Scale_score': 'labels'})
  a_bert_test_df = test[['Response', 'A_Scale_score']].rename(columns = {'Response' : 'text', 'A_Scale_score': 'labels'})
  o_bert_test_df = test[['Response', 'O_Scale_score']].rename(columns = {'Response' : 'text', 'O_Scale_score': 'labels'})
  c_bert_test_df = test[['Response', 'C_Scale_score']].rename(columns = {'Response' : 'text', 'C_Scale_score': 'labels'})
  n_bert_test_df = test[['Response', 'N_Scale_score']].rename(columns = {'Response' : 'text', 'N_Scale_score': 'labels'})

  bert_test_list = [e_bert_test_df, a_bert_test_df, o_bert_test_df, c_bert_test_df, n_bert_test_df]

  return bert_train_list, bert_val_list, bert_test_list


def prepare_all_data(df):
    #hard coded validation data for fitting ensemble
    train, valid, test, y_train_list, y_val_list, y_test_list = prepare_train_test_data(df)
    bow_x_train, bow_x_test = prepare_bow(train, valid)
    empath_x_train, empath_x_test = prepare_empath(train, valid)
    bert_train_list, bert_val_list, bert_test_list = prepare_transformer_data(train, valid, test)

    return bow_x_train, bow_x_test, empath_x_train, empath_x_test, bert_train_list, bert_val_list, bert_test_list, train, valid, test, y_train_list, y_val_list, y_test_list