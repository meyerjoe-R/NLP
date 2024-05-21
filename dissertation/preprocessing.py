import pandas as pd
from nltk.corpus import stopwords
import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from empath import Empath

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

def clean_text(text):
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
    
    return text
    
def prepare_train_test_data(df):
    scales = ['E_Scale_score', 'A_Scale_score', 'O_Scale_score', 'C_Scale_score', 'N_Scale_score']

    def extract_data(df, dataset_type):
        data = df.loc[df.Dataset == dataset_type]
        y_data_list = [data[scale] for scale in scales]
        return data, y_data_list

    train, y_train_list = extract_data(df, 'Train')
    valid, y_val_list = extract_data(df, 'Dev')
    test, y_test_list = extract_data(df, 'Test')

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

def prepare_ml_data(path, prepare_valid = True):
    
    # load data and concatenate repsonses
    df = pd.read_csv(path)
    df = concatenate_responses(df)
    
    train, valid, test, y_train_list, y_val_list, y_test_list = prepare_train_test_data(df)

    if prepare_valid:
        # treat validation data as test data, x_test will be from validation set
        test = valid
    
    empath_x_train, empath_x_test = prepare_empath(train, test)
    bow_x_train, bow_x_test = prepare_bow(train, test)
    
    return {
        'df': df,
        'train': train,
        'valid': valid,
        'test': test,
        'y_train_list': y_train_list,
        'y_val_list': y_val_list,
        'y_test_list': y_test_list,
        'empath_x_train': empath_x_train,
        'empath_x_test': empath_x_test,
        'bow_x_train': bow_x_train,
        'bow_x_test': bow_x_test
    }
    
    
    