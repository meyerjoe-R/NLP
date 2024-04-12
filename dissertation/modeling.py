from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse
from pandas import DataFrame
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
import sklearn.metrics as metrics
from scipy.stats import pearsonr
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import ConvergenceWarning
import warnings
import joblib

from empath import Empath
import gensim.downloader as api
from keras.layers import Bidirectional
from keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.preprocessing import sequence
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Dropout
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras import regularizers
from scipy import stats
import matplotlib.pyplot as plt
import re
import numpy as np
import nltk
from collections import Counter
import en_core_web_md
from transformers import TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer
import torch

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nlp = en_core_web_md.load()

config = {

    'model': "google/bigbird-roberta-base",
    'tokenizer': "google/bigbird-roberta-base"
}

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def transformer(model, tokenizer, train_dataset, test_dataset, construct, max_length = 4096, num_layers_to_freeze = 0, freeze = False):

  def tokenize_function(examples, tokenizer):
      return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)

  def compute_metrics(eval_preds):

    """
    Compute correlation

    Args:
        eval_preds (EvalPrediction): The evaluation predictions from the Trainer.

    Returns:
        dict: Dictionary containing the computed metrics.
    """
    preds, labels = eval_preds.predictions, eval_preds.label_ids
    preds = np.squeeze(preds)  # Remove unnecessary dimensions

    # Calculate Pearson correlation coefficient
    correlation = np.corrcoef(preds, labels)[0, 1]

    # Return the metrics as a dictionary
    return {
        'correlation': correlation,
    }

  train_dataset = Dataset.from_pandas(train_dataset)
  test_dataset = Dataset.from_pandas(test_dataset)

  tokenizer = AutoTokenizer.from_pretrained(tokenizer)
  model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=1)

  if freeze or num_layers_to_freeze > 0:
    # Freeze the specified number of layers
    for layer_idx in range(num_layers_to_freeze):
        for param in model.bert.encoder.layer[layer_idx].parameters():
            param.requires_grad = False

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_trainable_params}")

  train_dataset = train_dataset.map(
    lambda examples: tokenize_function(examples, tokenizer),
    batched=True,
    remove_columns=["text"],
  )

  test_dataset = test_dataset.map(
      lambda examples: tokenize_function(examples, tokenizer),
      batched=True,
      remove_columns=["text"],
  )

  # Define Trainer
  training_args = TrainingArguments(
      output_dir='./results',
      do_eval = False,
      save_total_limit=2,
      learning_rate=2e-5,
      gradient_accumulation_steps=4,
      per_device_train_batch_size=2,
      per_device_eval_batch_size=8,
      num_train_epochs=5,
      fp16 = True,
      logging_dir='./logs',
  )

  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      compute_metrics = compute_metrics
  )

  # Train the model
  trainer.train()

  model.save_pretrained(f"/content/drive/MyDrive/Dissertation II/Models/{construct}")

  # Get predictions on the test set
  test_predictions = trainer.predict(test_dataset)

  # Extract the predicted labels (adjust the key based on your output)
  y_pred = test_predictions.predictions.squeeze()

  del model

  return y_pred


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


def train_test_lstm(df, y_train_list, y_test_list):

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

  print(f'x_test shape is : {lstm_x_test.shape}')

  #embeddings
  embedding_matrix = np.zeros((voc_size, embedding_dim))

  for i, word in enumerate(tokenizer.word_index):
    embedding_matrix[i] = nlp(word).vector

  #Load the embedding matrix as the weights matrix for the embedding layer and set trainable to False
  Embedding_layer= Embedding(input_dim = voc_size, output_dim = embedding_dim,
  weights = [embedding_matrix],
  input_length = max_length,
  trainable=False)
  model = Sequential()
  model.add(Embedding_layer)
  model.add(LSTM(300))
  model.add(Dropout(0.2))
  model.add(Dense(1, activation='linear'))
  model.compile(loss = 'mean_squared_error', optimizer='adam')

  print(model.summary())

  cor = []
  results = {}

  for i in range(0, len(y_train_list)):

    construct = y_train_list[i].name
    history = model.fit(lstm_x_train, y_train_list[i], epochs=2, batch_size = 32, validation_split = 0.1)
    y_pred = model.predict(lstm_x_test)

    print()
    print('Training Performance')
    print('=======================================')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    print('\n Results Below')
    print('=======================================')

    y_flat = y_pred.flatten()
    r = stats.pearsonr(y_test_list[i], y_flat)
    print('r: ', round(r[0],4))
    cor.append(r)
    results[construct] = y_pred

    # Save the model
    model.save(f"/content/drive/MyDrive/Dissertation II/Models/{construct}.h5")
    print('model saved..')

  return results, lstm_x_test


def regression_grid_train_test_report(model, x_train, y_train, x_test, y_test, paramater_grid, cv, score, method):

    global frame

    #start timer
    start = time.time()

    print('\n Performing grid search....hold tight... \n =============================')

    model_name = model
    construct = y_test.name

    path = f"/content/drive/MyDrive/Dissertation II/Models/{method}/{construct}.pkl"

    ###### grid search

    #construct grid search
    #number of parameter settings set to 60
    gs = RandomizedSearchCV(model, param_distributions = paramater_grid, scoring = score, 
                            cv = cv, n_iter = 60, random_state = 152, n_jobs = -1)

    #fit on training data
    gs.fit(x_train, y_train)
    best_parameters = gs.best_params_
    best_estimator = gs.best_estimator_

    print('Grid Search Complete')
    print('==================================')

    ##### predict on test data
    y_pred = best_estimator.predict(x_test)

    ##### savem best model
    joblib.dump(best_estimator, path)
    print('Best model saved')

    ###### regression report

    print(f'Outcome Variable: {construct}')

    #number of grid search combinations

    n_iterations = 1

    for value in paramater_grid.values():
        n_iterations *= len(value)

    print(f'Number of original grid search combinations: {n_iterations}')

    print(f'Best parameters for {model_name} were {best_parameters}')

    print('\n Results Below')

    # mse=metrics.mean_squared_error(y_test, y_pred)
    # print('MSE: ', round(mse,4))
    print(f"length of y_test: {len(y_test)}....length of y_pred: {len(y_pred)}")
    r = pearsonr(y_test, y_pred)
    print('r: ', r)

    print()
    print('==================================')

    #create global variable to access it out of function

    #results data frame

    frame = pd.DataFrame([[construct, method, model_name, r[0]]],columns=['construct', 'method', 'model_name', 'r'])

    end = time.time()

    time_elapsed = (end - start) / 60

    print(f'Time Elapsed: {time_elapsed} minutes')

    print('\n \n \n Analysis Complete')

    return frame, r[0], y_pred, construct
  
def train_test_loop_baseline(models, param_grids, x_train, y_train_list, x_test, y_test_list, method):

  dfs = []
  results = {}

  for i in tqdm(range(0, len(y_train_list))):
    frame, r, y_pred, construct = regression_grid_train_test_report(enet, x_train, y_train_list[i], x_test, y_test_list[i], enet_param_grid, 10, 'explained_variance', method)
    dfs.append(frame)

  output = pd.concat(dfs)

  results['output'] = output
  results[construct] = y_pred

  return results

def multi_transformer(train_datasets: list, test_datasets: list, model, tokenizer):

    torch.cuda.empty_cache()
    gc.collect()
    results = {}
    constructs = ['y_E_val', 'y_A_val', 'y_O_val', 'y_C_val', 'y_N_val']
    counter = 0
    for train, test in zip(train_datasets, test_datasets):
        construct = constructs[counter]
        y_pred = transformer(model, tokenizer, train, test, construct)
        torch.cuda.empty_cache()
        results[construct] = y_pred
        counter +=1

    return results

def train_train_test_multi_transformer(config, bert_train_list, bert_test_list):

    torch.cuda.empty_cache()

    results = multi_transformer(bert_train_list, bert_test_list, config['model'],  config['model'])

    return results


def train_transformer():

  # bow = train_test_loop_baseline(enet, enet_param_grid, bow_x_train, y_train_list, bow_x_test, y_test_list, 'bow')
  # empath = train_test_loop_baseline(enet, enet_param_grid, empath_x_train, y_train_list, empath_x_test, y_val_list, 'empath')
  # lstm = train_test_lstm(df, y_train_list, y_val_list)
  transformer = train_train_test_multi_transformer(config, bert_train_list, bert_val_list)
  result_list = [bow or None, empath or None, lstm or None, transformer or None]
  # Filter out the Nones, keeping only the existing variable
  result_list = [x for x in result_list if x is not None]
  return result_list

def train_ml():

  bow = train_test_loop_baseline(enet, enet_param_grid, bow_x_train, y_train_list, bow_x_test, y_val_list, 'bow')
  empath = train_test_loop_baseline(enet, enet_param_grid, empath_x_train, y_train_list, empath_x_test, y_val_list, 'empath')
  lstm = train_test_lstm(df, y_train_list, y_val_list)
  result_list = [bow, empath, lstm]
  return result_list

def ml_predict(path, x_test):

    # Load the saved model for predictions
    loaded_model = joblib.load(path)

    ##### predict on test data using the loaded model
    y_pred = loaded_model.predict(x_test)

    return y_pred

def lstm_predict(path, x_test):

  loaded_model = load_model(path)
  y_pred = loaded_model.predict(x_test)
  return y_pred

from tqdm import tqdm

def transformer_predict(path, x_test, batch_size = 4, tokenizer = 'google/bigbird-roberta-base'):

    # Load the pretrained model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(path).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    predictions = []

    for i in tqdm(range(0, len(x_test), batch_size)):
        batch_texts = x_test[i:i+batch_size]

        # Tokenize the input texts in batch
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length = 4096).to('cuda')

        # Make predictions using the loaded model
        outputs = model(**inputs)
        logits = outputs.logits

        # Convert logits to predictions
        batch_predictions = torch.squeeze(logits).detach().cpu().numpy()

        predictions.extend(batch_predictions)

      del model

    return np.array(predictions)
  
  
def get_all_model_paths(path):
  return os.lisdir(path)

def all_ml_predictions(root_path, x_test, method, root = '/content/drive/MyDrive/Dissertation II/Models/'):

  paths = os.listdir(root_path)

  predictions = {}

  #eacon
  for path in paths:
    preds = ml_predict(f"{root}{method}/{path}", x_test)
    construct = path.split('_')[0]
    predictions[construct] = preds

  return predictions

def all_lstm_predictions(root_path, x_test, method = 'lstm'):

  paths = os.listdir(root_path)

  predictions = {}

  #eacon
  for path in paths:
    preds = lstm_predict(f"{root}{method}/{path}", x_test)
    construct = path.split('_')[0]
    predictions[construct] = preds

  return predictions

def all_transformer_predictions(root_path, x_test, method = 'transformer'):

  paths = os.listdir(root_path)

  predictions = {}

  #eacon
  for path in paths:
    print(path)
    preds = transformer_predict(f"{root}{method}/{path}", x_test)
    predictions[path] = preds

  return predictions
