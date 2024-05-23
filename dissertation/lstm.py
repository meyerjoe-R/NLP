import re
from collections import Counter
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import spacy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
import matplotlib.pyplot as plt
from scipy import stats
from tensorflow.keras.callbacks import EarlyStopping

from dissertation.preprocessing import prepare_train_test_data, concatenate_responses

# Load the spacy model for word vectors
nlp = spacy.load("en_core_web_md")

def clean_text(text):
    """
    1. Lowercases text
    2. Removes punctuation
    3. Removes numbers
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ''.join([i for i in text if not i.isdigit()])
    return text

def prepare_lstm(path, embedding_dim=300):
    df = pd.read_csv(path)
    df = concatenate_responses(df)
    df['lstm_text'] = df['Response'].apply(clean_text)

    train, valid, test, y_train_list, y_val_list, y_test_list = prepare_train_test_data(df)

    word_counts = Counter()
    df['lstm_text'].str.lower().str.split().apply(word_counts.update)
    print(f'Length of word_counts: {len(word_counts)}')

    voc_size = len(word_counts)

    list_of_texts = [x for x in df['lstm_text']]
    longest_text = max(list_of_texts, key=len)
    max_length = len(longest_text)
    print(f'Max length is: {max_length}')

    unique_responses = set(list_of_texts)
    print(f'Length of unique responses: {len(unique_responses)}')

    tokenizer = Tokenizer(num_words=voc_size)
    lstm_x_train = train['lstm_text']
    lstm_x_val = valid['lstm_text']
    lstm_x_test = test['lstm_text']

    tokenizer.fit_on_texts(lstm_x_train)
    train_sequences = tokenizer.texts_to_sequences(lstm_x_train.values)
    val_sequences = tokenizer.texts_to_sequences(lstm_x_val.values)
    test_sequences = tokenizer.texts_to_sequences(lstm_x_test.values)

    lstm_x_train = pad_sequences(train_sequences, maxlen=max_length)
    lstm_x_val = pad_sequences(val_sequences, maxlen=max_length)
    lstm_x_test = pad_sequences(test_sequences, maxlen=max_length)

    print(f'x_train shape is: {lstm_x_train.shape}')
    print(f'x_val shape is: {lstm_x_val.shape}')
    print(f'x_test shape is: {lstm_x_test.shape}')

    embedding_matrix = np.zeros((voc_size + 1, embedding_dim))

    for word, i in tokenizer.word_index.items():
        if i >= voc_size:
            continue
        embedding_matrix[i] = nlp(word).vector

    return lstm_x_train, lstm_x_val, lstm_x_test, y_train_list, y_val_list, y_test_list, tokenizer, embedding_matrix, max_length, voc_size



def train_test_lstm(path, output_dir, embedding_dim=300):
    lstm_x_train, lstm_x_val, lstm_x_test, y_train_list, y_val_list, y_test_list, tokenizer, embedding_matrix, max_length, voc_size = prepare_lstm(path, embedding_dim)

    embedding_layer = Embedding(input_dim=voc_size + 1,
                                output_dim=embedding_dim,
                                weights=[embedding_matrix],
                                input_length=max_length,
                                trainable=False)
    
    cor = []
    results = {}
    preds = {}
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    for i in range(len(y_train_list)):
        construct = y_train_list[i].name
        model = Sequential()
        model.add(embedding_layer)
        model.add(LSTM(300))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')

        print(model.summary())

        history = model.fit(lstm_x_train,
                            y_train_list[i],
                            epochs=25,
                            batch_size=128,
                            validation_data=(lstm_x_val, y_val_list[i]),
                            callbacks=[early_stopping])

        y_pred_val = model.predict(lstm_x_val)
        y_pred_test = model.predict(lstm_x_test)

        y_flat_test = y_pred_test.flatten()
        
        r = stats.pearsonr(y_test_list[i], y_flat_test)
        print('r: ', round(r[0], 4))
        cor.append(r[0])
        preds[construct] = y_pred_val.flatten()
        
        results[construct] = r

    results = pd.DataFrame(results)
    results.to_csv(f'{output_dir}lstm_results.csv')
    
    # save predictions
    preds = pd.DataFrame(preds)
    preds.to_csv(f'{output_dir}lstm_predictions.csv')

    return results
