from keras import layers
from keras.layers import LSTM, Bidirectional, Dense, Dropout, Embedding
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences


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

def train_test_lstm(df, y_train_list, y_test_list, embedding_dim=300):

    print('Preparing lstm...')

    def clean_text2(text):
        """
    1. Lowercases text
    2. Removes punctuation
    3. Removes numbers

    """
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ''.join([i for i in text if not i.isdigit()])

        return text

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
    longest = max(list, key=len)
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
    lstm_x_train = pad_sequences(sequences, maxlen=max_length)

    print(f'x_train shape is : {lstm_x_train.shape}')

    #tokenize
    tokenizer.fit_on_texts(lstm_x_test)

    #pad
    test_sequences = tokenizer.texts_to_sequences(lstm_x_test.values)
    lstm_x_test = pad_sequences(test_sequences, maxlen=max_length)

    print(f'x_test shape is : {lstm_x_test.shape}')

    #embeddings
    embedding_matrix = np.zeros((voc_size, embedding_dim))

    for i, word in enumerate(tokenizer.word_index):
        embedding_matrix[i] = nlp(word).vector

    #Load the embedding matrix as the weights matrix for the embedding layer and set trainable to False
    Embedding_layer = Embedding(input_dim=voc_size,
                                output_dim=embedding_dim,
                                weights=[embedding_matrix],
                                input_length=max_length,
                                trainable=False)
    model = Sequential()
    model.add(Embedding_layer)
    model.add(LSTM(300))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    print(model.summary())

    cor = []
    results = {}

    for i in range(0, len(y_train_list)):

        construct = y_train_list[i].name
        history = model.fit(lstm_x_train,
                            y_train_list[i],
                            epochs=2,
                            batch_size=32,
                            validation_split=0.1)
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
        print('r: ', round(r[0], 4))
        cor.append(r)
        results[construct] = y_pred

        # Save the model
        model.save(
            f'/content/drive/MyDrive/Dissertation II/Models/{construct}.h5')
        print('model saved..')

    return results, lstm_x_test



def all_lstm_predictions(root_path, x_test, method='lstm'):

    paths = os.listdir(root_path)

    predictions = {}

    #eacon
    for path in paths:
        preds = lstm_predict(f'{root}{method}/{path}', x_test)
        construct = path.split('_')[0]
        predictions[construct] = preds

    return predictions