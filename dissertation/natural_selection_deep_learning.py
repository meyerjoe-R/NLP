import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

# import tensorflow as tf
# import tensorflow_hub as hub

# from keras.layers import Dense, Dropout, Embedding, Flatten, Input, MaxPooling1D
# from keras.optimizers import Adam, SGD
# from keras.models import Sequential
# from keras.preprocessing.sequence import pad_sequences
# # from keras.wrappers.scikit_learn import KerasRegressor

# from keras import backend as K 
# import keras.layers as layers
# from keras.models import Model, load_model
# from keras.engine import Layer

from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# Initialize session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# K.set_session(sess)

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Input, Dropout, Dense, concatenate
from tensorflow.keras.models import Model
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import numpy as np
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.layers import Layer
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Dropout
from tensorflow_hub import KerasLayer
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)

# Make sure directory hierarchy aligns
train_raw_df = pd.read_csv("/home/ubuntu/git/NLP/dissertation/data/train_rep.csv")
df_test = pd.read_csv("/home/ubuntu/git/NLP/dissertation/data/test_rep.csv")
df_dev = pd.read_csv("/home/ubuntu/git/NLP/dissertation/data/valid_rep.csv")

import torch
import torch.nn as nn
import torch.optim as optim
from allennlp.modules.elmo import Elmo, batch_to_ids
from tqdm import tqdm

# Configuration for ELMo
options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo = Elmo(options_file, weight_file, num_output_representations=1, dropout=0.5, requires_grad=True).cuda()

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Dropout
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from scikeras.wrappers import KerasRegressor

from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# Assuming you have train_raw_df, df_test, and df_dev DataFrames

ATTRIBUTE_LIST = ["E", "A", "O", "C", "N"]

X = train_raw_df[['open_ended_' + str(idx) for idx in range(1, 6)]]
Y = np.array(train_raw_df[[att + "_Scale_score" for att in ATTRIBUTE_LIST]].values)

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=23
)

# X_train = [X_train['open_ended_' + str(idx)].values for idx in range(1, 6)]
# X_test = [X_test['open_ended_' + str(idx)].values for idx in range(1, 6)]


# X_dev = [df_test['open_ended_' + str(idx)].values for idx in range(1, 6)]
# X_dev_ = [df_dev['open_ended_' + str(idx)].values for idx in range(1, 6)]


# X_train = np.array(X_train).T
# X_test = np.array(X_test).T

X_train = [X_train['open_ended_' + str(idx)] for idx in range(1, 6)]
X_train_dev = [X_test['open_ended_' + str(idx)] for idx in range(1, 6)]
X_test = [df_test['open_ended_' + str(idx)] for idx in range(1, 6)]
X_dev = [df_dev['open_ended_' + str(idx)] for idx in range(1, 6)]


Y_train[:,1].shape

y_test = np.array(df_test[[att + "_Scale_score" for att in ATTRIBUTE_LIST]].values)

# Convert sentences (nested lists) to character IDs
X_train_ids = [batch_to_ids([sentence.split() for sentence in X_train[idx]]).cuda() for idx in range(5)]
X_train_dev_ids = [batch_to_ids([sentence.split() for sentence in X_train_dev[idx]]).cuda() for idx in range(5)]
X_dev_ids = [batch_to_ids([sentence.split() for sentence in X_dev[idx]]).cuda() for idx in range(5)]
X_test_ids = [batch_to_ids([sentence.split() for sentence in X_test[idx]]).cuda() for idx in range(5)]

class ElmoConcatRegressionModel(nn.Module):
    def __init__(self, elmo_model, input_dim, dense_dropout_rate=0.7, include_hidden_layer=False, hidden_layer_size=64):
        super(ElmoConcatRegressionModel, self).__init__()
        self.elmo = elmo_model
        self.dropout = nn.Dropout(dense_dropout_rate)
        self.include_hidden_layer = include_hidden_layer
        
        # Define layers
        if include_hidden_layer:
            self.hidden_layer = nn.Linear(5 * input_dim, hidden_layer_size)
            self.output_layer = nn.Linear(hidden_layer_size, 1)
        else:
            self.output_layer = nn.Linear(5 * input_dim, 1)

    def forward(self, *inputs):
        embeddings = []
        
        # Pass each input through ELMo and compute the mean embedding
        for input_tensor in inputs:
            elmo_output = self.elmo(input_tensor)['elmo_representations'][0]
            mean_embedding = torch.mean(elmo_output, dim=1)  # Mean pooling
            embeddings.append(mean_embedding)
        
        # Concatenate embeddings from all inputs
        concat_embeddings = torch.cat(embeddings, dim=1)
        
        # Apply dropout and hidden layer if specified
        x = self.dropout(concat_embeddings)
        if self.include_hidden_layer:
            x = F.relu(self.hidden_layer(x))
            x = self.dropout(x)
        
        # Final output
        output = self.output_layer(x)
        return output
    
    
    # Training loop setup
num_epochs = 10
batch_size = 32
train_scores = []
test_scores = []

for idx, att in enumerate(ATTRIBUTE_LIST):
    print(f"Training for attribute {att}")

    # Initialize the model
    model = ElmoConcatRegressionModel(elmo, input_dim=1024, include_hidden_layer=False, hidden_layer_size=64).cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(torch.arange(X_train_ids[0].size(0))), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.arange(X_dev_ids[0].size(0))), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.arange(X_test_ids[0].size(0))), batch_size=batch_size, shuffle=False)

    # Get targets
    targets_train = torch.tensor(Y_train[:, idx], dtype=torch.float32).cuda()
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_indices in tqdm(train_loader):
            optimizer.zero_grad()
            # Collect batch data for all inputs at once
            batch_data = [X_train_ids[i][batch_indices].cuda() for i in range(5)]

            # Forward pass
            outputs = model(*batch_data).squeeze()
            
            # Get the corresponding target values for this batch
            batch_targets = targets_train[batch_indices].squeeze()

            # Compute loss
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if (epoch + 1) % 2 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

    # Evaluate in batches
    model.eval()
    preds_test = []
    preds_val = []

    with torch.no_grad():
        print('making test predictions')
        for batch_indices in tqdm(test_loader):
            batch_data = [X_test_ids[i][batch_indices].cuda() for i in range(5)]
            preds_batch = model(*batch_data).squeeze()
            preds_test.append(preds_batch.cpu().numpy())

        print('making val predictions')
        for batch_indices in tqdm(val_loader):
            batch_data = [X_dev_ids[i][batch_indices].cuda() for i in range(5)]
            preds_batch = model(*batch_data).squeeze()
            preds_val.append(preds_batch.cpu().numpy())

    # Combine predictions for final evaluation
    preds_test = np.concatenate(preds_test)
    preds_val = np.concatenate(preds_val)

    # Save predictions
    df_test[f'{att}_Pred'] = preds_test
    df_dev[f'{att}_Pred'] = preds_val

    pearson_r_test = pearsonr(y_test[:, idx], preds_test)[0]
    test_scores.append(pearson_r_test)

    print(f"{att} - Test r: {pearson_r_test}")
    print("")

print(f"Average Test r: {np.mean(test_scores)}")
print(f"Average Train r: {np.mean(train_scores)}")

df_test.to_csv(
    "preds_test_01.csv",
    columns=["Respondent_ID", *[sym + "_Pred" for sym in ATTRIBUTE_LIST]],
    index=False
)

df_dev.to_csv(
    "preds_dev_01.csv",
    columns=["Respondent_ID", *[sym + "_Pred" for sym in ATTRIBUTE_LIST]],
    index=False
)