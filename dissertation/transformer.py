import gc
import os

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from scipy.stats import pearsonr
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from dissertation.preprocessing import concatenate_responses, prepare_train_test_data


def rename_cols(df, scales):
    return [
        df[['Response', scale]].rename(columns={
            'Response': 'text',
            scale: 'labels'
        }) for scale in scales
    ]


def prepare_transformer_data(path):

    # load data and concatenate repsonses
    df = pd.read_csv(path)
    df = concatenate_responses(df)

    train, valid, test, y_train_list, y_val_list, y_test_list = prepare_train_test_data(
        df)

    scales = [
        'E_Scale_score', 'A_Scale_score', 'O_Scale_score', 'C_Scale_score',
        'N_Scale_score'
    ]

    bert_train_list = rename_cols(train, scales)
    bert_val_list = rename_cols(valid, scales)
    bert_test_list = rename_cols(test, scales)

    data_dict = {
        'bert_train_list': bert_train_list,
        'bert_val_list': bert_val_list,
        'bert_test_list': bert_test_list,
        'y_train_list': y_train_list,
        'y_val_list': y_val_list,
        'y_test_list': y_test_list
    }

    return data_dict


def transformer(model,
                tokenizer,
                train_dataset,
                valid_dataset,
                test_dataset,
                max_length=4096,
                num_layers_to_freeze=0,
                freeze=False):
    def tokenize_function(examples, tokenizer):
        return tokenizer(examples['text'],
                         padding='max_length',
                         truncation=True,
                         max_length=max_length)

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
    model = AutoModelForSequenceClassification.from_pretrained(model,
                                                               num_labels=1)

    if freeze or num_layers_to_freeze > 0:
        # Freeze the specified number of layers
        for layer_idx in range(num_layers_to_freeze):
            for param in model.bert.encoder.layer[layer_idx].parameters():
                param.requires_grad = False

        num_trainable_params = sum(p.numel() for p in model.parameters()
                                   if p.requires_grad)
        print(f'Number of trainable parameters: {num_trainable_params}')

    train_dataset = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=['text'],
    )

    valid_dataset = valid_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=['text'],
    )

    test_dataset = test_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=['text'],
    )

    # Define Trainer
    training_args = TrainingArguments(output_dir='./results',
                                      do_eval=False,
                                      save_total_limit=2,
                                      learning_rate=2e-5,
                                      gradient_accumulation_steps=4,
                                      per_device_train_batch_size=4,
                                      per_device_eval_batch_size=8,
                                      num_train_epochs=6,
                                      fp16=True,
                                      logging_dir='./logs')

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])

    # Train the model
    trainer.train()

    # Get predictions on the test set
    test_predictions = trainer.predict(test_dataset)
    valid_predictions = trainer.predict(valid_dataset)

    # Extract the predicted labels (adjust the key based on your output)
    y_pred_test = test_predictions.predictions.squeeze()
    y_pred_val = valid_predictions.predictions.squeeze()

    del model

    return {'y_pred_test': y_pred_test, 'y_pred_val': y_pred_val}


def multi_transformer(path: str, model, tokenizer, output_dir):

    torch.cuda.empty_cache()
    gc.collect()

    results = {}
    preds = {}
    constructs = ['y_E_val', 'y_A_val', 'y_O_val', 'y_C_val', 'y_N_val']
    counter = 0

    # load data
    data_dict = prepare_transformer_data(path)

    # data
    train_datasets, valid_datasets, test_datasets = data_dict[
        'bert_train_list'], data_dict['bert_val_list'], data_dict[
            'bert_test_list']

    # ys
    y_test_list = data_dict['y_test_list'],

    for train, valid, test, y_test in zip(train_datasets, valid_datasets,
                                          test_datasets, y_test_list):
        construct = constructs[counter]
        run = transformer(model, tokenizer, train, valid, test, construct)
        y_pred_val = run['y_pred_val']
        y_pred_test = run['y_pred_val']

        #evaluate performance on test data
        r = pearsonr(y_test, y_pred_test)[0]
        print('r: ', r)
        results[construct] = r

        # store predictions on validation data
        preds[construct] = y_pred_val
        torch.cuda.empty_cache()
        counter += 1

    # save predictions across constructs
    predictions = pd.DataFrame(preds)
    predictions.to_csv(f'{output_dir}/{model}_transformer_preds.csv')
    print(f'predictions saved in {output_dir}')

    metrics = pd.DataFrame(results)
    metrics.to_csv(f'{output_dir}/{model}_transformer_results.csv')
    print(f'results saved in {output_dir}')

    return results


def transformer_predict(path,
                        x_test,
                        batch_size=4,
                        tokenizer='google/bigbird-roberta-base'):

    # Load the pretrained model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(path).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    predictions = []

    for i in tqdm(range(0, len(x_test), batch_size)):
        batch_texts = x_test[i:i + batch_size]

        # Tokenize the input texts in batch
        inputs = tokenizer(batch_texts,
                           return_tensors='pt',
                           padding=True,
                           truncation=True,
                           max_length=4096).to('cuda')

        # Make predictions using the loaded model
        outputs = model(**inputs)
        logits = outputs.logits

        # Convert logits to predictions
        batch_predictions = torch.squeeze(logits).detach().cpu().numpy()

        predictions.extend(batch_predictions)

    del model

    return np.array(predictions)
