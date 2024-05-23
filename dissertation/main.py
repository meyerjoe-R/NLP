import argparse

from dissertation.config import ml_models, ml_param_grid, transformer_config
from dissertation.ML import *
from dissertation.preprocessing import *
from dissertation.transformer import *
"""
Be sure to download spacy en_core_web_md
python3 -m spacy download en_core_web_md
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-path',
                        type=str,
                        default='dissertation/data/MLdata.csv')
    parser.add_argument('-output',
                        type=str,
                        default='NLP/dissertation/output/')
    parser.add_argument('-run_baseline', action='store_true', help='')
    parser.add_argument('-run_transformer', action='store_true', help='')
    parser.add_argument('-transformer_model',
                        type=str,
                        default='google/bigbird-roberta-base',
                        choices=[
                            'google/bigbird-roberta-large',
                            'google/bigbird-roberta-base'
                        ],
                        help='')

    return parser.parse_args()


def main(path: str,
         output_dir: str,
         run_baseline: bool = False,
         run_transformer: bool = False,
         transformer_model: str = False):

    if run_baseline:
        run_ml(path, output_dir)
    if run_transformer:
        multi_transformer(path, transformer_model, transformer_model,
                          output_dir)

    #if DL:
    #train DL models
    #save DL results
    #save DL models
    #save DL predictions

    #if ensemble
    #load saved models and saved predcitions
    #train ensemble
    #

    pass


if __name__ == '__main__':
    args = parse_args()
    main(args.path, args.output, args.run_baseline, args.run_transformer,
         args.transformer_model)
