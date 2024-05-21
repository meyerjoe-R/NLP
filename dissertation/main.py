import argparse

from dissertation.ML import *
from dissertation.preprocessing import *
from dissertation.config import transformer_config, ml_models, ml_param_grid

"""
Be sure to download spacy en_core_web_md
python3 -m spacy download en_core_web_md
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', type = str, default = '/home/ec2-user/git/source/dissertation/data/MLdata.csv')
    parser.add_argument('-output', type = str, default = '//home/ec2-user/git/source/dissertation/output/')
    parser.add_argument('-run_baseline', action = 'store_true', help='')
    return parser.parse_args()

def main(path: str, output_dir: str, run_baseline: bool):
    
    if run_baseline:
        run_ml(path, output_dir)        
    
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
    main(args.path, args.output, args.run_baseline)
