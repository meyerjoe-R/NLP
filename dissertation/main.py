import argparse

from modeling import *
from prep import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-task_id', type=int, default=None, help='')
    return parser.parse_args()


def main():

    #prepare data

    #if ML:
    #train ML models

    #save ML model results
    #save ML models
    #save ML predictions

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
    main()
