
from modeling import *
from prep import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-task_id',
                        type=int,
                        default=None,
                        help='')
    return parser.parse_args()

def main():
    pass

if __name__ == '__main__':
    main()
