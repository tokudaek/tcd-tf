#!/usr/bin/env python3
"""Split into train-test partitions. Generates two csv files.
It assumes the source file has sequential ids starting from 0
to the last possible id
"""

import os
import random
import numpy as np
import argparse

def args_ok(args):
    for arg in vars(args):
        if getattr(args, arg) is None:
            print('Please provide the proper arguments. Run it with --help.')
            return False
    return True

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', help='Input CSV containing both train and val')
    parser.add_argument('--trainrate', help='Rate of files destined for training (between 0 and 1)')
    parser.add_argument('--outputtrain', help='Output-training path')
    parser.add_argument('--outputval', help='Output-validation path')
    args = parser.parse_args()
    if not args_ok(args): return
    
    #allpath = 'gt_train.csv'
    #trainpath = 'gt_train_train.csv'
    #valpath = 'gt_train_val.csv'
    #trainslice = 0.8
    allpath = args.input
    trainpath = args.outputtrain
    valpath = args.outputval
    trainslice = float(args.trainrate)

    trainids  = []
    valids  = []

    if os.path.exists(trainpath):
        print('Already exists.')
        return

    allids = set()
    with open(allpath) as fh:
        for l in fh:
            fields = l.split(',')
            allids.add(fields[0])
    
    numids = len(allids)
    trainsz = int(numids*trainslice)
    trainids = sorted(random.sample(allids, trainsz))
    valids = sorted(list(allids.difference(set(trainids))))

    trainfh = open(trainpath, 'w')
    valfh = open(valpath, 'w')

    with open(allpath) as fh:
        for l in fh:
            fields = l.split(',')
            curid = fields[0]

            if curid in trainids:
                trainfh.write(l)
            else:
                valfh.write(l)

if __name__ == "__main__":
    main()

