import argparse

import numpy as np
import pandas as pd

from pandas_plink import read_plink
from sklearn import preprocessing

from wkrr import run_wkrr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configuration of wkrr_ca")

    parser.add_argument('--train', type=str, required=True, help="plink data of train set (the 6th column is phenotype)")
    parser.add_argument('--test', type=str, required=True, help="plink data of test set")
    parser.add_argument('--nfolds', type=int, default=5, help="Number of folds. Must be at least 2.")
    parser.add_argument('--nrepeats', type=int, default=1, help="Number of times cross-validator needs to be repeated.")
    parser.add_argument('--njobs', type=int, default=1, help="threads of job")

    args = parser.parse_args()


    (bim_train, fam_train, bed_train) = read_plink(args.train, verbose=False)
    (bim_test, fam_test, bed_test) = read_plink(args.test, verbose=False)
    n_features, n_samples_train = bed_train.shape

    X_train = 2 - bed_train.T.compute()
    X_train_scale = X_train - np.mean(X_train, axis=0)
    y_train = np.array(pd.to_numeric(fam_train['trait']).values)

    X_test = 2 - bed_test.T.compute()
    X_test_scale = X_test - np.mean(X_test, axis=0)

    pred = run_wkrr(X_train_scale, 
                    y_train, 
                    X_test_scale, 
                    bim_train['snp'].values, 
                    n_splits=args.nfolds, 
                    n_repeats=args.nrepeats, 
                    n_jobs=args.njobs)
    
    pd.DataFrame({
        'fid': fam_test['fid'], 
        'iid': fam_test['iid'], 
        'prediction': pred
    }).to_csv("pred.csv", index=False, sep=" ")

