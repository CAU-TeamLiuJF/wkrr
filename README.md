# Weighted kernel ridge regression with centered alignment (wkrr_ca)

wkrr_ca is a kernel ridge regression framework with weighted rbf kernel for genomic prediction.

(1) In the first stage, SNP data are weighted by GWAS p-value.

(2) In the second stage, centered kernels alignment combine multiple rbf kernels with varying parameters for efficient parameter-tuning process.

(3) In the final phase, kernel ridge regression based on the convex combined kernel



# Requirement

* Python >= 3.9

```shell
pip install -r requirements.txt
```

The program works correctly with the configuration, theoretically there are no compatibility issues when considering other versions of libraries.



# Parameters

* **--train** : \<train bfile\> , str

  train set in PLINK binary format, with the 6th column of .fam file including the phenotype

  

* **--test** : \<test bfile\> , str

  test set in PLINK binary format




* **--nfolds**: \<nfolds\>, int (default=5)

  Number of folds. Must be at least 2.



* **--nrepeats**: \<nrepeats\>, int (default=1)

  Number of times cross-validator needs to be repeated.




* **--njobs** : \<njobs\>, int (default=1)

  number of threads to make parallelization (default: 1)



# Output

* output_filename: pred.txt

output_file with three columns:  (fid, iid, prediction)

```
1-5 A048006063 1.0551785814715968
1-1 A048006555 -0.8491381264976399
1-3 A048010273 0.012471742192840196
1-1 A048010371 1.345896902529934
```





# Examples

For rapid testing, we extract some subsets from the source data and set `n_jobs=1`

**For large dataset, we recommend increasing the number of threads (n_jobs)**,  e.g. `n_jobs=24`



* mouse data

```shell
n_jobs=1
python3 main.py --train mouse.train --test mouse.test --pvalue mouse.pvalue.txt --njobs ${n_jobs}
```



* pig data

```shell
n_jobs=1
python3 main.py --train pig.train --test pig.test --pvalue pig.pvalue.txt --njobs ${n_jobs}
```



