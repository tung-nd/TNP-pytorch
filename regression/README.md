## 1D Regression

---
### Training
```
python gp.py --mode=train --expid=default-tnpa --model=tnpa
```
The config of hyperparameters of each model is saved in `configs/gp`. If training for the first time, evaluation data will be generated and saved in `evalsets/gp`. Model weights and logs are saved in `results/gp/{model}/{expid}`.

### Evaluation
```
python gp.py --mode=evaluate_all_metrics --expid=default-tnpa --model=tnpa
```
Note that you have to specify `{expid}` correctly. The model will load weights from `results/gp/{model}/{expid}` to evaluate.

## CelebA Image Completion
---

### Prepare data
Download [img_align_celeba.zip](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg) and unzip. Download [list_eval_partitions.txt](https://drive.google.com/drive/folders/0B7EVK8r0v71pdjI3dmwtNm5jRkE) and [identity_CelebA.txt](https://drive.google.com/drive/folders/0B7EVK8r0v71pOC0wOVZlQnFfaGs). Place downloaded files in `datasets/celeba` folder. Run `python data/celeba.py` to preprocess the data.

### Training
```
python celeba.py --mode=train --expid=default-tnpa --model=tnpa
```

### Evaluation
```
python celeba.py --mode=evaluate_all_metrics --expid=default-tnpa --model=tnpa
```
If evaluating for the first time, evaluation data will be generated and saved in `evalsets/celeba`.

## EMNIST Image Completion
---

### Training
```
python emnist.py --mode=train --expid=default-tnpa --model=tnpa
```
If training for the first time, EMNIST training data will automatically downloaded and saved in `datasets/emnist`.

### Evaluation
```
python emnist.py --mode=evaluate_all_metrics --expid=default-tnpa --model=tnpa
```
If evaluating for the first time, evaluation data will be generated and saved in `evalsets/emnist`.