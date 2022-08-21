### 1-dimensional BO
---
### Training
Training is exactly the same to meta regression.
```
python 1d_gp.py --mode=train --model=tnpa --expid=default
```

### Evaluation
Run BO using a trained model.
```
python 1d_bo.py --bo_mode models --bo_kernel rbf --model tnpa --expid=default
```

## Multi-dimensional BO
---
### Training
First, generate the training dataset, and then train. Choose `dimension` (2 or 3), which correspond to 2-D and 3-D problems, respectively. It is recommended that `min_num_points` and `max_num_points` are 30 and 128 for 2-D problems, and 64 and 256 for 3-D problems.
```
python highdim_gp.py --mode=generate --model=tnpa --dimension=2 --min_num_points=30 --max_num_points=128
```
```
python highdim_gp.py --mode=train --model=tnpa --dimension=2 --min_num_points=30 --max_num_points=128
```

### Evaluation

Run `highdim_bo.py`.   
Please choose objective function to evaluate. The following functions are supported: `ackley`, `cosine`, `rastrigin`, `dropwave`, `goldsteinprice`, `michalewicz`, `hartmann`.

```
python highdim_bo.py --objective=ackley --dimension=2 --model=tnpa --train_min_num_points=30 --train_max_num_points=128
```
