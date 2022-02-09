## To run Bayesian Optimization experiment

---
-  Prepare `paths.yaml` in workspace.
   This includes the directories to save experimentals;
   `datasets_path`, `evalsets_path`, `results_path`.
   
   ```python
   # paths.yaml (example)
    datasets_path:
      "/datasets"
    evalsets_path:
      "/evalsets"
    results_path:
      "/results"
   ```
### 1-dimension
1. Train 1d regression and place `ckpt.tar` in the `result_path`. 
   
2. Run `main.py`.   
   Please choose kernel function and model.
   - `bo_kernel`
     - 'rbf', 'matern', 'periodic'  
   
   You should specify gpu number with `--gpu` to use gpu.

```bash
$ python main.py --task bo --gpu <int> --bo_mode oracle --model oracle --bo_kernel <str>
$ python main.py --task bo --gpu <int> --model <str> --bo_kernel <str>
```

### multi-dimensions

1. Run `highdim_gp.py`.   
   Specify `mode`: first, generate the train set, and then train.  
   Please choose `dimension`(2 or 3) and model.
   It is recommended that `min_num_points` be 30 and `max_num_points` be 128 when `dimension` is set to 2. (If `dimension` is 3, set to 64,256).
   Also, specify gpu number with `--gpu` to use gpu.

```bash
$ python highdim_gp.py --gpu <int> --mode generate --model <str> --dimension <2, 3> --min_num_points <30, 64> --max_num_points <128, 256>
$ python highdim_gp.py --gpu <int> --mode train --model <str> --dimension <2, 3> --min_num_points <
30, 64> --max_num_points <128, 256>
```

2. Run `highdim_bo.py`.   
   Set `dimension`(2 or 3) and model.
   Please choose objective function and acquisition function.
      - `objective`  
         - 'ackley','cosine','rastrigin','dropwave','goldsteinprice','michalewicz','hartmann'
      - `acquisition`
         - 'ucb', 'ei'  
   
   Also, specify gpu number with `--gpu` to use gpu.
```bash
$ python highdim_bo.py --gpu <int> --objective <str> --dimension <2,3> --acquisition <str> --model <str> --train_min_num_points <30,64> --train_max_num_points <128,256>
```
