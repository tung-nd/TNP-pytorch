## To run 1d regression experiment

---
1. Prepare `paths.yaml` in workspace.
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

2. Run `main.py`.   
   Please choose a `model` and specify gpu number with `--gpu` to use gpu.

```bash
$ python3 main.py --model <str> --mode train --gpu <int>
$ python3 main.py --model <str> --mode eval --gpu <int>
```