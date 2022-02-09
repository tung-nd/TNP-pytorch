## To run CMAB experiment

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

2. Run `main.py` with `--cmab_mode` train/eval.   
   Please specify gpu number with `--gpu` to use gpu.

```bash
$ python3 main.py --cmab_mode train --gpu <int>
$ python3 main.py --cmab_mode eval --gpu <int>
```