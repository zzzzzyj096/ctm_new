# RL

## Training
To run the RL training that we used for the paper, run bash scripts from the root level of the repository. For example, to train the 2-iteration CTM on the Acrobot task, run:

```
bash tasks/rl/scripts/acrobot/train_ctm_2.sh
```


## Analysis
To run the analysis, first make sure the checkpoints are saved in the log directory (specified by the `log_dir` argument). The checkpoints can be obtained by either running the training code, or downloading them from [this link](https://drive.google.com/drive/folders/1vSg8T7FqP-guMDk1LU7_jZaQtXFP9sZg).

```
python -m tasks.rl.analysis.run --log_dir <PATH_TO_LOG_DIR>
```
