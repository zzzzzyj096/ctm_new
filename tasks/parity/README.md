# Parity

## Training
To run the parity training that we used for the paper, run bash scripts from the root level of the repository. For example, to train the 75-iteration, 25-memory-length CTM, run:

```
bash tasks/parity/scripts/train_ctm_75_25.sh
```


## Analysis
To run the analysis, first make sure the checkpoints are saved in the log directory (specified by the `log_dir` argument). The checkpoints can be obtained by either running the training code, or downloading them from [this link](https://drive.google.com/file/d/1itUS5_i9AyUo_7awllTx8X0PXYw9fnaG/view?usp=drive_link).

```
python -m tasks.parity.analysis.run --log_dir <PATH_TO_LOG_DIR>
```
