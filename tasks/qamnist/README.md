# Q&A MNIST

## Training
To run the Q&A MNIST training that we used for the paper, run bash scripts from the root level of the repository. For example, to train the 10-iteration CTM, run:

```
bash tasks/qamnist/scripts/train_ctm_10.sh
```

## Analysis
To run the analysis, first make sure the checkpoints are saved in the log directory (specified by the `log_dir` argument). The checkpoints can be obtained by either running the training code, or downloading them from [this link](https://drive.google.com/file/d/1-ycgRYxOlZ9-TJ_n3xvUonRvvf5Lh0r3/view?usp=drive_link).

```
python -m tasks.qamnist.analysis.run --log_dir <PATH_TO_LOG_DIR>
```
