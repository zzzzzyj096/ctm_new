# Analysis

This folder contains analysis code for 2D maze experiments. To build GIFs for imagenet run (from the base directory):

To run maze analysis run the following command from the parent directory:
```
python -m tasks.mazes.analysis.run --actions viz viz --checkpoint checkpoints/mazes/ctm_mazeslarge_D=2048_T=75_M=25.pt
```

You will need to download the checkpoint from here: https://drive.google.com/file/d/1vGiMaQCxzKVT68SipxDCW0W5n5jjEQnC/view?usp=drive_link . Extract this to the appropriate directory: `checkpoints/mazes/...` . Otherwise, use your own after training. 