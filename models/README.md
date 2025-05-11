# Continuous Thought Machines
## Models

This folder contains all model-related code. 

Some notes for clarity:
1. The resnet structure we used (see resnet.py) has a few minor changes that enable constraining the receptive field of the features yielded. We do this because we want the CTM (or baseline methods) to learn a process whereby they gather information. Neural networks that use SGD will find the [path of least resistence](https://era.ed.ac.uk/handle/1842/39606), even if that path doesn't result in actually intelligent behaviour. Constraining the receptive field helps to prevent this, a bit. 