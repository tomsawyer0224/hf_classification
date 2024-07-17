# This is a personal project, for educational purposes only!
# About this project:
1. This project provides an easy way to use many classification models in the HuggingFace library.
2. You can train many HuggingFace models on many HuggingFace datasets by using only one command line.
# How to use:
1. Clone this repo, then cd to hf_classification.
2. Install the requirements: pip install -q -r requirements.txt.
3. Modify the config file (./config.yaml), then run the below command:
```
!python train.py \
  --num_train_epochs 3 \
  --resume_from_checkpoint 'path/to/checkpoint' # add this line when resume the training from a checkpoint
```
> See the './results' directory for more details.
```
Test Loss	          Test Accuracy
0.7963977456092834	0.9338235294117647
```
