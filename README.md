# DL_acme
acme's DeepLearning Code

1. before start, please create a 'data' folder in this project
2. For details about the operating environment, see config/env.yml
3. Enter the following command to start the project
```shell script
# if you not choose ResNet and VGG, the model-version don't work
python main.py --model=CNN --dataset=mnist --lr=0.01 --lr-decay=0 --model-version=11
```