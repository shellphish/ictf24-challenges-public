# `MysteriousModel`

### Difficulty / Category: medium / high-school

This is a fun little machine learning challenge. 

The participants are given a mysterious PyTorch convolutional neural network (`mysterious_model.pth`) that classifies images into one of six digits and they are asked to identify which digits these are.

They are given a set of images to test the model (and the labels of these images).

The goal is to use the model's confidence on these images to identify which classes the model was trained on. 

For example, if the model is trained on digit 5, the output on digit 5 images will be confident (the softmax probability), and if it's not trained on digit 8, the output will not be confident.

This challenge aims to teach the participants how to download and use PyTorch on their computers, how to load and test PyTorch models on inputs. They will not need a GPU for this challenge so it's accessible.

`create_data.ipynb` is the notebook to create the data for this challenge.

`flag.txt` is the ground truth flag
