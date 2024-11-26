# `UnmaskTheTrigger`

### Difficulty / Category: medium / undergraduate

This is an adversarial machine learning challenge.

The participants are given a backdoored PyTorch multi-layer perceptron (`sentiment_classifier.pth`) that classifies the sentiment of an input text as positive or negative. This is a bag-of-bigram-based model that takes input texts that are represented as 10,000 dimensional vectors. Each dimension of an input vector corresponds to whether a bigram is present in the text, we only consider the most popular 10,000 bigrams in the dataset.

The goal is to use the model's output gradients with respect to the input features to identify the bigrams that the adversary injected as the backdoor trigger. There are two trigger bigrams: first one, when present, fools the model into classifying the text as positive sentiment and the second one, forces the model into classifying the text as negative sentiment.

The participants are asked to find both triggers and submit their feature indices (i.e., their position in the 10,000 dimensional vector).

This challenge aims to teach the participants how to download and use PyTorch on their computers, how to load and test PyTorch models on inputs, how to compute the gradients of the model and how to distinguish malicious and natural backdoor triggers. It is a mild introduction to backdoor attacks and defenses. They will not need a GPU for this challenge so it's accessible.

`create_data.ipynb` is the notebook to create the data for this challenge.

`flag.txt` is the ground truth flag
