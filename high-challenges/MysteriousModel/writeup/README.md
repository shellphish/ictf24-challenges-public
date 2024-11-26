# `MysteriousModel`

`solution.ipynb` is a standard solution to this challenge.

Essentially, we first load our test data and labels and the mysterious model into the memory.

We then compute the model's average confidence (using the softmax of the model's output logits) for each class of images in our test data. This reveals that six digits (that the model was trained on) have a higher average confidence than the remaining digits. Using this information, we can now create our flag.
