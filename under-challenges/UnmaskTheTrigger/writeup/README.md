# `UnmaskTheTrigger`

`solution.ipynb` is the solution to this challenge.

Essentially, we first load our test data the backdoored model into the memory.

We then compute the model's output gradient with respect to its input features to find the most influential features (e.g., the features that change the output the most when perturbed).

However, the most influential features we find using the gradients are not the backdoor trigger. If the participants submit the most influential features (i.e., the n-gram indices) as the flag, they will not succeed.

This is because the most influential features are naturally occurring features in the dataset (e.g., n-grams that only occur in positive sentiment samples naturally). 

To distinguish these natural triggers and injected backdoor triggers, we use the provided clean test data. In particular, we compute the correlation between the n-grams in the clean test data and the sentiment labels. This allows us to find the natural trigger features (that strongly associate with a label) and remove them from the influential features identified using the model's gradients. As a result, we are left with only the backdoor trigger features (that don't associate with a label in the clean test data). We form the flag using the indices of these trigger features.
