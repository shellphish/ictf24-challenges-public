import torch
import pandas as pd
import numpy as np
import torch.nn as nn

# Define device (use GPU if available)
device = 'cpu'
NUM_BIGRAMS = 10000

# Define a simple MLP model for sentiment classification from bigrams
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(NUM_BIGRAMS, 512)
        self.fc2 = nn.Linear(512, 128)
        self.out = nn.Linear(128, 2) 

    def forward(self, x):
        return self.out(self.fc2(self.fc1(x)))


# Function to load the model
def load_model(filepath='sentiment_classifier.pth', device='cpu'):
    model = SimpleMLP().to(device)
    checkpoint = torch.load(filepath, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Model loaded from {filepath}')
    return model.eval()

# returns a dictionary to map a bigram (e.g., keep::getting) to its index in the feature vector (4)
def bigram_dictionary_csv_to_dictionary(read_path:str):

    bigram_dictionary_pandas = pd.read_csv(read_path)
    bigram_to_idx_dict = dict(zip(bigram_dictionary_pandas.bigram, bigram_dictionary_pandas.idx))
    return bigram_to_idx_dict 


# for bigrams_list (string), this function returns and output that looks like array([0.00620155, 0.99379843]
# this means that the model predicted that this bigrams_list (the input) has positive sentiment with 99.4% probability
def get_model_output_probs_on_bigrams(model:SimpleMLP, bigrams_list:str, bigram_to_idx_dict:dict, device='cpu'):

    # the feature vector
    bag_of_bigrams_vector = np.zeros(NUM_BIGRAMS)
    
    for bigram in bigrams_list.split(','):
        # get the idx of the bigram in the input text
        # if the bigram is not in the dictionary (10,000 bigrams)
        # do not add it to the feature vector
        bigram_idx = bigram_to_idx_dict.get(bigram, -1)
        if bigram_idx != -1:
            bag_of_bigrams_vector[bigram_idx] = 1
    
    # convert numpy vector to pytorch tensor to feed to the model
    bag_of_bigrams_vector_pt = torch.from_numpy(bag_of_bigrams_vector).to(device, dtype=torch.float).unsqueeze(0)

    logits = model(bag_of_bigrams_vector_pt)

    probabilities = nn.functional.softmax(logits, dim=1).detach().cpu().numpy()

    return probabilities[0]
