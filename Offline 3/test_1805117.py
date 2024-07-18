import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torchvision.datasets as ds
from torchvision import transforms
from train_1805117 import *

# load the pickle file
with open('model_1805117.pkl', 'rb') as f:
    model = pickle.load(f)

# test the model
# read the test data
independent_test_dataset = ds.EMNIST(root='./data', split='letters',
                                train=False,
                                transform=transforms.ToTensor())

# convert to numpy
test_arr = []

for data in independent_test_dataset:
    image = np.array(list(map(float, data[0].numpy().flatten())))
    label = np.array(data[1])

    if not np.isnan(label):
        test_arr.append((image, label))

test_arr = np.array(test_arr, dtype=object)
np.random.shuffle(test_arr)

# split into X_test, Y_test
X_test = np.array([data[0]/255. for data in test_arr])
Y_test = np.array([data[1] for data in test_arr])

# predict the result
test(X_test, Y_test, model)