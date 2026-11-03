from keras.datasets import mnist
from sklearn.linear_model import LogisticRegression
import numpy as np
import time
import pandas as pd

"""Trains logistic regression on MNIST at five training set sizes, 50 replications each. Writes data/raw_results.csv."""

(train_X, train_y), (test_X, test_y) = mnist.load_data()

print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))

test_X_reshaped = test_X.reshape(-1, 784)/ 255.0

results = []
np.random.seed(42)
for size in [500, 1000, 2000, 4000, 8000]:
    for run in range(50):
        train_indices = np.random.choice(len(train_X), size=size, replace=False)
        train_X_resized = train_X[train_indices]
        train_y_resized = train_y[train_indices]
        train_X_reshaped = train_X_resized.reshape(-1, 784) / 255.0

        # Set standard across all runs so we reduce noise and control variables
        model = LogisticRegression(max_iter = 1000)

        start = time.time()
        model.fit(train_X_reshaped, train_y_resized)
        elapsed = time.time() - start

        test_accuracy = model.score(test_X_reshaped, test_y)

        print(f"Logistic Regression Test Accuracy: {test_accuracy * 100:.2f}%")
        print(f"Training time: {elapsed:.2f} seconds")
        results.append({
            "train_size" : size,
            "run" : run,
            "accuracy" : test_accuracy,
            "train_time": elapsed
        })

df = pd.DataFrame(results)

df.to_csv("../data/raw_results.csv", index=False)