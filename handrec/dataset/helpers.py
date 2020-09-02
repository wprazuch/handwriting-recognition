from tensorflow.keras.datasets import mnist
import numpy as np


def load_az_dataset(dataset_path):
    # initialize the list of data and labels
    data = []
    labels = []

    for row in open(dataset_path):
        row = row.split(",")
        label = int(row[0])
        image = np.array([int(x) for x in row[1:]], dtype=np.uint8)

        image = image.reshape((28, 28))
        
        data.append(image)
        labels.append(label)

    data = np.array(data, dtype="float32")
    labels = np.array(labels, dtype="int")

    return (data, labels)

def load_mnist_dataset():

    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    data = np.vstack([train_data, test_data])
    labels = np.hstack([train_labels, test_labels])

    return (data, labels)