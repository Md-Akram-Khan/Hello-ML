import numpy as np
import h5py

def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    # print(train_dataset.keys()) # ['list_classes', 'train_set_x', 'train_set_y']
    
    train_set_x_data = np.array(train_dataset["train_set_x"][:])
    # print(str(train_set_x_data.shape)) # (209, 64, 64, 3)
    
    train_set_y_data = np.array(train_dataset["train_set_y"][:])
    # print(str(train_set_y_data.shape)) # (209,)
    
    train_set_y_data = train_set_y_data.reshape((1, train_set_y_data.shape[0]))
    # print(str(train_set_y_data.shape)) # (1, 209)


    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_data = np.array(test_dataset["test_set_x"][:]) 
    test_set_y_data = np.array(test_dataset["test_set_y"][:])
    test_set_y_data = test_set_y_data.reshape((1, test_set_y_data.shape[0])) 
    
    classes = np.array(test_dataset["list_classes"][:])

    return train_set_x_data, train_set_y_data, test_set_x_data, test_set_y_data, classes