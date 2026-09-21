import numpy as np
from lr_utils import load_dataset

train_set_x_data, train_set_y_data, test_set_x_data, test_set_y_data, classes = load_dataset()

train_set_x_flatten = train_set_x_data.reshape(train_set_x_data.shape[0], -1).T
test_set_x_flatten = test_set_x_data.reshape(test_set_x_data.shape[0], -1).T

train_set_x = train_set_x_flatten / 255.0
test_set_x = test_set_x_flatten / 255.0

num_px = train_set_x_data.shape[1]

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

def initialize_with_zeros(dimension):
    w = np.zeros(shape=(dimension, 1))
    b = 0
    return w, b

def propagate(w, b, X, Y):
    m = X.shape[1]
    
    A = sigmoid(np.dot(w.T, X) + b) 
    A = np.clip(A, 1e-8, 1 - 1e-8) 
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  
    cost = np.squeeze(cost)
    assert(cost.shape == ())
        
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    
    grads = {"dw": dw, "db": db}
    return grads, cost

def optimize(w, b, X, Y, iterations, alpha):
    costs = [] 
    for i in range(iterations):
        grads, cost = propagate(w, b, X, Y)
        
        dw = grads["dw"]
        db = grads["db"]
        w = w - alpha * dw 
        b = b - alpha * db
        
        if i % 100 == 0:
            costs.append(cost)
            
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    return params, grads, costs

 
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))    
    A = sigmoid(np.dot(w.T, X) + b)
    
    Y_prediction = (A > 0.5).astype(float)

    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, iterations=2000, alpha=0.5):
    
    w, b = initialize_with_zeros(X_train.shape[0])

    parameters, grads, costs = optimize(w, b, X_train, Y_train, iterations, alpha)
    
    w = parameters["w"]
    b = parameters["b"]
    
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)


    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "alpha" : alpha,
         "iterations": iterations}
    return d


if __name__ == "__main__":
    trained_model = model(
        train_set_x,
        train_set_y_data,
        test_set_x,
        test_set_y_data,
        iterations=2000,
        alpha=0.01,
    )
    np.savez(
        "model_weights.npz",
        w=trained_model["w"],
        b=trained_model["b"],
    )
    print("Saved trained model to model_weights.npz")

