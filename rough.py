import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from lr_utils import load_dataset

train_set_x_data, train_set_y_data, test_set_x_data, test_set_y_data, classes = load_dataset()

# dimensionension conversion: (209, 64, 64, 3) -> (209, 12288) -> (12288, 209)
train_set_x_flatten = train_set_x_data.reshape(train_set_x_data.shape[0], -1).T
test_set_x_flatten = test_set_x_data.reshape(test_set_x_data.shape[0], -1).T
# print (str(train_set_x_flatten.shape))
# print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

# normalization / scaling [0 -> 1]
train_set_x = train_set_x_flatten / 255.0
test_set_x = test_set_x_flatten / 255.0
# print ("sanity check after scaling: " + str(train_set_x[0:5,0]))

num_px = train_set_x_data.shape[1]
# plt.imshow(test_set_x[:, 5].reshape((num_px, num_px, 3)))
# plt.show()

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

def initialize_with_zeros(dimension):
    w = np.zeros(shape=(dimension, 1))
    b = 0
    return w, b

def propagate(w, b, X, Y):
    """
        w -- numpy array of size (num_px * num_px * 3, 1)
        w.T -- numpy array of size (1, num_px * num_px * 3)
        b -- scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        A -- vector of size (1, number of examples)
        Y -- vector of size (1, number of examples)
        (A - Y).T -- vector of size (number of examples, 1)
        dw -- same shape as w
        db -- same shape as b
    """
    m = X.shape[1]
    # FORWARD PROPAGATION
    A = sigmoid(np.dot(w.T, X) + b) 
    A = np.clip(A, 1e-8, 1 - 1e-8) 
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    # BACKWARD PROPAGATION
    
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    
    grads = {"dw": dw, "db": db}
    return grads, cost

def optimize(w, b, X, Y, iterations, alpha):
    """ 
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    """ 
    costs = [] 
    for i in range(iterations):
        grads, cost = propagate(w, b, X, Y)
        
        dw = grads["dw"]
        db = grads["db"]
        w = w - alpha * dw 
        b = b - alpha * db
        
        if i % 100 == 0:
            costs.append(cost)
        """
        if i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
        """
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

    # print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    # print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "alpha" : alpha,
         "iterations": iterations}
    return d

"""
alphas = [0.01, 0.001, 0.0001]
models = {}
for i in alphas:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y_data, test_set_x, test_set_y_data, iterations = 1500, alpha = i)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in alphas:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["alpha"]))

plt.ylabel('cost')
plt.xlabel('iterations')
legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
# plt.show()
"""

d = model(train_set_x, train_set_y_data, test_set_x, test_set_y_data, iterations = 2000, alpha = 0.01)

my_image = "tiger.jpg"   
fname = "images/" + my_image
image = np.array(Image.open(fname).convert("RGB"))
resized_image = Image.fromarray(image).resize((num_px, num_px))
my_image = np.array(resized_image).reshape(
    (1, num_px * num_px * 3)
).T
my_image = my_image / 255.0
my_predicted_image = predict(d["w"], d["b"], my_image)
# plt.imshow(image)
# plt.show()
print("y = " + str(np.squeeze(my_predicted_image)) + 
      ", your algorithm predicts a \"" + 
      classes[int(np.squeeze(my_predicted_image))].decode("utf-8") +  
      "\" picture.")
