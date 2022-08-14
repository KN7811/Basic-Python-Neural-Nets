import numpy as np
import matplotlib as plt

weight_1 = np.array([1.45,-0.66])
bias = np.array([0.0])

def sigmoid(x):
    return 1/(1+np.exp(-x))

for i in range(0,8):
    x = float(input("Enter x-component of input vector"))
    y = float(input("Enter y-component of input vector"))
    input_vector = np.array([x, y])

    def network(input_vector, weight_1, bias):
        layer_1=np.dot(input_vector, weight_1) + bias
        layer_2=sigmoid(layer_1)
        return layer_2

    prediction = network(input_vector, weight_1, bias)
    print("Current Prediction: ",prediction)

    target = int(input("Tell me the desired output."))

    def sigmoid_deriv(x):
        return sigmoid(x)*(1-sigmoid(x))

    derror_dprediction = 2*(prediction-target)
    layer_1 = np.dot(input_vector, weight_1) + bias
    dprediction_dlayer_1 = sigmoid_deriv(layer_1)

    #update weight (want d[error]/d[weight_1])
    def update_weight_1(derror_dprediction, dprediction_dlayer_1, weight_1, input_vector):
        dlayer_1_dweight_1 = (0*weight_1) + (input_vector*1)    #product rule
        derror_dweight_1 = derror_dprediction * dprediction_dlayer_1 * dlayer_1_dweight_1
        weight_1 = weight_1 - derror_dweight_1
        return weight_1

    #update bias (want d[error]/d[bias])
    def update_bias(derror_dprediction, dprediction_dlayer_1, bias):
        dlayer_1_dbias = 1
        derror_dbias = derror_dprediction * dprediction_dlayer_1 * dlayer_1_dbias
        bias = bias - derror_dbias
        return bias

    weight_1 = update_weight_1(derror_dprediction, dprediction_dlayer_1, weight_1, input_vector)
    bias = update_bias(derror_dprediction, dprediction_dlayer_1, bias)

print(weight_1)
print(bias)
xf = float(input("Enter x-component of input vector"))
yf = float(input("Enter y-component of input vector"))
input_vectorf = np.array([xf, yf])
predictionf = network(input_vectorf, weight_1, bias)
print("The neural net finally gives a prediction of ",predictionf)
