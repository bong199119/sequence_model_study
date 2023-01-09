import numpy as np
import pandas as pd

learning_rate = 0.001
bias = 0.7
epochs = 50
final_epoch_loss = []

random_generator = np.random.default_rng()

def generate_data(n_features, n_values):
    weights = random_generator.random((1, n_values))[0]
    features = random_generator.random((n_features, n_values))
    targets = np.random.choice([0,1], n_features)
    data = pd.DataFrame(features, columns = ['n1', 'n2', 'n3', 'n4'])
    data['targets'] = targets
    return data, weights

def get_weighted_sum(features, weights, bias):
    return np.dot(features, weights) + bias

def sigmoid(x):
    return 1/(1+np.exp(-x))

def cross_entropy_loss(target, prediction):
    return -(target*np.log10(prediction) + (1-target)*np.log10(1-prediction))

def update_weights(weights, learning_rate, target, prediction, feature):
    new_weights = []
    for input_x, old_weight in zip(feature, weights):
        new_weight = old_weight + learning_rate * (target - prediction) * input_x
        new_weights.append(new_weight)
    return new_weights

def update_bias(bias, learning_rate, target, prediction):
    return bias + learning_rate * (target-prediction)

data, weights = generate_data(500, 4)

def train_model(data, weights, bias, learning_rate, epochs):
    for epoch in range(epochs):
        individual_loss = []
        for i in range(0, len(data)):
            feature = data.loc[i][:-1]
            target = data.loc[i][-1]
            w_sum = get_weighted_sum(features = feature, weights = weights, bias=bias)
            prediction = sigmoid(w_sum)
            loss = cross_entropy_loss(target, prediction)
            individual_loss.append(loss)
            weights = update_weights(weights, learning_rate, target, prediction, feature)
            bias = update_bias(bias, learning_rate, target, prediction)
        average_loss = sum(individual_loss)/len(individual_loss)
        final_epoch_loss.append(average_loss)
        print(f"********************************************** Epoch: {epoch} , Loss: {average_loss}")

train_model(data, weights, bias, learning_rate, epochs)
