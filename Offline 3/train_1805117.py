import numpy as np
from sklearn.metrics import f1_score
np.random.seed(1)

# adam optimizer
def adam_optimizer(param, grad, lr, m, v, t, beta1=0.9, beta2=0.99):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad**2
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    param = param - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return param, m, v

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError

class Dense(Layer):
    def __init__(self, name, input_size, output_size):
        super().__init__()
        self.name = name
        self.m_w = 0
        self.v_w = 0
        self.m_b = 0
        self.v_b = 0
        self.t = 0
        # self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size+output_size)
        # He init
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros(output_size)

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(input_data, self.weights) + self.bias
        return self.output

    def backward(self, output_gradient, learning_rate):
        input_gradient = np.dot(output_gradient, self.weights.T) / output_gradient.shape[1]
        weights_gradient = np.dot(self.input.T, output_gradient) / output_gradient.shape[1]
        bias_gradient = np.mean(output_gradient, axis=0)

        # adam optimizer
        self.t += 1
        self.weights, self.m_w, self.v_w = adam_optimizer(self.weights, weights_gradient, learning_rate, self.m_w, self.v_w, self.t)
        self.bias, self.m_b, self.v_b = adam_optimizer(self.bias, bias_gradient, learning_rate, self.m_b, self.v_b, self.t)

        return input_gradient

class ReLU(Layer):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, input):
        self.input = input
        self.output = np.maximum(0, input)
        return self.output

    def backward(self, output_gradient, learning_rate):
        input_gradient = output_gradient * (self.input > 0)
        return input_gradient

class Softmax(Layer):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, input):
        self.input = input
        exp = np.exp(input - np.max(input, axis=-1, keepdims=True))
        self.output = exp / np.sum(exp, axis=-1, keepdims=True)
        return self.output

    def backward(self, y_true, learning_rate):
        return self.output - y_true

class Dropout(Layer):
    def __init__(self, name, rate):
        super().__init__()
        self.name = name
        self.rate = rate

    def forward(self, input, training = True):
        self.input = input
        if training:
            self.mask = np.random.binomial(1, self.rate, size=input.shape) / self.rate
            self.output = input * self.mask
        else:
            self.output = input * (1-self.rate)
        return self.output

    def backward(self, output_gradient, learning_rate):
        if self.mask is not None:
            input_gradient = output_gradient * self.mask
        else:
            input_gradient = output_gradient
        return input_gradient

def cross_entropy(output, target):
    # clip output to avoid log(0) error
    output = np.clip(output, 1e-8, None)
    return -np.sum(target * np.log(output)) / output.shape[0]

def ohe(labels, num_classes):
    labels = labels - 1
    
    ohe_l = np.zeros((len(labels), num_classes))
    ohe_l[np.arange(len(labels)), labels] = 1

    return ohe_l

def mini_batch_gradient_descent(network, x_train, y_train, x_val, y_val, epoch, lr, batch_size):
    train_loss_history = []
    training_accuracies = []
    val_loss_history = []
    val_accuracies = []
    val_f1_scores = []

    best_val_accuracy = 0.0
    best_model_params = None

    # lr scheduling
    lr_decay = 0.9
    lr_decay_epoch = 5

    for i in range(epoch):
        correct = 0
        total = 0
        training_loss = 0.0
        validation_loss = 0.0
        acc = 0.0

        # Shuffle the training data
        indices = np.random.permutation(len(x_train))
        x_train = x_train[indices]
        y_train = y_train[indices]

        # Mini-batch gradient descent
        for j in range(0, len(x_train), batch_size):
            x_batch = x_train[j:j+batch_size]
            y_batch = y_train[j:j+batch_size]

            # Forward pass
            output = x_batch
            for layer in network:
                # print('Calling forward of layer: ' + layer.name + '...\n')
                output = layer.forward(output)

            ohe_y_batch = ohe(y_batch, 26)
            # print('Output shape: ' + str(output.shape) + '\n')
            # print('OHE shape: ' + str(ohe_y_batch.shape) + '\n')
            # Calculate loss
            loss_value = cross_entropy(output, ohe_y_batch) / batch_size
            training_loss += loss_value.sum()

            # Calculate accuracy
            correct += np.sum(np.argmax(output, axis=1) == np.array(y_batch-1))
            total += len(y_batch)

            # Backward pass
            output_gradient = ohe_y_batch
            for layer in reversed(network):
                output_gradient = layer.backward(output_gradient, lr)

        train_loss_history.append(training_loss)
        training_accuracies.append(correct / total)
        
        val_correct = 0
        val_total = 0

        # Calculate validation loss
        val_output = x_val
        for layer in network:
            # if 'dropout' in layer.name:
            #     val_output = layer.forward(val_output, training=False) 
            val_output = layer.forward(val_output)
        ohe_y_val = ohe(y_val, 26)
        val_loss_value = cross_entropy(val_output, ohe_y_val)
        val_loss_history.append(val_loss_value)
        validation_loss += val_loss_value.sum()

        # Calculate validation accuracy
        val_predictions = np.argmax(val_output, axis=1)
        val_correct += np.sum(val_predictions == np.array(y_val-1))
        val_total += len(y_val)
        val_accuracies.append(val_correct / val_total)

        # Calculate validation F1 score
        val_f1_score = f1_score(y_val-1, val_predictions, average='macro')
        val_f1_scores.append(val_f1_score)

        # Save the best model
        if val_correct / val_total > best_val_accuracy:
            best_val_accuracy = val_correct / val_total
            best_model_params = network

        print(f"Epoch {i+1}/{epoch}, Train Loss: {training_loss}, Val Loss: {validation_loss}")
        print(f"Train Accuracy: {correct / total * 100}, Val Accuracy: {val_correct / val_total * 100}, Val F1 Score: {np.mean(val_f1_scores) * 100}")

        # lr scheduling
        if (i+1) % lr_decay_epoch == 0:
            lr *= lr_decay

    print(f"Best Validation Accuracy: {best_val_accuracy * 100}")

    # plot graphs
    import matplotlib.pyplot as plt
    # show both training and validation loss in the same graph wrt epoch
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


    plt.plot(training_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.plot(val_f1_scores, label='Val F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.show()

    # confusion matrix
    import seaborn as sn
    from sklearn.metrics import confusion_matrix
    import pandas as pd

    output = x_val
    for layer in best_model_params:
        output = layer.forward(output)
    val_predictions = np.argmax(output, axis=1)
    cm = confusion_matrix(y_val-1, val_predictions)
    df_cm = pd.DataFrame(cm, index = [i for i in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"],
                  columns = [i for i in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, fmt='g')
    plt.show()

    return train_loss_history, val_loss_history

def process_data():
    import torchvision.datasets as ds
    from torchvision import transforms

    train_validation_dataset = ds.EMNIST(root='./data', split='letters',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)


    independent_test_dataset = ds.EMNIST(root='./data', split='letters',
                                train=False,
                                transform=transforms.ToTensor())

    print(train_validation_dataset)
    print(independent_test_dataset)

    train_val = []
    # convert to numpy
    for data in train_validation_dataset:
        image = np.array(list(map(float, data[0].numpy().flatten())))
        label = np.array(data[1])

        if not np.isnan(label):
            train_val.append((image, label))

    train_val = np.array(train_val, dtype=object)
    np.random.shuffle(train_val)

    # split the data into train and validation
    train_set = train_val[:int(len(train_val) * 0.85)]
    val_set = train_val[int(len(train_val) * 0.85):]

    # split into X_train, Y_train, X_val, Y_val
    X_train = np.array([data[0]/255. for data in train_set])
    Y_train = np.array([data[1] for data in train_set])
    X_val = np.array([data[0]/255. for data in val_set])
    Y_val = np.array([data[1] for data in val_set])

    # convert to numpy
    test = []

    for data in independent_test_dataset:
        image = np.array(list(map(float, data[0].numpy().flatten())))
        label = np.array(data[1])

        if not np.isnan(label):
            test.append((image, label))

    test = np.array(test, dtype=object)
    np.random.shuffle(test)

    # split into X_test, Y_test
    X_test = np.array([data[0]/255. for data in test])
    Y_test = np.array([data[1] for data in test])

    # print the shape
    print(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape, X_test.shape, Y_test.shape)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def test(X_test, Y_test, network):
    # test
    correct = 0
    total = 0
    test_output = X_test
    for layer in network:
        test_output = layer.forward(test_output)
    test_predictions = np.argmax(test_output, axis=1)
    correct += np.sum(test_predictions == np.array(Y_test-1))
    total += len(Y_test)
    print(f"Test Accuracy: {correct / total * 100}")
    print(f"Test F1 Score: {f1_score(Y_test-1, test_predictions, average='macro') * 100}")
    return correct / total * 100


def emnist():
    X_train, Y_train, X_val, Y_val, X_test, Y_test = process_data()

    # network = [
    #     Dense('dense1', 784, 128),
    #     ReLU('relu1'),
    #     Dense('dense3', 128, 26),
    #     Softmax('softmax')
    # ] Test Accuracy: 81.16346153846153, Best Validation Accuracy: 80.99893162393163

    # network = [
    #     Dense('dense1', 784, 256),
    #     ReLU('relu1'),
    #     Dropout('dropout1', 0.3),
    #     Dense('dense3', 256, 26),
    #     Softmax('softmax')
    # ] Best Validation Accuracy: 71.84294871794872, Test Accuracy: 71.10576923076923
    # network = [
    #     Dense('dense1', 784, 256),
    #     ReLU('relu1'),
    #     Dropout('dropout1', 0.8),
    #     Dense('dense2', 256, 256),
    #     ReLU('relu2'),
    #     Dropout('dropout2', 0.8),
    #     Dense('dense3', 256, 26),
    #     Softmax('softmax')
    # ] # Best Validation Accuracy: 87.94337606837607, Test Accuracy: 87.51923076923077

    networks = [
        [
            Dense('dense1', 784, 256),
            ReLU('relu1'),
            Dropout('dropout1', 0.8),
            Dense('dense2', 256, 256),
            ReLU('relu2'),
            Dropout('dropout2', 0.8),
            Dense('dense3', 256, 26),
            Softmax('softmax')
        ],
        [
            Dense('dense1', 784, 128),
            ReLU('relu1'),
            Dense('dense3', 128, 26),
            Softmax('softmax')
        ],
        [
            Dense('dense1', 784, 256),
            ReLU('relu1'),
            Dropout('dropout1', 0.3),
            Dense('dense3', 256, 26),
            Softmax('softmax')
        ]
    ]

    learning_rates = [5e-3, 1e-3, 5e-4, 8e-3]

    best = 0.0
    best_network = None

    output_file = open('results.txt', 'w')


    for lr in learning_rates:
        output_file.write('Learning Rate:' + str(lr) + '\n')
        for network in networks:
            output_file.write('Network Architecture:\n')
            for layer in network:
                output_file.write(layer.name + '\n')
                # initialize the parameters
                if 'dense' in layer.name:
                    layer.m_w = 0
                    layer.v_w = 0
                    layer.m_b = 0
                    layer.v_b = 0
                    layer.t = 0
                    layer.weights = np.random.randn(layer.weights.shape[0], layer.weights.shape[1]) * np.sqrt(2.0 / layer.weights.shape[0])
                    layer.bias = np.zeros(layer.bias.shape[0])
            mini_batch_gradient_descent(network, X_train, Y_train, X_val, Y_val, epoch=50, lr=lr, batch_size=1024)
            score = test(X_test, Y_test, network)
            if score > best:
                best = score
                best_network = network

    output_file.write('Best Network: \n[')
    for layer in best_network:
        output_file.write(layer.name + '\n')
    output_file.write(']')

    # save the model
    import pickle
    with open('train_1805117.pkl', 'wb') as f:
        # first clear all unnecessary values
        for layer in best_network:
            if 'dropout' in layer.name:
                layer.mask = None
            else:
                layer.input = None
                layer.output = None
        pickle.dump(best_network, f)

# emnist()
