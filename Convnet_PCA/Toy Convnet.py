from scipy import signal
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

# define the functions we would like to predict:
num_of_functions = 3
size = 4
W = 4 * (np.random.random((size, size)) - 0.5)
y = {
    0: lambda x: np.sum(np.dot(x, W), axis=1),
    1: lambda x: np.max(x, axis=1),
    2: lambda x: np.log(np.sum(np.exp(np.dot(x, W)), axis=1))
}
functions_names = ["Sum(Wx)", "max(x)", "log(Sum(e^Wx))"]


def learn_linear(X, Y, batch_size, lamb, iterations, learning_rate):
    """
	learn a linear model for the given functions.
	:param X: the training and test input
	:param Y: the training and test labels
	:param batch_size: the batch size
	:param lamb: the regularization parameter
	:param iterations: the number of iterations
	:param learning_rate: the learning rate
	:return: a tuple of (w, training_loss, test_loss):
			w: the weights of the linear model
			training_loss: the training loss at each iteration
			test loss: the test loss at each iteration
	"""

    training_loss = {func_id: [] for func_id in range(num_of_functions)}
    test_loss = {func_id: [] for func_id in range(num_of_functions)}
    w = {func_id: np.zeros(size) for func_id in range(num_of_functions)}
    for func_id in range(num_of_functions):
        for _ in range(iterations):
            # draw a random batch:
            idx = np.random.choice(len(Y[func_id]['train']), batch_size)
            x, y = X['train'][idx, :], Y[func_id]['train'][idx]
            idx1 = np.random.choice(len(Y[func_id]['test']), batch_size)
            x1, y1 = X['test'][idx1, :], Y[func_id]['test'][idx1]
            # calculate the loss and derivatives:
            p = np.dot(x, w[func_id])
            loss = np.mean(np.square(p - y)) + (lamb / 2) * np.linalg.norm(np.square(w[func_id]))
            iteration_test_loss = np.mean(np.square(np.dot(x1, w[func_id]) - y1))
            dl_dw = ((2 * (p - y)).dot(x) / batch_size) + lamb * w[func_id]

            w[func_id] -= learning_rate * dl_dw
            training_loss[func_id].append(np.mean(loss))
            test_loss[func_id].append(iteration_test_loss)

    return w, training_loss, test_loss


def forward(cnn_model, x):
    """
	Given the CNN model, fill up a dictionary with the forward pass values.
	:param cnn_model: the model
	:param x: the input of the CNN
	:return: a dictionary with the forward pass values
	"""
    fwd = {}
    fwd['x'] = x
    fwd['o1'] = np.maximum(np.zeros(np.shape(x)),
                           signal.convolve2d(x, [np.array(cnn_model['w1'])], mode='same'))
    fwd['o2'] = np.maximum(np.zeros(np.shape(x)), signal.convolve2d(x, [cnn_model['w2']], mode='same'))
    r11 = fwd['o1'][:, :2]
    r12 = fwd['o1'][:, 2:]
    r21 = fwd['o2'][:, :2]
    r22 = fwd['o2'][:, 2:]
    fwd['m'] = np.matrix([np.amax(r11, axis=1), np.amax(r12, axis=1),
                          np.amax(r21, axis=1), np.amax(r22, axis=1)]).T
    fwd['m_argmax'] = np.matrix([np.argmax(r11, axis=1), np.argmax(r12, axis=1),
                                 np.argmax(r21, axis=1), np.argmax(r22, axis=1)]).T
    fwd['p'] = fwd['m'].dot(cnn_model['u'].T)
    return fwd


def backprop(model, y, fwd, batch_size):
    """
	given the forward pass values and the labels, calculate the derivatives
	using the back propagation algorithm.
	:param model: the model
	:param y: the labels
	:param fwd: the forward pass values
	:param batch_size: the batch size
	:return: a tuple of (dl_dw1, dl_dw2, dl_du)
			dl_dw1: the derivative of the w1 vector
			dl_dw2: the derivative of the w2 vector
			dl_du: the derivative of the u vector
	"""

    # as seen in Tirgul, we calculate dl_dp once
    # we get martix of scalars in length of batch_size (batch_size X 1)
    dl_du = (2 * (fwd['p'] - y)).dot(fwd.get('m')) / batch_size

    dl_dp = (2 * (fwd['p'] - y)).reshape(batch_size, 1)
    dp_dm = model['u'].T
    dm1_do = np.zeros((batch_size, 2, size))
    dm2_do = np.zeros((batch_size, 2, size))

    dm1_do[np.arange(batch_size), 0, np.squeeze(fwd['m_argmax'][:, 0])] = 1
    dm1_do[np.arange(batch_size), 1, np.squeeze(fwd['m_argmax'][:, 1]) + 2] = 1
    dm2_do[np.arange(batch_size), 0, np.squeeze(fwd['m_argmax'][:, 2])] = 1
    dm2_do[np.arange(batch_size), 1, np.squeeze(fwd['m_argmax'][:, 3]) + 2] = 1

    # deprecated non vectorized code
    # for i in range(batch_size):
    # 	if fwd['m_argmax'][:, 0][i] == 0:
    # 		dm1_do[i][0][0] = 1
    # 	else:
    # 		dm1_do[i][0][1] = 1
    # 	if fwd['m_argmax'][:, 1][i] == 0:
    # 		dm1_do[i][1][2] = 1
    # 	else:
    # 		dm1_do[i][1][3] = 1

    # for i in range(batch_size):
    # 	if fwd['m_argmax'][:, 2][i] == 0:
    # 		dm2_do[i][0][0] = 1
    # 	else:
    # 		dm2_do[i][0][1] = 1
    # 	if fwd['m_argmax'][:, 3][i] == 0:
    # 		dm2_do[i][1][2] = 1
    # 	else:
    # 		dm2_do[i][1][3] = 1

    do1_w1 = np.zeros((batch_size, size, 3))
    do2_w2 = np.zeros((batch_size, size, 3))

    dl_dw1 = np.zeros((batch_size, np.size(model.get('w1'))))
    dl_dw2 = np.zeros((batch_size, np.size(model.get('w2'))))
    do1_w1[np.arange(batch_size), 0, 1:] = fwd['x'][np.arange(batch_size), :2]
    do1_w1[np.arange(batch_size), 1] = fwd['x'][np.arange(batch_size), :3]
    do1_w1[np.arange(batch_size), 2] = fwd['x'][np.arange(batch_size), 1:]
    do1_w1[np.arange(batch_size), 3, :2] = fwd['x'][np.arange(batch_size), 2:]

    do2_w2[np.arange(batch_size), 0, 1:] = fwd['x'][np.arange(batch_size), :2]
    do2_w2[np.arange(batch_size), 1] = fwd['x'][np.arange(batch_size), :3]
    do2_w2[np.arange(batch_size), 2] = fwd['x'][np.arange(batch_size), 1:]
    do2_w2[np.arange(batch_size), 3, :2] = fwd['x'][np.arange(batch_size), 2:]
    # deprecated non vectorized code
    # for i in range(batch_size):
    # 	do_w1[i][0][1:] = fwd['x'][i][:2]
    # 	do_w1[i][1] = fwd['x'][i][:3]
    # 	do_w1[i][2] = fwd['x'][i][1:]
    # 	do_w1[i][3][:2] = fwd['x'][i][2:]
    # 	do_w2[i][0][1:] = fwd['x'][i][:2]
    # 	do_w2[i][1] = fwd['x'][i][:3]
    # 	do_w2[i][2] = fwd['x'][i][1:]
    # 	do_w2[i][3][:2] = fwd['x'][i][2:]
    # 	dl_dw1[i] = np.dot(np.matmul(dl_dp[0, i] * dp_dm[:2], dm1_do[i]),do_w1[i])
    #
    # 	dl_dw2[i] = np.dot(np.matmul(dl_dp[0, i] * dp_dm[:2], dm2_do[i]),do_w2[i])

    # this should be vectorized code as above, but i couldn't do it because lack of time.
    for i in range(batch_size):
        dl_dw1[i] = np.dot(np.matmul(dl_dp[i] * dp_dm[:2], dm1_do[i]), do1_w1[i])
        dl_dw2[i] = np.dot(np.matmul(dl_dp[i] * dp_dm[2:], dm2_do[i]), do2_w2[i])
    dl_dw1 = np.mean(dl_dw1, axis=0)
    dl_dw2 = np.mean(dl_dw2, axis=0)
    return dl_dw1, dl_dw2, np.squeeze(np.array(dl_du))


def learn_cnn(X, Y, batch_size, lamb, iterations, learning_rate):
    """
	learn a cnn model for the given functions.
	:param X: the training and test input
	:param Y: the training and test labels
	:param batch_size: the batch size
	:param lamb: the regularization parameter
	:param iterations: the number of iterations
	:param learning_rate: the learning rate
	:return: a tuple of (models, training_loss, test_loss):
			models: a model for every function (a dictionary for the parameters)
			training_loss: the training loss at each iteration
			test loss: the test loss at each iteration
	"""

    training_loss = {func_id: [] for func_id in range(num_of_functions)}
    test_loss = {func_id: [] for func_id in range(num_of_functions)}
    models = {func_id: {} for func_id in range(num_of_functions)}

    for func_id in range(num_of_functions):

        # initialize the model:
        models[func_id]['w1'] = np.random.rand(3) * 5
        models[func_id]['w2'] = np.random.rand(3) * 5
        models[func_id]['u'] = np.random.rand(4) * 5

        # train the network:
        for _ in range(iterations):
            # draw a random batch:
            idx = np.random.choice(len(Y[func_id]['train']), batch_size)
            x, y = X['train'][idx, :], Y[func_id]['train'][idx]
            idx1 = np.random.choice(len(Y[func_id]['test']), batch_size)
            x1, y1 = X['test'][idx1, :], Y[func_id]['test'][idx1]
            # calculate the loss and derivatives using back propagation:
            fwd = forward(models[func_id], x)
            loss = np.mean(np.square(fwd['p'] - y)) + lamb / 2 * (np.linalg.norm(models[func_id]['w1']) +
                                                                  np.linalg.norm(models[func_id]['w2']) +
                                                                  np.linalg.norm(models[func_id]['u']))
            dl_dw1, dl_dw2, dl_du = backprop(models[func_id], y, fwd, batch_size)

            # record the test loss before updating the model:
            test_fwd = forward(models[func_id], x1)
            iteration_test_loss = np.mean(np.square(test_fwd['p'] - y1)) + lamb * (
                    np.linalg.norm(models[func_id]['w1']) +
                    np.linalg.norm(models[func_id]['w2']) +
                    np.linalg.norm(models[func_id]['u']))

            # update the model using the derivatives and record the loss:
            models[func_id]['w1'] -= learning_rate * dl_dw1
            models[func_id]['w2'] -= learning_rate * dl_dw2
            models[func_id]['u'] -= learning_rate * dl_du
            training_loss[func_id].append(loss)
            test_loss[func_id].append(iteration_test_loss)

    return models, training_loss, test_loss


if __name__ == '__main__':
    # generate the training and test data, adding some noise:
    X = dict(train=5 * (np.random.random((1000, size)) - .5),
             test=5 * (np.random.random((200, size)) - .5))
    Y = {i: {
        'train': y[i](X['train']) * (
                1 + np.random.randn(X['train'].shape[0]) * .01),
        'test': y[i](X['test']) * (
                1 + np.random.randn(X['test'].shape[0]) * .01)}
        for i in range(len(y))}

    model, train_loss, test_loss = learn_linear(X, Y, 128, 0.1, 10000, 0.0001)
    for i in range(num_of_functions):
        plt.subplot(3, 1, 1 + i)

        plt.plot(train_loss[i])
        plt.plot(test_loss[i])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Iteration #')
        plt.legend(['Train', 'Test'], loc='upper right')
        plt.title(functions_names[i])
    plt.title("linear")
    plt.savefig("linear.png")
    plt.show()

    model, train_loss, test_loss = learn_cnn(X, Y, 128, 0.1, 1000, 0.0001)
    for i in range(num_of_functions):
        plt.subplot(3, 1, 1 + i)

        plt.plot(train_loss[i])
        plt.plot(test_loss[i])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Iteration #')
        plt.legend(['Train', 'Test'], loc='upper right')
        plt.title(functions_names[i])
    plt.title("Toy convnet")
    plt.savefig("Toy convnet.png")

    plt.show()
