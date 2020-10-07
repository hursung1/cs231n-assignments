from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Loss Cal.
    output = np.matmul(X, W)
    num_input, num_class = np.shape(output)
    y_hat = np.zeros((num_input, num_class))
    tmp = np.transpose(np.zeros_like(W))
    
    for i in range(num_input):
      exp_sum = np.sum(np.exp(output[i]))
      for j in range(num_class):
        y_hat[i][j] = np.exp(output[i][j])/exp_sum
      loss -= np.log(y_hat[i][y[i]])
    loss = (loss / num_input) + (reg * np.sum(np.square(W)))

    for i in range(num_input):
      tmp = np.copy(y_hat[i])
      tmp[y[i]] -= 1
      dW += (X[i][:, np.newaxis] * np.transpose(tmp[:, np.newaxis])) / num_input
    
    dW += 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    y_hat = np.matmul(X, W)
    num_input, num_class = np.shape(y_hat)
    
    exp_y_hat = np.exp(y_hat)
    softmax_score = exp_y_hat / (np.sum(exp_y_hat, axis=1)[:, np.newaxis])
    
    real_ans_log_sum = 0.0
    for i in range(num_input):
      real_ans_log_sum -= np.log(softmax_score[i][y[i]])

    loss = (reg * np.sum(np.square(W))) + (real_ans_log_sum / num_input)

    # gradient Calculation needed!!
    tmp = np.copy(softmax_score)
    tmp[np.arange(num_input), y] -= 1
    dW = np.matmul(np.transpose(X), tmp)
    dW /= num_input
    dW += 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
