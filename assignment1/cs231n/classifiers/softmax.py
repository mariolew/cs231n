import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
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
  num_train = X.shape[1]
  num_class = W.shape[0]
  for i in range(num_train):
      scores = W.dot(X[:, i])
      C = np.max(scores)
      scores -= C
      # scores = scores
      # Num = scores[y[i]]
      Den = np.sum(np.exp(scores))
      loss += np.log(Den) - scores[y[i]]
      for j in range(num_class):
          dW[j, :] += (np.exp(scores[j]) / Den - (j == y[i])) * X[:, i]

  dW /= num_train
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = W.dot(X)
  C = np.max(scores, axis=0)
  scores -= C
  f = scores[y, np.arange(num_train)]
  s = np.sum(np.exp(scores), axis=0)
  loss = np.sum(-f + np.log(s))
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  temp = np.zeros_like(scores)
  temp[y, np.arange(num_train)] = 1
  dW = (np.exp(scores) / s - temp).dot(X.T)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
