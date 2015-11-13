import numpy as np

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)

  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  # out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  out = x.reshape(x.shape[0], w.shape[0]).dot(w) + b
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  dx = dout.dot(w.T)

  dw = np.dot(x.reshape(dx.shape).T, dout)
  db = dout.sum(axis=0)

  dx = dx.reshape(x.shape)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = np.maximum(x, 0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  dx = dout
  dx[x <= 0] = 0
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  N = x.shape[0]
  H, W = x.shape[2], x.shape[3]
  F = w.shape[0]
  HH = w.shape[2]
  WW = w.shape[3]
  pad, stride = conv_param['pad'], conv_param['stride']
  Hout = 1 + (H + 2 * pad - HH) / stride
  Wout = 1 + (W + 2 * pad - WW) / stride
  x_pad = np.lib.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), 'constant');
  # print x_pad[0,:,:,:]
  # w_iters = (x_pad.shape[3] - WW) / stride
  # print w_iters
  # h_iters = (x_pad.shape[2] - HH) / stride
  out = np.zeros([N, F, Hout, Wout])
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  for i in range(N):
      for j in range(F):
          for hh in range(Hout):
              for ww in range(Wout):
                  out[i, j, hh, ww] = np.sum(x_pad[i, :, hh*stride:hh*stride+HH
                  , ww*stride:ww*stride+WW]*w[j,:,:,:]) + b[j]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  x, w, b, conv_param = cache
  N = x.shape[0]
  F = w.shape[0]
  HH = w.shape[2]
  WW = w.shape[3]
  Hout, Wout = dout.shape[2], dout.shape[3]
  pad, stride = conv_param['pad'], conv_param['stride']
  x_pad = np.lib.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), 'constant');
  dx, dw, db = np.zeros(x.shape), np.zeros(w.shape), np.zeros(b.shape)
  dx_pad = np.lib.pad(dx, ((0,0),(0,0),(pad,pad),(pad,pad)), 'constant');
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  for i in range(N):
      for j in range(F):
          for ww in range(Wout):
              for hh in range(Hout):
                  dx_pad[i, :, hh*stride:hh*stride+HH, ww*stride:ww*stride+WW] \
                  += dout[i, j, hh, ww] * w[j, :, :, :]
                  dw[j, :, :, :] +=  \
                  dout[i, j, hh, ww] * x_pad[i, :, hh*stride:hh*stride+HH, ww*stride:ww*stride+WW]
                  db[j] += dout[i, j, hh, ww]
  dx = dx_pad[:, :, pad:-pad, pad:-pad]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  HH, WW, stride = pool_param['pool_height'], pool_param['pool_width'],\
  pool_param['stride']
  N, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
  Hout = 1 + (H - HH) / stride
  Wout = 1 + (W - WW) / stride
  out = np.zeros([N, C, Hout, Wout])
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  for i in range(N):
      for j in range(C):
          for hh in range(Hout):
              for ww in range(Wout):
                  out[i, j, hh, ww] = np.max(x[i, j, hh*stride:hh*stride+HH, \
                  ww*stride:ww*stride+WW])
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  x, pool_param = cache
  dx = np.zeros(x.shape)
  HH, WW, stride = pool_param['pool_height'], pool_param['pool_width'],\
  pool_param['stride']
  N, C, Hout, Wout = dout.shape[0], dout.shape[1], dout.shape[2], dout.shape[3]
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  for i in range(N):
      for j in range(C):
          for hh in range(Hout):
              for ww in range(Wout):
                  position = np.argmax(x[i, j, hh*stride:hh*stride+HH, \
                  ww*stride:ww*stride+WW])
                  # print position
                  h, w = hh*stride+position / WW, \
                  ww*stride+position % WW
                  # print h, w
                  dx[i, j, h, w] += dout[i, j, hh, ww]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
