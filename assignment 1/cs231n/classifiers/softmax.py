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
    
    # set S, N, D and C
    S = X @ W
    N,D = np.shape(X)
    C = np.shape(W)[1]
    # begins
    for i in range(N):
        yi = y[i]
        deno = 0
        for j in range(C):
            deno += np.exp(S[i][j])  # deno = \sum_j e^{sj}
        for j in range(C):
            dW[:,j] += np.exp(S[i][j]) / deno * X[i].T
        dW[:,yi] -= X[i].T
        loss += -S[i][yi] + np.log(deno)
    
    # Add regularization
    loss /= N
    dW /= N
    loss += reg * np.sqrt(np.sum(W * W))
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
    
    # Initialize some useful parameters
    S = X @ W
    N,D = np.shape(X)
    enm = np.arange(0,N) # enm = [0,1,...,N]
    correct_S = S[enm,y].reshape(N,1) # correct_S[i] = S[i][y_i]
    
    denos = np.sum(np.exp(S), axis=1).reshape(N,1) # compute the denominators
    # compute 'naive' loss
    loss -= np.sum(correct_S)
    loss += np.sum(np.log(denos))
    # compute 'naive' gradient dW
    dW[:,y] -= X.T
    dW += X.T @ (S / denos) #####
    
    # Add regularization
    loss /= N
    dW /= N
    loss += reg * np.sqrt(np.sum(W * W))
    dW += 2 * reg * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
