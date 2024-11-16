"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
import math
from .nn_basic import Parameter, Module, Tanh, ReLU


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return (1 + ops.exp(-x)) ** (-1)
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        interval = math.sqrt(1 / hidden_size)
        
        self.W_ih = Parameter(init.rand(*(input_size, hidden_size), low=-interval, high=interval, device=device, requires_grad=True))
        self.W_hh = Parameter(init.rand(*(hidden_size, hidden_size), low=-interval, high=interval, device=device, requires_grad=True))
        
        if bias:
            self.bias_ih = Parameter(init.rand(hidden_size, low=-interval, high=interval, device=device, requires_grad=True))
            self.bias_hh = Parameter(init.rand(hidden_size, low=-interval, high=interval, device=device, requires_grad=True))
        else:
            self.bias_ih = None
            self.bias_hh = None

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.nonlinearity = Tanh() if nonlinearity == 'tanh' else ReLU()
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]

        if h is None:
            h = init.zeros(*(bs, self.hidden_size), device=self.device, requires_grad=True)
        
        out = X @ self.W_ih + h @ self.W_hh
        if self.bias_ih:
            out = out + self.bias_ih.reshape((1, self.hidden_size)).broadcast_to((bs, self.hidden_size)) + self.bias_hh.reshape((1, self.hidden_size)).broadcast_to((bs, self.hidden_size))
        
        out = self.nonlinearity(out)
        
        return out
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        self.rnn_cells = []
        for k in range(num_layers):
            if k == 0:
                self.rnn_cells.append(RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype))
            else:
                self.rnn_cells.append(RNNCell(hidden_size, hidden_size, bias, nonlinearity, device, dtype))
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h0 is None:
            h0 = init.zeros(*(self.num_layers, X.shape[1], self.hidden_size), device=self.device, requires_grad=True)

        x_t = list(ops.split(X, axis=0))
        h_n = list(ops.split(h0, axis=0))

        for i in range(len(x_t)):
            x = x_t[i]
            for j in range(len(h_n)):
                h = h_n[j]
                out = self.rnn_cells[j](x, h)
                h_n[j] = out
                x = out
            x_t[i] = x
        
        return ops.stack(tuple(x_t), axis=0), ops.stack(tuple(h_n), axis=0)
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        interval = 1 / math.sqrt(hidden_size)

        self.W_ih = Parameter(init.rand(*(input_size, 4 * hidden_size), low=-interval, high=interval, device=device, requires_grad=True))
        self.W_hh = Parameter(init.rand(*(hidden_size, 4 * hidden_size), low=-interval, high=interval, device=device, requires_grad=True))

        if bias:
            self.bias_ih = Parameter(init.rand(4 * hidden_size, low=-interval, high=interval, device=device, requires_grad=True))
            self.bias_hh = Parameter(init.rand(4 * hidden_size, low=-interval, high=interval, device=device, requires_grad=True))
        else:
            self.bias_ih = None
            self.bias_hh = None
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

        self.sigmoid = Sigmoid()
        self.tanh = Tanh()
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]

        if h:
            h0, c0 = h
        else:
            h0 = init.zeros(*(bs, self.hidden_size), device=self.device, requires_grad=True)
            c0 = init.zeros(*(bs, self.hidden_size), device=self.device, requires_grad=True)
        
        gates_all = X @ self.W_ih + h0 @ self.W_hh
        if self.bias_ih:
            gates_all = gates_all + self.bias_ih.reshape((1, 4 * self.hidden_size)).broadcast_to((bs, 4 * self.hidden_size)) + self.bias_hh.reshape((1, 4 * self.hidden_size)).broadcast_to((bs, 4 * self.hidden_size))
        
        # out (bs, 4 * hidden_size)
        gates_all_split = tuple(ops.split(gates_all, axis=1))

        gates = []
        for i in range(4):
            gates.append(ops.stack(gates_all_split[i * self.hidden_size : (i + 1) * self.hidden_size], axis=1))
        
        i, f, g, o = gates
        i, f, g, o = self.sigmoid(i), self.sigmoid(f), self.tanh(g), self.sigmoid(o)

        c_ = f * c0 + i * g
        h_ = o * self.tanh(c_)

        return h_, c_
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.lstm_cells = []
        for k in range(num_layers):
            if k == 0:
                self.lstm_cells.append(LSTMCell(input_size, hidden_size, bias, device, dtype))
            else:
                self.lstm_cells.append(LSTMCell(hidden_size, hidden_size, bias, device, dtype))
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            c_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h:
            h0, c0 = h
        else:
            h0 = init.zeros(*(self.num_layers, X.shape[1], self.hidden_size), device=self.device, requires_grad=True)
            c0 = init.zeros(*(self.num_layers, X.shape[1], self.hidden_size), device=self.device, requires_grad=True)

        x_t = list(ops.split(X, axis=0))
        h_n = list(ops.split(h0, axis=0))
        c_n = list(ops.split(c0, axis=0))

        for i in range(len(x_t)):
            x = x_t[i]
            for j in range(len(h_n)):
                h, c = h_n[j], c_n[j]
                o1, o2 = self.lstm_cells[j](x, (h, c))
                x = o1
                h_n[j] = o1
                c_n[j] = o2
            x_t[i] = x
        
        x_t = ops.stack(tuple(x_t), axis=0)
        h_n = ops.stack(tuple(h_n), axis=0)
        c_n = ops.stack(tuple(c_n), axis=0)

        return (x_t, (h_n, c_n))
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION