# Keras IndRNN Implementation #

This repository contains a Keras implementation of Independently Recurrent Neural Networks (IndRNN, Li et al. [[1]](https://arxiv.org/abs/1803.04831)).

The file [`indrnn.py`](https://github.com/flandolfi/indrnn/blob/master/indrnn.py) contains the classes:

 - `IndRNNCell`, which construct the basic IndRNN model by slight modifying Keras' `SimpleRNNCell`. Its recurrent kernel is a single-row matrix, which is multiplied element-wise at each timesteps with the model's previous output;
 - `IndRNN`, which is a `RNN` with a single `IndRNNCell`;
 - `CuDNNRNN`, which uses the CuDNN-RNN as a black-box to compute the IndRNN step by trasforming the recurrent kernel in a diagonal matrix. **Warning:** This is just a stub and it is recommended to use the `IndRNN` class instead.

## Basic Usage ##

The `IndRNN` by default uses a ReLU activation, its recurrent kernel is constrained and initialized with random values in the range (-1, 1). It can be used "as-is", as following
```python
from keras.models import Sequential
from keras.layers import Dense
from indrnn import IndRNN

model = Sequential()
model.add(IndRNN(128, input_shape=(None, 2)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.summary()
```
```
Layer (type)                 Output Shape              Param #   
=================================================================
ind_rnn_1 (IndRNN)           (None, 128)               512       
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 129       
=================================================================
Total params: 641
Trainable params: 641
Non-trainable params: 0
```

If you want to control better the recurrent kernel values, as in [1], you can use Keras' initializers and contraints, as following
```python
from keras.initializers import RandomUniform
from keras.constraints import MaxNorm

T = 5000
limit = 2**(1/T)

model = Sequential()
model.add(IndRNN(128, 
                 recurrent_initializer=RandomUniform(-limit, limit),
                 recurrent_constraint=MaxNorm(limit),
                 input_shape=(None, 2)))
model.add(Dense(1))
```
Keep in mind that Keras' constraints act by default on kernel's first axis. Since `IndRNN` is a 2-dimensional single-row matrix, applying `MaxNorm` is the same as constraining the kernel's absolute values.

## Test ##

You can see the `IndRNN` in action [in this Jupyter notebook](https://github.com/flandolfi/indrnn/blob/master/test.ipynb).

## References ##

[1] Li, S., Li, W., Cook, C., Zhu, C. and Gao, Y., 2018. Independently recurrent neural network (indrnn): Building a longer and deeper rnn. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 5457-5466).