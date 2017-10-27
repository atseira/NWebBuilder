import tensorflow as tf

# TODO: change these two to Builder's attributes
BIAS_IV = 0.1 # IV: initial value
WEIGHT_SD = 0.1 # Weight random initialization std. deviation


class Builder:
    MATMUL = 'matmul'
    RELU = 'relu'

    def __init__(self, x):
        ''' Class initialization

        Keyword arguments:
        x -- input tensor (the data)

        Attributes:
        layers -- a list of tensors of a neural network
        '''
        self.x = x
        self.layers = [self.x]

    def add_layer(self, layer_set):
        n_layers = len(self.layers)

        # What this layer will do to the input & its size (# of nodes)
        tensor_in = self.layers[n_layers-1] # the input tensor
        op = layer_set[0] # the operation
        n_out = layer_set[1] # the size

        # Create the tensor according to the operation given
        if op == self.MATMUL: # matmul: matrix multiplication
            last_layer = self.layers[n_layers-1]
            tensor = matmul(last_layer, n_out)
        elif op == self.RELU: # ReLU: rectified linear units
            tensor = tf.nn.relu(tensor_in)

        # Name the tensor for easy calling later
        # P.S. ls stands for layer set
        tensor_name = 'ls' + str(n_layers) + '-' + str(op)
        tensor = tf.identity(tensor, name=tensor_name)

        # Add the new tensor to self.layers list
        self.layers.append(tensor)

def matmul(x, n_out):
    ''' Generate matrix multiplication tensor

    Keyword arguments:
    x -- input tensor
    n_out -- number of neurons/number of elements after multiplication
    '''
    n_in = x.get_shape().as_list()[-1]
    W = weight_var([n_in, n_out])
    b = bias_var([n_out])
    return tf.matmul(x,W) + b

def weight_var(shape):
    ''' Initiate weight variables to train

    Keyword arguments:
    shape -- a tuple of integers to generate matrix shape (e.g. row * columns)
    '''
    initial = tf.truncated_normal(shape, stddev=WEIGHT_SD)
    return tf.Variable(initial)

def bias_var(shape):
    ''' Initiate bias variables to train

    Keyword arguments:
    shape -- a tuple of integers to generate matrix shape
    '''
    initial = tf.constant(BIAS_IV, shape=shape)
    return tf.Variable(initial)
