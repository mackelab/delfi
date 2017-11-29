import lasagne
import lasagne.layers as ll
import theano
import theano.tensor as tt

dtype = theano.config.floatX


class ImputeMissingLayer(lasagne.layers.Layer):
    def __init__(self, incoming, n_inputs, R=lasagne.init.Normal(0.01), **kwargs):
        """Inputs that are NaN will be replaced by learned imputation value"""
        super(ImputeMissingLayer, self).__init__(incoming, **kwargs)
        self.R = self.add_param(R, (*n_inputs,), name='imputation_values')

    def get_output_for(self, input, **kwargs):
        return tt.cast(tt.switch(tt.isnan(input), self.R, input), dtype)

    def get_output_shape_for(self, input_shape):
        return input_shape


class ReplaceMissingLayer(lasagne.layers.Layer):
    def __init__(self, incoming, n_inputs, **kwargs):
        """Inputs that are NaN will be replaced by zero through this layer"""
        super(ReplaceMissingLayer, self).__init__(incoming, **kwargs)
        self.Z = tt.zeros((*n_inputs,), dtype)

    def get_output_for(self, input, **kwargs):
        return tt.cast(tt.switch(tt.isnan(input), self.Z, input), dtype)

    def get_output_shape_for(self, input_shape):
        return input_shape
