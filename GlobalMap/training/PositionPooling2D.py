from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import tensorflow as tf

class PositionPooling2D(Layer):
  def __init__(
      self,
      pool_size=(2, 2),
      strides=(2, 2),
      padding='same',
      **kwargs
  ):
    super(PositionPooling2D, self).__init__(**kwargs)
    self.padding = padding
    self.pool_size = pool_size
    self.strides = strides

  def call(self, inputs, **kwargs):
    if not K.backend() == 'tensorflow': 
      raise NotImplementedError(
        '{} backend is not supported for layer {}'.format(K.backend(), type(self).__name__)
      )

    values, indices = tf.nn.max_pool_with_argmax(
      inputs,
      ksize=[1, self.pool_size[0], self.pool_size[1], 1],
      strides=[1, self.strides[0], self.strides[1], 1],
      padding=self.padding.upper(),
      output_dtype=tf.int64
    )
    
    shape = K.cast(K.shape(values), tf.int64)
    X = tf.math.floormod(indices, shape[2]) / (shape[2] - 1) 
    Y = tf.math.floordiv(indices, shape[2]) / (shape[1] - 1)
    return K.permute_dimensions(K.concatenate([X, Y], axis=-1), (0, 3, 1, 2))

  def compute_output_shape(self, input_shape):
    output_shape = (input_shape[0], 2, input_shape[1] - self.strides[1], input_shape[2] - self.strides[0])
    return [output_shape]

  def compute_mask(self, inputs, mask=None):
    return 2 * [None]