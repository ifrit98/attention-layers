
# modification from https://github.com/tensorflow/tensorflow/pull/21276
# without special initialization for g
class WeightNorm(tf.keras.layers.Wrapper):
  """Decouple weight magnitude and direction.
  This wrapper reparameterizes a layer by decoupling the weight's
  magnitude and direction. This speeds up convergence by improving the
  conditioning of the optimization problem.
  Weight Normalization: A Simple Reparameterization to Accelerate
  Training of Deep Neural Networks: https://arxiv.org/abs/1602.07868
  Tim Salimans, Diederik P. Kingma (2016)
  WeightNorm wrapper works for keras and tf layers.
  ```python
    net = WeightNorm(tf.keras.layers.Conv2D(2, 2, activation='relu'),
           input_shape=(32, 32, 3), data_init=True)(x)
    net = WeightNorm(tf.keras.layers.Conv2D(16, 5, activation='relu'),
                     data_init=True)
    net = WeightNorm(tf.keras.layers.Dense(120, activation='relu'),
                     data_init=True)(net)
    net = WeightNorm(tf.keras.layers.Dense(n_classes),
                     data_init=True)(net)
  ```
  Arguments:
    layer: a layer instance.
    data_init: If `True` use data dependent variable initialization
  Raises:
    ValueError: If not initialized with a `Layer` instance.
    ValueError: If `Layer` does not contain a `kernel` of weights
    NotImplementedError: If `data_init` is True and running graph execution
  """

  def __init__(self, layer, data_init=False, **kwargs):
    if not isinstance(layer, tf.keras.layers.Layer):
      raise ValueError(
          "Please initialize `WeightNorm` layer with a "
          "`Layer` instance. You passed: {input}".format(input=layer))

    super(WeightNorm, self).__init__(layer, **kwargs)
    self._track_trackable(layer, name="layer")

  def _compute_weights(self):
    """Generate weights with normalization."""
    with tf.variable_scope("compute_weights"):
      self.layer.kernel = tf.nn.l2_normalize(
          self.layer.v, axis=self.norm_axes) * self.layer.g

  def _init_norm(self, weights):
    """Set the norm of the weight vector."""
    with tf.variable_scope("init_norm"):
      flat = tf.reshape(weights, [-1, self.layer_depth])
      return tf.reshape(tf.norm(flat, axis=0), (self.layer_depth,))

  def _data_dep_init(self, inputs):
    """Data dependent initialization for eager execution."""

    with tf.variable_scope("data_dep_init"):
      # Generate data dependent init values
      activation = self.layer.activation
      self.layer.activation = None
      x_init = self.layer.call(inputs)
      m_init, v_init = tf.moments(x_init, self.norm_axes)
      scale_init = 1. / tf.sqrt(v_init + 1e-10)

    # Assign data dependent init values
    self.layer.g = self.layer.g * scale_init
    self.layer.bias = (-m_init * scale_init)
    self.layer.activation = activation
    self.initialized = True

  def build(self, input_shape=None):
    """Build `Layer`."""
    input_shape = tf.TensorShape(input_shape).as_list()
    self.input_spec = layers().InputSpec(shape=input_shape)

    if not self.layer.built:
      self.layer.build(input_shape)
      self.layer.built = False

      if not hasattr(self.layer, "kernel"):
        raise ValueError("`WeightNorm` must wrap a layer that"
                         " contains a `kernel` for weights")

      # The kernel's filter or unit dimension is -1
      self.layer_depth = int(self.layer.kernel.shape[-1])
      self.norm_axes = list(range(self.layer.kernel.shape.ndims - 1))

      self.layer.v = self.layer.kernel
      self.layer.g = self.layer.add_variable(
          name="g",
          shape=(self.layer_depth,),
          initializer=tf.ones_initializer,
          dtype=self.layer.kernel.dtype,
          trainable=True)

      # with ops.control_dependencies([self.layer.g.assign(
      #     self._init_norm(self.layer.v))]):
      #   self._compute_weights()
      self._compute_weights()

      self.layer.built = True

    super(WeightNorm, self).build()
    self.built = True

  def call(self, inputs):
    """Call `Layer`."""
    # if context.executing_eagerly():
    #   if not self.initialized:
    #     self._data_dep_init(inputs)
    self._compute_weights()  # Recompute weights for each forward pass

    output = self.layer.call(inputs)
    return output

  def compute_output_shape(self, input_shape):
    return tf.TensorShape(
        self.layer.compute_output_shape(input_shape).as_list())
