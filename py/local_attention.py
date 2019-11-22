


def local_attention_1d(q, k, v, block_length=128, filter_width=100, name=None):
  """strided block local self-attention.
  Args:
    q: a Tensor with shape [batch, heads, length, depth_k]
    k: a Tensor with shape [batch, heads, length, depth_k]
    v: a Tensor with shape [batch, heads, length, depth_v]
    block_length: an integer
    filter_width: an integer indicating how much to look left.
    name: an optional string
  Returns:
    a Tensor of shape [batch, heads, length, depth_v]
  """
  with tf.variable_scope(
      name, default_name="local_self_attention_1d", values=[q, k, v]):
    v_shape = v.get_shape()
    depth_v = common_layers.shape_list(v)[3]
    batch_size = common_layers.shape_list(q)[0]
    num_heads = common_layers.shape_list(q)[1]
    original_length = common_layers.shape_list(q)[2]

    # making sure q is a multiple of d
    def pad_to_multiple(x, pad_length):
      x_length = common_layers.shape_list(x)[2]
      return tf.pad(x, [[0, 0], [0, 0], [0, -x_length % pad_length], [0, 0]])

    def pad_l_and_r(x, pad_length):
      return tf.pad(x, [[0, 0], [0, 0], [pad_length, pad_length], [0, 0]])

    q = pad_to_multiple(q, block_length)
    k = pad_to_multiple(k, block_length)
    v = pad_to_multiple(v, block_length)

    # Setting up q blocks
    new_q_shape = common_layers.shape_list(q)
    # Setting up q blocks
    q = tf.reshape(q, [
        new_q_shape[0], new_q_shape[1], new_q_shape[2] // block_length,
        block_length, new_q_shape[3]
    ])

    # Setting up k and v values
    k = pad_l_and_r(k, filter_width)
    v = pad_l_and_r(v, filter_width)

    length = common_layers.shape_list(k)[2]
    full_filter_width = block_length + 2 * filter_width
    # getting gather indices
    indices = tf.range(0, length, delta=1, name="index_range")
    # making indices [1, length, 1] to appy convs
    indices = tf.reshape(indices, [1, -1, 1])
    kernel = tf.expand_dims(tf.eye(full_filter_width), axis=1)
    gather_indices = tf.nn.conv1d(
        tf.cast(indices, tf.float32),
        kernel,
        block_length,
        padding="VALID",
        name="gather_conv")

    gather_indices = tf.squeeze(tf.cast(gather_indices, tf.int32), axis=0)

    # [length, batch, heads, dim]
    k_t = tf.transpose(k, [2, 0, 1, 3])
    k_new = tf.gather(k_t, gather_indices)

    # [batch, heads, blocks, block_length, dim]
    k_new = tf.transpose(k_new, [2, 3, 0, 1, 4])

    attention_bias = tf.expand_dims(embedding_to_padding(k_new) * -1e9, axis=-2)

    v_t = tf.transpose(v, [2, 0, 1, 3])
    v_new = tf.gather(v_t, gather_indices)
    v_new = tf.transpose(v_new, [2, 3, 0, 1, 4])

    output = dot_product_attention(
        q,
        k_new,
        v_new,
        attention_bias,
        dropout_rate=0.,
        name="local_1d",
        make_image_summary=False)
    output = tf.reshape(output, [batch_size, num_heads, -1, depth_v])
    # Remove the padding if introduced
    output = tf.slice(output, [0, 0, 0, 0], [-1, -1, original_length, -1])
    output.set_shape(v_shape)
    return output




def local_attention_2d(q,
                       k,
                       v,
                       query_shape=(8, 16),
                       memory_flange=(8, 16),
                       name=None):
  """strided block local self-attention.
  Args:
    q: a Tensor with shape [batch, heads, h, w, depth_k]
    k: a Tensor with shape [batch, heads, h, w, depth_k]
    v: a Tensor with shape [batch, heads, h, w, depth_v]
    query_shape: an tuple indicating the height and width of each query block.
    memory_flange: an integer indicating how much to look in height and width
      from each query block.
    name: an optional string
  Returns:
    a Tensor of shape [batch, heads, h, w, depth_v]
  """
  with tf.variable_scope(
      name, default_name="local_self_attention_2d", values=[q, k, v]):
    q_shape = q.get_shape().as_list()
    v_shape = common_layers.shape_list(v)

    q = pad_to_multiple_2d(q, query_shape)
    k = pad_to_multiple_2d(k, query_shape)
    v = pad_to_multiple_2d(v, query_shape)
    padded_q_shape = common_layers.shape_list(q)
    # Setting up k and v values
    paddings = [[0, 0], [0, 0], [memory_flange[0], memory_flange[1]],
                [memory_flange[0], memory_flange[1]], [0, 0]]
    k = tf.pad(k, paddings)
    v = tf.pad(v, paddings)

    # Setting up q blocks
    q_indices = gather_indices_2d(q, query_shape, query_shape)
    q_new = gather_blocks_2d(q, q_indices)

    # Setting up k and v blocks
    memory_shape = (query_shape[0] + 2 * memory_flange[0],
                    query_shape[1] + 2 * memory_flange[1])
    k_and_v_indices = gather_indices_2d(k, memory_shape, query_shape)
    k_new = gather_blocks_2d(k, k_and_v_indices)
    v_new = gather_blocks_2d(v, k_and_v_indices)

    attention_bias = tf.expand_dims(
        tf.to_float(embedding_to_padding(k_new)) * -1e9, axis=-2)

    output = dot_product_attention(
        q_new,
        k_new,
        v_new,
        attention_bias,
        dropout_rate=0.,
        name="local_2d",
        make_image_summary=False)
    # putting the representations back in the right place
    output = scatter_blocks_2d(output, q_indices, padded_q_shape)
    # Remove the padding if introduced
    output = tf.slice(output, [0, 0, 0, 0, 0],
                      [-1, -1, v_shape[2], v_shape[3], -1])
    output.set_shape(q_shape)
    return output




def masked_local_attention_2d(q,
                              k,
                              v,
                              query_shape=(8, 16),
                              memory_flange=(8, 16),
                              name=None):
  """strided block local self-attention.
    Each position in a query block can attend to all the generated queries in
    the query block, which are generated in raster scan, and positions that are
    generated to the left and top. The shapes are specified by query shape and
    memory flange. Note that if you're using this function, you do not need to
    right shift. Right shifting happens inside this function separately for each
    block.
  Args:
    q: a Tensor with shape [batch, heads, h, w, depth_k]
    k: a Tensor with shape [batch, heads, h, w, depth_k]
    v: a Tensor with shape [batch, heads, h, w, depth_v]
    query_shape: an tuple indicating the height and width of each query block.
      query_shape = block_shape
    memory_flange: an integer indicating how much to look in height and width
      from each query block.
      memory shape = query_shape + (block_flange[0], 2*block_flange[1])
    name: an optional string
  Returns:
    a Tensor of shape [batch, heads, h, w, depth_v]
  """
  with tf.variable_scope(
      name, default_name="local_masked_self_attention_2d", values=[q, k, v]):
    q_shape = q.get_shape().as_list()
    v_shape = common_layers.shape_list(v)

    q = pad_to_multiple_2d(q, query_shape)
    padded_q_shape = common_layers.shape_list(q)
    # Setting up q blocks
    q_indices = gather_indices_2d(q, query_shape, query_shape)
    q_new = gather_blocks_2d(q, q_indices)
    # Setting up k and v blocks
    k_flange, k_center = get_memory_region(k, query_shape, memory_flange,
                                           q_indices)
    v_flange, v_center = get_memory_region(v, query_shape, memory_flange,
                                           q_indices)
    if k_flange is not None:
      k_new = tf.concat([k_flange, k_center], axis=3)
      v_new = tf.concat([v_flange, v_center], axis=3)
    else:
      k_new = k_center
      v_new = v_center
    # Getting the masks ready
    query_elements = np.prod(query_shape)
    padding_mask = None
    if k_flange is not None:
      padding_mask = tf.expand_dims(
          embedding_to_padding(k_flange) * -1e9, axis=-2)
      padding_mask = tf.tile(padding_mask, [1, 1, 1, query_elements, 1])

    center_attention_bias = attention_bias_lower_triangle(
        np.prod(query_elements))
    center_attention_bias = tf.reshape(
        center_attention_bias, [1, 1, 1, query_elements, query_elements])
    v_center_shape = common_layers.shape_list(v_center)
    center_attention_bias = tf.tile(
        center_attention_bias,
        [v_center_shape[0], v_center_shape[1], v_center_shape[2], 1, 1])
    if padding_mask is not None:
      # Combining the mask for padding and visible region
      attention_bias = tf.concat([padding_mask, center_attention_bias], axis=4)
    else:
      attention_bias = center_attention_bias

    output = dot_product_attention(
        q_new,
        k_new,
        v_new,
        attention_bias,
        dropout_rate=0.,
        name="masked_local_2d",
        make_image_summary=False)
    # putting the representations back in the right place
    output = scatter_blocks_2d(output, q_indices, padded_q_shape)
    # Remove the padding if introduced
    output = tf.slice(output, [0, 0, 0, 0, 0],
                      [-1, -1, v_shape[2], v_shape[3], -1])
    output.set_shape(q_shape)
    return output
