

def scaled_dot_product_attention_simple(q, k, v, bias, name=None):
  """scaled dot-product attention.  One head.  One spatial dimension.
  Args:
    q: a Tensor with shape [batch, length_q, depth_k]
    k: a Tensor with shape [batch, length_kv, depth_k]
    v: a Tensor with shape [batch, length_kv, depth_v]
    bias: optional Tensor broadcastable to [batch, length_q, length_kv]
    name: an optional string
  Returns:
    A Tensor.
  """
  with tf.variable_scope(
      name, default_name="scaled_dot_product_attention_simple"):
    scalar = tf.rsqrt(tf.to_float(common_layers.shape_list(q)[2]))
    logits = tf.matmul(q * scalar, k, transpose_b=True)
    if bias is not None:
      logits += bias
    weights = tf.nn.softmax(logits, name="attention_weights")
    if common_layers.should_generate_summaries():
      tf.summary.image(
          "attention", tf.expand_dims(tf.pow(weights, 0.2), 3), max_outputs=1)
    return tf.matmul(weights, v)




def ffn_self_attention_layer(x,
                             filter_depth,
                             output_depth,
                             num_parts,
                             dropout_rate,
                             share_kv=False,
                             name=None):
  """Self-attention feedforward layer.
  We use self-attention to do feedforward computations. We apply this function
  positionwise where for each position, we linearly transform the output to have
  depth filter_depth, and break up the result depth-wise into num_parts
  contiguous parts.  The parts self-attend, we concatenate the results
  depth-wise, and we linearly transform to a depth of output_depth. The
  goal is to get multiplicative interactions between components of a
  representation.
  Args:
    x: a Tensor with shape [batch, length, channels]
    filter_depth: an integer
    output_depth: an integer
    num_parts: an integer dividing filter depth
    dropout_rate: a floating point number
    share_kv: Share the key value transform
    name: an optional string
  Returns:
    A Tensor.
  """

  with tf.variable_scope(
      name, default_name="feedforward_self_attention", values=[x]):
    x_shape = common_layers.shape_list(x)
    part_depth = filter_depth // num_parts
    if not share_kv:
      combined = common_layers.dense(
          x, filter_depth * 3, use_bias=False, name="qkv_transform")
      combined = tf.expand_dims(combined, axis=2)
      q, k, v = tf.split(combined, 3, axis=3)
    else:
      q = tf.expand_dims(
          common_layers.dense(
              x, filter_depth, use_bias=False, name="q_transform"),
          axis=2)
      kv_combined = tf.expand_dims(
          common_layers.dense(
              tf.concat([x, x], axis=1),
              filter_depth,
              use_bias=False,
              name="kv_transform"),
          axis=2)
      k, v = tf.split(kv_combined, [x_shape[1], x_shape[1]], axis=1)

    batch_q = tf.reshape(q, [-1, 1, num_parts, part_depth])
    batch_k = tf.reshape(k, [-1, 1, num_parts, part_depth])
    batch_v = tf.reshape(v, [-1, 1, num_parts, part_depth])

    batch_q *= part_depth**-0.5
    # non-masked bias
    bias = None
    x = dot_product_attention(batch_q, batch_k, batch_v, bias, dropout_rate)
    x = tf.reshape(x, [x_shape[0], x_shape[1], filter_depth])
    x = common_layers.dense(
        x, output_depth, use_bias=False, name="output_transform")
    return x



def dilated_self_attention_1d(q,
                              k,
                              v,
                              query_block_size=128,
                              memory_block_size=128,
                              gap_size=2,
                              num_memory_blocks=2,
                              name=None):
  """dilated self-attention.
  Args:
    q: a Tensor with shape [batch, heads, length, depth_k]
    k: a Tensor with shape [batch, heads, length, depth_k]
    v: a Tensor with shape [batch, heads, length, depth_v]
    query_block_size: an integer indicating size of query block
    memory_block_size: an integer indicating the size of a memory block.
    gap_size: an integer indicating the gap size
    num_memory_blocks: how many memory blocks to look at to the left and right.
      Each will be separated by gap_size.
    name: an optional string
  Returns:
    a Tensor of shape [batch, heads, length, depth_v]
  """
  with tf.variable_scope(
      name, default_name="dilated_self_attention_1d", values=[q, k, v]):
    v_list_shape = v.get_shape().as_list()
    v_shape = common_layers.shape_list(v)
    depth_v = v_shape[3]
    batch_size = v_shape[0]
    num_heads = v_shape[1]
    original_length = common_layers.shape_list(q)[2]

    # making sure q is a multiple of query block size
    def pad_to_multiple(x, pad_length):
      x_length = common_layers.shape_list(x)[2]
      return tf.pad(x, [[0, 0], [0, 0], [0, -x_length % pad_length], [0, 0]])

    def pad_l_and_r(x, pad_length):
      return tf.pad(x, [[0, 0], [0, 0], [pad_length, pad_length], [0, 0]])

    q = pad_to_multiple(q, query_block_size)
    v = pad_to_multiple(v, query_block_size)
    k = pad_to_multiple(k, query_block_size)

    q.set_shape(v_list_shape)
    v.set_shape(v_list_shape)
    k.set_shape(v_list_shape)
    # Setting up q blocks
    new_q_shape = common_layers.shape_list(q)
    # Setting up q blocks
    q = reshape_by_blocks(q, new_q_shape, query_block_size)
    self_k_part = reshape_by_blocks(k, new_q_shape, query_block_size)
    self_v_part = reshape_by_blocks(v, new_q_shape, query_block_size)

    # Setting up k and v windows
    k_v_padding = (gap_size + memory_block_size) * num_memory_blocks
    k = pad_l_and_r(k, k_v_padding)
    v = pad_l_and_r(v, k_v_padding)
    # getting gather indices
    index_length = (new_q_shape[2] - query_block_size + memory_block_size)
    indices = tf.range(0, index_length, delta=1, name="index_range")
    # making indices [1, length, 1] to appy convs
    indices = tf.reshape(indices, [1, -1, 1])
    kernel = tf.expand_dims(tf.eye(memory_block_size), axis=1)
    gather_indices = tf.nn.conv1d(
        tf.cast(indices, tf.float32),
        kernel,
        query_block_size,
        padding="VALID",
        name="gather_conv")

    gather_indices = tf.squeeze(tf.cast(gather_indices, tf.int32), axis=0)

    # get left and right memory blocks for each query
    # [length, batch, heads, dim]
    k_t = tf.transpose(k, [2, 0, 1, 3])
    v_t = tf.transpose(v, [2, 0, 1, 3])
    left_k = gather_dilated_memory_blocks(
        k_t[:-k_v_padding, :, :, :], num_memory_blocks, gap_size,
        query_block_size, memory_block_size, gather_indices)
    left_v = gather_dilated_memory_blocks(
        v_t[:-k_v_padding, :, :, :], num_memory_blocks, gap_size,
        query_block_size, memory_block_size, gather_indices)

    right_k = gather_dilated_memory_blocks(
        k_t[k_v_padding:, :, :, :],
        num_memory_blocks,
        gap_size,
        query_block_size,
        memory_block_size,
        gather_indices,
        direction="right")
    right_v = gather_dilated_memory_blocks(
        v_t[k_v_padding:, :, :, :],
        num_memory_blocks,
        gap_size,
        query_block_size,
        memory_block_size,
        gather_indices,
        direction="right")

    k_windows = tf.concat([left_k, self_k_part, right_k], axis=3)
    v_windows = tf.concat([left_v, self_v_part, right_v], axis=3)
    attention_bias = tf.expand_dims(
        embedding_to_padding(k_windows) * -1e9, axis=-2)

    output = dot_product_attention(
        q,
        k_windows,
        v_windows,
        attention_bias,
        dropout_rate=0.,
        name="dilated_1d",
        make_image_summary=False)
    output = tf.reshape(output, [batch_size, num_heads, -1, depth_v])
    # Remove the padding if introduced
    output = tf.slice(output, [0, 0, 0, 0], [-1, -1, original_length, -1])
    output.set_shape(v_list_shape)
    return output
    
    
    

def masked_dilated_self_attention_1d(q,
                                     k,
                                     v,
                                     query_block_size=64,
                                     memory_block_size=64,
                                     gap_size=2,
                                     num_memory_blocks=2,
                                     name=None):
  """dilated self-attention. TODO(avaswani): Try it and write a paper on it.
  Args:
    q: a Tensor with shape [batch, heads, length, depth_k]
    k: a Tensor with shape [batch, heads, length, depth_k]
    v: a Tensor with shape [batch, heads, length, depth_v]
    query_block_size: an integer
    memory_block_size: an integer indicating how much to look left.
    gap_size: an integer indicating the gap size
    num_memory_blocks: how many memory blocks to look at to the left. Each will
      be separated by gap_size.
    name: an optional string
  Returns:
    a Tensor of shape [batch, heads, length, depth_v]
  """
  with tf.variable_scope(
      name, default_name="masked_dilated_self_attention_1d", values=[q, k, v]):
    v_list_shape = v.get_shape().as_list()
    v_shape = common_layers.shape_list(v)
    depth_v = v_shape[3]
    batch_size = v_shape[0]
    num_heads = v_shape[1]
    original_length = common_layers.shape_list(q)[2]

    # making sure q is a multiple of query block size
    def pad_to_multiple(x, pad_length):
      x_length = common_layers.shape_list(x)[2]
      return tf.pad(x, [[0, 0], [0, 0], [0, -x_length % pad_length], [0, 0]])

    def pad_l(x, left_pad_length):
      return tf.pad(x, [[0, 0], [0, 0], [left_pad_length, 0], [0, 0]])

    q = pad_to_multiple(q, query_block_size)
    v = pad_to_multiple(v, query_block_size)
    k = pad_to_multiple(k, query_block_size)
    q.set_shape(v_list_shape)
    v.set_shape(v_list_shape)
    k.set_shape(v_list_shape)
    # Setting up q blocks
    new_q_shape = common_layers.shape_list(q)

    # Setting up q blocks
    q = reshape_by_blocks(q, new_q_shape, query_block_size)
    self_k_part = reshape_by_blocks(k, new_q_shape, query_block_size)
    self_v_part = reshape_by_blocks(v, new_q_shape, query_block_size)
    # Setting up k and v windows
    k_v_padding = (gap_size + memory_block_size) * num_memory_blocks
    k = pad_l(k, k_v_padding)
    v = pad_l(v, k_v_padding)
    # getting gather indices
    index_length = (new_q_shape[2] - query_block_size + memory_block_size)

    indices = tf.range(0, index_length, delta=1, name="index_range")
    # making indices [1, length, 1] to appy convs
    indices = tf.reshape(indices, [1, -1, 1])
    kernel = tf.expand_dims(tf.eye(memory_block_size), axis=1)
    gather_indices = tf.nn.conv1d(
        tf.cast(indices, tf.float32),
        kernel,
        query_block_size,
        padding="VALID",
        name="gather_conv")
    gather_indices = tf.squeeze(tf.cast(gather_indices, tf.int32), axis=0)

    # get left and right memory blocks for each query
    # [length, batch, heads, dim]
    k_t = tf.transpose(k, [2, 0, 1, 3])
    v_t = tf.transpose(v, [2, 0, 1, 3])

    k_unmasked_windows = gather_dilated_memory_blocks(
        k_t, num_memory_blocks, gap_size, query_block_size, memory_block_size,
        gather_indices)
    v_unmasked_windows = gather_dilated_memory_blocks(
        v_t, num_memory_blocks, gap_size, query_block_size, memory_block_size,
        gather_indices)

    # combine memory windows
    block_q_shape = common_layers.shape_list(q)
    masked_attention_bias = tf.tile(
        tf.expand_dims(attention_bias_lower_triangle(query_block_size), axis=0),
        [block_q_shape[0], block_q_shape[1], block_q_shape[2], 1, 1])
    padding_attention_bias = tf.expand_dims(
        embedding_to_padding(k_unmasked_windows) * -1e9, axis=-2)
    padding_attention_bias = tf.tile(padding_attention_bias,
                                     [1, 1, 1, query_block_size, 1])
    attention_bias = tf.concat(
        [masked_attention_bias, padding_attention_bias], axis=-1)
    # combine memory windows
    k_windows = tf.concat([self_k_part, k_unmasked_windows], 3)
    v_windows = tf.concat([self_v_part, v_unmasked_windows], 3)
    output = dot_product_attention(
        q,
        k_windows,
        v_windows,
        attention_bias,
        dropout_rate=0.,
        name="dilated_1d",
        make_image_summary=False)
    output = tf.reshape(output, [batch_size, num_heads, -1, depth_v])
    # Remove the padding if introduced
    output = tf.slice(output, [0, 0, 0, 0], [-1, -1, original_length, -1])
    output.set_shape(v_list_shape)
    return output
