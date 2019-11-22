

def dot_product_attention(q,
                          k,
                          v,
                          bias,
                          dropout_rate=0.0,
                          image_shapes=None,
                          name=None,
                          make_image_summary=True,
                          save_weights_to=None,
                          dropout_broadcast_dims=None):
  """dot-product attention.
  Args:
    q: a Tensor with shape [batch, heads, length_q, depth_k]
    k: a Tensor with shape [batch, heads, length_kv, depth_k]
    v: a Tensor with shape [batch, heads, length_kv, depth_v]
    bias: bias Tensor (see attention_bias())
    dropout_rate: a floating point number
    image_shapes: optional tuple of integer scalars.
      see comments for attention_image_summary()
    name: an optional string
    make_image_summary: True if you want an image summary.
    save_weights_to: an optional dictionary to capture attention weights
      for visualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    dropout_broadcast_dims:  an optional list of integers less than 4
      specifying in which dimensions to broadcast the dropout decisions.
      saves memory.
  Returns:
    A Tensor.
  """
  with tf.variable_scope(
      name, default_name="dot_product_attention", values=[q, k, v]) as scope:
    # [batch, num_heads, query_length, memory_length]
    logits = tf.matmul(q, k, transpose_b=True)
    if bias is not None:
      bias = common_layers.cast_like(bias, logits)
      logits += bias
    weights = tf.nn.softmax(logits, name="attention_weights")
    if save_weights_to is not None:
      save_weights_to[scope.name] = weights
      save_weights_to[scope.name + "/logits"] = logits
    # dropping out the attention links for each of the heads
    weights = common_layers.dropout_with_broadcast_dims(
        weights, 1.0 - dropout_rate, broadcast_dims=dropout_broadcast_dims)
    if common_layers.should_generate_summaries() and make_image_summary:
      attention_image_summary(weights, image_shapes)
    return tf.matmul(weights, v)




def dot_product_attention_relative(q,
                                   k,
                                   v,
                                   bias,
                                   max_relative_position,
                                   dropout_rate=0.0,
                                   image_shapes=None,
                                   name=None,
                                   make_image_summary=True):
  """Calculate relative position-aware dot-product self-attention.
  The attention calculation is augmented with learned representations for the
  relative position between each element in q and each element in k and v.
  Args:
    q: a Tensor with shape [batch, heads, length, depth].
    k: a Tensor with shape [batch, heads, length, depth].
    v: a Tensor with shape [batch, heads, length, depth].
    bias: bias Tensor.
    max_relative_position: an integer specifying the maximum distance between
        inputs that unique position embeddings should be learned for.
    dropout_rate: a floating point number.
    image_shapes: optional tuple of integer scalars.
    name: an optional string.
    make_image_summary: Whether to make an attention image summary.
  Returns:
    A Tensor.
  Raises:
    ValueError: if max_relative_position is not > 0.
  """
  if not max_relative_position:
    raise ValueError("Max relative position (%s) should be > 0 when using "
                     "relative self attention." % (max_relative_position))
  with tf.variable_scope(
      name, default_name="dot_product_attention_relative", values=[q, k, v]):

    # This calculation only works for self attention.
    # q, k and v must therefore have the same shape.
    q.get_shape().assert_is_compatible_with(k.get_shape())
    q.get_shape().assert_is_compatible_with(v.get_shape())

    # Use separate embeddings suitable for keys and values.
    depth = q.get_shape().as_list()[3]
    length = common_layers.shape_list(q)[2]
    relations_keys = _generate_relative_positions_embeddings(
        length, depth, max_relative_position, "relative_positions_keys")
    relations_values = _generate_relative_positions_embeddings(
        length, depth, max_relative_position, "relative_positions_values")

    # Compute self attention considering the relative position embeddings.
    logits = _relative_attention_inner(q, k, relations_keys, True)
    if bias is not None:
      logits += bias
    weights = tf.nn.softmax(logits, name="attention_weights")
    weights = tf.nn.dropout(weights, 1.0 - dropout_rate)
    if not tf.get_variable_scope().reuse and make_image_summary:
      attention_image_summary(weights, image_shapes)
    return _relative_attention_inner(weights, v, relations_values, False)





def dot_product_self_attention_relative_v2(q,
                                           k,
                                           v,
                                           bias,
                                           max_length=None,
                                           dropout_rate=0.0,
                                           image_shapes=None,
                                           name=None,
                                           make_image_summary=True,
                                           dropout_broadcast_dims=None):
  """Calculate relative position-aware dot-product self-attention.
  Only works for masked self-attention (no looking forward).
  TODO(noam): extend to unmasked self-attention
  The attention calculation is augmented with learned representations for the
  relative position between each element in q and each element in k and v.
  Args:
    q: a Tensor with shape [batch, heads, length, depth].
    k: a Tensor with shape [batch, heads, length, depth].
    v: a Tensor with shape [batch, heads, length, depth].
    bias: bias Tensor.
    max_length: an integer - changing this invalidates checkpoints
    dropout_rate: a floating point number.
    image_shapes: optional tuple of integer scalars.
    name: an optional string.
    make_image_summary: Whether to make an attention image summary.
    dropout_broadcast_dims:  an optional list of integers less than 4
      specifying in which dimensions to broadcast the dropout decisions.
      saves memory.
  Returns:
    A Tensor.
  """
  with tf.variable_scope(
      name,
      default_name="dot_product_self_attention_relative_v2",
      values=[q, k, v]):

    # This calculation only works for self attention.
    # q, k and v must therefore have the same shape.
    q.get_shape().assert_is_compatible_with(k.get_shape())
    q.get_shape().assert_is_compatible_with(v.get_shape())

    # Use separate embeddings suitable for keys and values.
    length = common_layers.shape_list(q)[2]
    assert max_length is not None

    # [batch, num_heads, query_length, memory_length]
    logits = tf.matmul(q, k, transpose_b=True)

    # now add relative logits
    # [batch, num_heads, query_length, max_length]
    rel_logits = common_layers.dense(q, max_length, name="rel0")
    # [batch, num_heads, query_length, max_length]
    rel_logits = tf.slice(rel_logits, [0, 0, 0, max_length - length],
                          [-1, -1, -1, -1])
    rel_logits = _relative_position_to_absolute_position_masked(rel_logits)
    logits += rel_logits

    if bias is not None:
      logits += bias
    weights = tf.nn.softmax(logits, name="attention_weights")
    # dropping out the attention links for each of the heads
    weights = common_layers.dropout_with_broadcast_dims(
        weights, 1.0 - dropout_rate, broadcast_dims=dropout_broadcast_dims)
    if common_layers.should_generate_summaries() and make_image_summary:
      attention_image_summary(weights, image_shapes)
    ret = tf.matmul(weights, v)
    # [batch, num_heads, query_length, memory_length]
    relative_weights = _absolute_position_to_relative_position_masked(weights)
    # [batch, num_heads, query_length, memory_length]
    relative_weights = tf.pad(
        relative_weights, [[0, 0], [0, 0], [0, 0], [max_length - length, 0]])
    relative_weights.set_shape([None, None, None, max_length])
    depth_v = common_layers.shape_list(v)[3]
    ret += common_layers.dense(relative_weights, depth_v, name="rel1")
    return ret





def masked_local_attention_1d(q,
                              k,
                              v,
                              block_length=128,
                              make_image_summary=False,
                              name=None):
  """Attention to the source position and a neighborhood to the left of it.
  The sequence is divided into blocks of length block_size.
  Attention for a given query position can only see memory positions
  less than or equal to the query position, in the corresponding block
  and the previous block.
  If mask_right is True, then a target position cannot see greater source
  positions.
  Args:
    q: a Tensor with shape [batch, heads, length, depth_k]
    k: a Tensor with shape [batch, heads, length, depth_k]
    v: a Tensor with shape [batch, heads, length, depth_v]
    block_length: an integer
    make_image_summary: a boolean, whether to make an attention image summary.
    name: an optional string
  Returns:
    a Tensor of shape [batch, heads, length, depth_v]
  """
  with tf.variable_scope(
      name, default_name="local_attention_1d", values=[q, k, v]):
    batch = common_layers.shape_list(q)[0]
    heads = common_layers.shape_list(q)[1]
    length = common_layers.shape_list(q)[2]
    if isinstance(block_length, tf.Tensor):
      const = tf.contrib.util.constant_value(block_length)
      if const is not None:
        block_length = int(const)

    # If (length < 2 * block_length), then we use only one block.
    if isinstance(length, int) and isinstance(block_length, int):
      block_length = length if length < block_length * 2 else block_length
    else:
      block_length = tf.where(
          tf.less(length, block_length * 2), length, block_length)
    depth_k = common_layers.shape_list(k)[3]
    depth_v = common_layers.shape_list(v)[3]
    original_length = length
    padding_size = tf.mod(-length, block_length)
    length += padding_size
    padding = [[0, 0], [0, 0], [0, padding_size], [0, 0]]
    q = tf.pad(q, padding)
    k = tf.pad(k, padding)
    v = tf.pad(v, padding)

    if isinstance(length, int) and isinstance(block_length, int):
      num_blocks = length // block_length
    else:
      num_blocks = tf.div(length, block_length)

    # compute attention for the first query block.
    first_q = tf.slice(q, [0, 0, 0, 0], [-1, -1, block_length, -1])
    first_k = tf.slice(k, [0, 0, 0, 0], [-1, -1, block_length, -1])
    first_v = tf.slice(v, [0, 0, 0, 0], [-1, -1, block_length, -1])
    first_output = dot_product_attention(
        first_q,
        first_k,
        first_v,
        attention_bias_lower_triangle(block_length),
        make_image_summary=make_image_summary,
        name="fist_block")

    # compute attention for all subsequent query blocks.
    q = tf.reshape(q, [batch, heads, num_blocks, block_length, depth_k])
    k = tf.reshape(k, [batch, heads, num_blocks, block_length, depth_k])
    v = tf.reshape(v, [batch, heads, num_blocks, block_length, depth_v])

    def local(x, depth):
      """Create a local version of the keys or values."""
      prev_block = tf.slice(x, [0, 0, 0, 0, 0],
                            [-1, -1, num_blocks - 1, -1, -1])
      cur_block = tf.slice(x, [0, 0, 1, 0, 0], [-1, -1, -1, -1, -1])
      local_block = tf.concat([prev_block, cur_block], 3)
      return tf.reshape(local_block,
                        [batch, heads, num_blocks - 1, block_length * 2, depth])

    local_k = local(k, depth_k)
    local_v = local(v, depth_v)
    tail_q = tf.slice(q, [0, 0, 1, 0, 0], [-1, -1, -1, -1, -1])
    tail_q = tf.reshape(tail_q,
                        [batch, heads, num_blocks - 1, block_length, depth_k])
    local_length = common_layers.shape_list(local_k)[3]

    # [batch, heads, num_blocks - 1, block_length, local_length]
    attention = tf.matmul(tail_q, local_k, transpose_b=True)

    # make sure source_pos <= target_pos
    good_part = common_layers.ones_matrix_band_part(block_length, local_length,
                                                    -1, block_length)
    mask = (1.0 - good_part) * -1e9
    mask = common_layers.cast_like(mask, attention)
    attention += tf.reshape(mask, [1, 1, 1, block_length, local_length])
    attention = tf.nn.softmax(attention)
    # TODO(noam): figure out how to show a summary for the remaining blocks.
    # The naive way currently causes errors due to empty tensors.
    # output: [batch, heads, num_blocks-1, block_length, depth_v]
    output = tf.matmul(attention, local_v)
    output = tf.reshape(
        output, [batch, heads, (num_blocks - 1) * block_length, depth_v])
    output = tf.concat([first_output, output], axis=2)
    output = tf.slice(output, [0, 0, 0, 0], [-1, -1, original_length, -1])
    output = tf.reshape(output, [batch, heads, original_length, depth_v])
    return output





def multihead_self_attention_memory_efficient(x,
                                              bias,
                                              num_heads,
                                              head_size=None,
                                              epsilon=1e-6,
                                              forget=True,
                                              test_vars=None,
                                              name=None):
  """Multihead scaled-dot-product self-attention.
  Includes layer norm.
  Returns multihead-self-attention(layer_norm(x))
  Computes one attention head at a time to avoid exhausting memory.
  If forget=True, then forget all forwards activations and recompute on
  the backwards pass.
  Args:
    x: a Tensor with shape [batch, length, input_size]
    bias: an attention bias tensor broadcastable to [batch, 1, length, length]
    num_heads: an integer
    head_size: an optional integer - defaults to input_size/num_heads
    epsilon: a float, for layer norm
    forget: a boolean - forget forwards activations and recompute on backprop
    test_vars: optional tuple of variables for testing purposes
    name: an optional string
  Returns:
    A Tensor.
  """
  io_size = x.get_shape().as_list()[-1]
  if head_size is None:
    assert io_size % num_heads == 0
    head_size = io_size / num_heads

  def forward_internal(x, wqkv, wo, attention_bias, norm_scale, norm_bias):
    """Forward function."""
    n = common_layers.layer_norm_compute_python(x, epsilon, norm_scale,
                                                norm_bias)
    wqkv_split = tf.unstack(wqkv, num=num_heads)
    wo_split = tf.unstack(wo, num=num_heads)
    y = 0
    for h in range(num_heads):
      with tf.control_dependencies([y] if h > 0 else []):
        combined = tf.nn.conv1d(n, wqkv_split[h], 1, "SAME")
        q, k, v = tf.split(combined, 3, axis=2)
        o = scaled_dot_product_attention_simple(q, k, v, attention_bias)
        y += tf.nn.conv1d(o, wo_split[h], 1, "SAME")
    return y

  key = (
      "multihead_self_attention_memory_efficient %s %s" % (num_heads, epsilon))
  if not forget:
    forward_fn = forward_internal
  elif key in _function_cache:
    forward_fn = _function_cache[key]
  else:

    @function.Defun(compiled=True)
    def grad_fn(x, wqkv, wo, attention_bias, norm_scale, norm_bias, dy):
      """Custom gradient function."""
      with tf.control_dependencies([dy]):
        n = common_layers.layer_norm_compute_python(x, epsilon, norm_scale,
                                                    norm_bias)
        wqkv_split = tf.unstack(wqkv, num=num_heads)
        wo_split = tf.unstack(wo, num=num_heads)
        deps = []
        dwqkvs = []
        dwos = []
        dn = 0
        for h in range(num_heads):
          with tf.control_dependencies(deps):
            combined = tf.nn.conv1d(n, wqkv_split[h], 1, "SAME")
            q, k, v = tf.split(combined, 3, axis=2)
            o = scaled_dot_product_attention_simple(q, k, v, attention_bias)
            partial_y = tf.nn.conv1d(o, wo_split[h], 1, "SAME")
            pdn, dwqkvh, dwoh = tf.gradients(
                ys=[partial_y],
                xs=[n, wqkv_split[h], wo_split[h]],
                grad_ys=[dy])
            dn += pdn
            dwqkvs.append(dwqkvh)
            dwos.append(dwoh)
            deps = [dn, dwqkvh, dwoh]
        dwqkv = tf.stack(dwqkvs)
        dwo = tf.stack(dwos)
        with tf.control_dependencies(deps):
          dx, dnorm_scale, dnorm_bias = tf.gradients(
              ys=[n], xs=[x, norm_scale, norm_bias], grad_ys=[dn])
        return (dx, dwqkv, dwo, tf.zeros_like(attention_bias), dnorm_scale,
                dnorm_bias)

    @function.Defun(
        grad_func=grad_fn, compiled=True, separate_compiled_gradients=True)
    def forward_fn(x, wqkv, wo, attention_bias, norm_scale, norm_bias):
      return forward_internal(x, wqkv, wo, attention_bias, norm_scale,
                              norm_bias)

    _function_cache[key] = forward_fn

  if bias is not None:
    bias = tf.squeeze(bias, 1)
  with tf.variable_scope(name, default_name="multihead_attention", values=[x]):
    # TODO(noam): it would be nice to save memory by casting x to float16
    # here, but this causes problems with the gradients.  Figure out if there
    # is a way to leave the gradients as float32.
    if test_vars is not None:
      wqkv, wo, norm_scale, norm_bias = list(test_vars)
    else:
      wqkv = tf.get_variable(
          "wqkv", [num_heads, 1, io_size, 3 * head_size],
          initializer=tf.random_normal_initializer(stddev=io_size**-0.5))
      wo = tf.get_variable(
          "wo", [num_heads, 1, head_size, io_size],
          initializer=tf.random_normal_initializer(
              stddev=(head_size * num_heads)**-0.5))
      norm_scale, norm_bias = common_layers.layer_norm_vars(io_size)
    y = forward_fn(x, wqkv, wo, bias, norm_scale, norm_bias)
    y.set_shape(x.get_shape())
    return y





def dot_product_area_attention(q,
                               k,
                               v,
                               bias,
                               dropout_rate=0.0,
                               image_shapes=None,
                               name=None,
                               attention_image_summary=None,
                               save_weights_to=None,
                               dropout_broadcast_dims=None,
                               max_area_width=1,
                               max_area_height=1,
                               memory_height=1,
                               area_key_mode="mean",
                               area_value_mode="sum",
                               top_k_areas=0,
                               area_temperature=1.0,
                               training=True):
  """Dot-product area attention.
  Args:
    q: Tensor with shape [..., length_q, depth_k].
    k: Tensor with shape [..., length_kv, depth_k]. Leading dimensions must
      match with q.
    v: Tensor with shape [..., length_kv, depth_v] Leading dimensions must
      match with q.
    bias: bias Tensor (see attention_bias())
    dropout_rate: a float.
    image_shapes: optional tuple of integer scalars.
      see comments for attention_image_summary()
    name: an optional string
    attention_image_summary: the callback for making image summary of attention.
    save_weights_to: an optional dictionary to capture attention weights
      for visualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    dropout_broadcast_dims: an optional list of integers less than rank of q.
      Specifies in which dimensions to broadcast the dropout decisions.
    max_area_width: the max width allowed for an area.
    max_area_height: the max height allowed for an area.
    memory_height: the height of the memory.
    area_key_mode: the mode for computing area keys, which can be "mean",
      "concat", "sum", "sample_concat", and "sample_sum".
    area_value_mode: the mode for computing area values, which can be either
      "mean", or "sum".
    top_k_areas: Use the top key areas for attention.
    area_temperature: the temperature for attention softmax.
    training: indicating if it is in the training mode.
  Returns:
    Tensor with shape [..., length_q, depth_v].
  """

  tf.logging.info("dot_product_area_attention: "
                  "area_h=%d, area_w=%d, mem_h=%d, "
                  "area_key_mode=%s, area_value_mode=%s, "
                  "area_temperature=%f",
                  max_area_height, max_area_width, memory_height,
                  area_key_mode, area_value_mode,
                  area_temperature)
  with tf.variable_scope(
      name, default_name="dot_product_area_attention",
      values=[q, k, v]) as scope:
    mem_shape = common_layers.shape_list(k)
    batch_size = mem_shape[0]
    head_size = mem_shape[1]
    length = mem_shape[2]
    depth = mem_shape[3]
    k_area = compute_area_key(
        tf.reshape(k, [-1, length, depth]),
        max_area_width=max_area_width,
        max_area_height=max_area_height,
        height=memory_height,
        mode=area_key_mode,
        training=training)
    if area_value_mode == "mean":
      v_area, _, _, _, _ = compute_area_features(
          tf.reshape(v, [-1, length, depth]), max_area_width=max_area_width,
          max_area_height=max_area_height, height=memory_height)
    elif area_value_mode == "max":
      v_area, _, _ = basic_pool(tf.reshape(v, [-1, length, depth]),
                                max_area_width=max_area_width,
                                max_area_height=max_area_height,
                                height=memory_height,
                                fn=tf.reduce_max)
    elif area_value_mode == "sum":
      _, _, v_area, _, _ = compute_area_features(
          tf.reshape(v, [-1, length, depth]), max_area_width=max_area_width,
          max_area_height=max_area_height, height=memory_height)
    else:
      raise ValueError("Unsupported area value mode=%s" % area_value_mode)
    k = tf.reshape(k_area, [batch_size, head_size, -1, depth])
    v = tf.reshape(v_area, [batch_size, head_size, -1, depth])
    logits = tf.matmul(q, k, transpose_b=True)  # [..., length_q, length_kv]
    if bias is not None:
      bias = common_layers.cast_like(bias, logits)
      with tf.name_scope("compute_area_att_bias", values=[bias]):
        bias_shape = common_layers.shape_list(bias)
        mem_length = bias_shape[-1]
        bias_values = tf.reshape(
            tf.to_float(tf.less(bias, -1)), [-1, mem_length, 1])
        _, _, padding_sum, _, _ = compute_area_features(
            bias_values, max_area_width=max_area_width,
            max_area_height=max_area_height, height=memory_height)
        bias = tf.where(
            tf.cast(tf.to_int32(padding_sum), tf.bool),
            tf.fill(tf.shape(padding_sum), -np.inf),
            tf.zeros_like(padding_sum, dtype=tf.float32))
        bias = tf.reshape(bias,
                          [bias_shape[0], bias_shape[1],
                           bias_shape[2], -1])
      logits += bias
    logits = logits / area_temperature
    weights = tf.nn.softmax(logits, name="attention_weights")
    if top_k_areas > 0:
      tf.logging.info("area_attention top_k_areas=%d", top_k_areas)
      top_k = tf.minimum(common_layers.shape_list(weights)[-1], top_k_areas)
      top_weights, _ = tf.nn.top_k(weights, k=top_k)
      min_values = tf.reduce_min(top_weights, -1, keepdims=True)
      weights = tf.where(tf.greater_equal(weights, min_values),
                         weights, tf.zeros_like(weights))
      weights = tf.div(weights, tf.reduce_sum(weights, -1, keepdims=True))
    if save_weights_to is not None:
      save_weights_to[scope.name] = weights
      save_weights_to[scope.name + "/logits"] = logits
    # Drop out attention links for each head.
    weights = common_layers.dropout_with_broadcast_dims(
        weights, 1.0 - dropout_rate, broadcast_dims=dropout_broadcast_dims)
    if common_layers.should_generate_summaries() and attention_image_summary:
      attention_image_summary(weights, image_shapes)
    return tf.matmul(weights, v)




def multihead_attention(query_antecedent,
                        memory_antecedent,
                        bias,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        dropout_rate,
                        attention_type="dot_product",
                        max_relative_position=None,
                        heads_share_relative_embedding=False,
                        add_relative_to_values=False,
                        image_shapes=None,
                        block_length=128,
                        block_width=128,
                        q_filter_width=1,
                        kv_filter_width=1,
                        q_padding="VALID",
                        kv_padding="VALID",
                        cache=None,
                        gap_size=0,
                        num_memory_blocks=2,
                        name="multihead_attention",
                        save_weights_to=None,
                        make_image_summary=True,
                        dropout_broadcast_dims=None,
                        vars_3d=False,
                        layer_collection=None,
                        recurrent_memory=None,
                        chunk_number=None,
                        hard_attention_k=0,
                        gumbel_noise_weight=0.0,
                        max_area_width=1,
                        max_area_height=1,
                        memory_height=1,
                        area_key_mode="mean",
                        area_value_mode="sum",
                        training=True,
                        **kwargs):
  """Multihead scaled-dot-product attention with input/output transformations.
  Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels] or None
    bias: bias Tensor (see attention_bias())
    total_key_depth: an integer
    total_value_depth: an integer
    output_depth: an integer
    num_heads: an integer dividing total_key_depth and total_value_depth
    dropout_rate: a floating point number
    attention_type: a string, either "dot_product", "dot_product_relative",
                    "local_mask_right", "local_unmasked", "masked_dilated_1d",
                    "unmasked_dilated_1d", graph, or any attention function
                    with the signature (query, key, value, **kwargs)
    max_relative_position: Maximum distance between inputs to generate
                           unique relation embeddings for. Only relevant
                           when using "dot_product_relative" attention.
    heads_share_relative_embedding: boolean to share relative embeddings
    add_relative_to_values: a boolean for whether to add relative component to
                            values.
    image_shapes: optional tuple of integer scalars.
                  see comments for attention_image_summary()
    block_length: an integer - relevant for "local_mask_right"
    block_width: an integer - relevant for "local_unmasked"
    q_filter_width: An integer specifying how wide you want the query to be.
    kv_filter_width: An integer specifying how wide you want the keys and values
                     to be.
    q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
               kv_padding: One of "VALID", "SAME" or "LEFT". Default is "VALID":
               no padding.
    cache: dict containing Tensors which are the results of previous
           attentions, used for fast decoding. Expects the dict to contrain two
           keys ('k' and 'v'), for the initial call the values for these keys
           should be empty Tensors of the appropriate shape.
               'k' [batch_size, 0, key_channels]
               'v' [batch_size, 0, value_channels]
    gap_size: Integer option for dilated attention to indicate spacing between
              memory blocks.
    num_memory_blocks: Integer option to indicate how many memory blocks to look
                       at.
    name: an optional string.
    save_weights_to: an optional dictionary to capture attention weights
      for vizualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.
    dropout_broadcast_dims:  an optional list of integers less than 4
      specifying in which dimensions to broadcast the dropout decisions.
      saves memory.
    vars_3d: use 3-dimensional variables for input/output transformations
    layer_collection: A tensorflow_kfac.LayerCollection. Only used by the
      KFAC optimizer. Default is None.
    recurrent_memory: An optional transformer_memory.RecurrentMemory, which
      retains state across chunks. Default is None.
    chunk_number: an optional integer Tensor with shape [batch] used to operate
      the recurrent_memory.
    hard_attention_k: integer, if > 0 triggers hard attention (picking top-k).
    gumbel_noise_weight: if > 0, apply Gumbel noise with weight
      `gumbel_noise_weight` before picking top-k. This is a no op if
      hard_attention_k <= 0.
    max_area_width: the max width allowed for an area.
    max_area_height: the max height allowed for an area.
    memory_height: the height of the memory.
    area_key_mode: the mode for computing area keys, which can be "mean",
      "concat", "sum", "sample_concat", and "sample_sum".
    area_value_mode: the mode for computing area values, which can be either
      "mean", or "sum".
    training: indicating if it is in the training mode.
    **kwargs (dict): Parameters for the attention function.
  Caching:
    WARNING: For decoder self-attention, i.e. when memory_antecedent == None,
    the caching assumes that the bias contains future masking.
    The caching works by saving all the previous key and value values so that
    you are able to send just the last query location to this attention
    function. I.e. if the cache dict is provided it assumes the query is of the
    shape [batch_size, 1, hidden_dim] rather than the full memory.
  Returns:
    The result of the attention transformation. The output shape is
        [batch_size, length_q, hidden_dim]
    unless the cache dict is provided in which case only the last memory
    position is calculated and the output shape is [batch_size, 1, hidden_dim]
    Optionally returns an additional loss parameters (ex: load balance loss for
    the experts) returned by the attention_type function.
  Raises:
    ValueError: if the key depth or value depth are not divisible by the
      number of attention heads.
  """
  if total_key_depth % num_heads != 0:
    raise ValueError("Key depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_key_depth, num_heads))
  if total_value_depth % num_heads != 0:
    raise ValueError("Value depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_value_depth, num_heads))
  vars_3d_num_heads = num_heads if vars_3d else 0

  if layer_collection is not None:
    if cache is not None:
      raise ValueError("KFAC implementation only supports cache is None.")
    if vars_3d:
      raise ValueError("KFAC implementation does not support 3d vars.")

  if recurrent_memory is not None:
    if memory_antecedent is not None:
      raise ValueError("Recurrent memory requires memory_antecedent is None.")
    if cache is not None:
      raise ValueError("Cache is not supported when using recurrent memory.")
    if vars_3d:
      raise ValueError("3d vars are not supported when using recurrent memory.")
    if layer_collection is not None:
      raise ValueError("KFAC is not supported when using recurrent memory.")
    if chunk_number is None:
      raise ValueError("chunk_number is required when using recurrent memory.")

  with tf.variable_scope(name, default_name="multihead_attention",
                         values=[query_antecedent, memory_antecedent]):

    if recurrent_memory is not None:
      (
          recurrent_memory_transaction,
          query_antecedent, memory_antecedent, bias,
      ) = recurrent_memory.pre_attention(
          chunk_number,
          query_antecedent, memory_antecedent, bias,
      )

    if memory_antecedent is None:
      q, k, v = compute_qkv(query_antecedent, memory_antecedent,
                            total_key_depth, total_value_depth, q_filter_width,
                            kv_filter_width, q_padding, kv_padding,
                            vars_3d_num_heads=vars_3d_num_heads,
                            layer_collection=layer_collection)

    q = split_heads(q, num_heads)
    k = split_heads(k, num_heads)
    v = split_heads(v, num_heads)

    key_depth_per_head = total_key_depth // num_heads
    if not vars_3d:
      q *= key_depth_per_head**-0.5

    additional_returned_value = None
    if callable(attention_type):  # Generic way to extend multihead_attention
      x = attention_type(q, k, v, **kwargs)
      if isinstance(x, tuple):
        x, additional_returned_value = x  # Unpack
    elif attention_type == "dot_product":
      if max_area_width > 1 or max_area_height > 1:
        x = area_attention.dot_product_area_attention(
            q, k, v, bias, dropout_rate, image_shapes,
            save_weights_to=save_weights_to,
            dropout_broadcast_dims=dropout_broadcast_dims,
            max_area_width=max_area_width,
            max_area_height=max_area_height,
            memory_height=memory_height,
            area_key_mode=area_key_mode,
            area_value_mode=area_value_mode,
            training=training)
      else:
        x = dot_product_attention(
            q, k, v, bias, dropout_rate, image_shapes,
            save_weights_to=save_weights_to,
            make_image_summary=make_image_summary,
            dropout_broadcast_dims=dropout_broadcast_dims,
            activation_dtype=kwargs.get("activation_dtype"),
            hard_attention_k=hard_attention_k,
            gumbel_noise_weight=gumbel_noise_weight)
    elif attention_type == "dot_product_relative":
      x = dot_product_attention_relative(
          q,
          k,
          v,
          bias,
          max_relative_position,
          dropout_rate,
          image_shapes,
          save_weights_to=save_weights_to,
          make_image_summary=make_image_summary,
          cache=cache is not None,
          allow_memory=recurrent_memory is not None,
          hard_attention_k=hard_attention_k,
          gumbel_noise_weight=gumbel_noise_weight)
    elif attention_type == "dot_product_unmasked_relative_v2":
      x = dot_product_unmasked_self_attention_relative_v2(
          q,
          k,
          v,
          bias,
          max_relative_position,
          dropout_rate,
          image_shapes,
          save_weights_to=save_weights_to,
          make_image_summary=make_image_summary,
          dropout_broadcast_dims=dropout_broadcast_dims,
          heads_share_relative_embedding=heads_share_relative_embedding,
          add_relative_to_values=add_relative_to_values)
    elif attention_type == "dot_product_relative_v2":
      x = dot_product_self_attention_relative_v2(
          q,
          k,
          v,
          bias,
          max_relative_position,
          dropout_rate,
          image_shapes,
          save_weights_to=save_weights_to,
          make_image_summary=make_image_summary,
          dropout_broadcast_dims=dropout_broadcast_dims,
          heads_share_relative_embedding=heads_share_relative_embedding,
          add_relative_to_values=add_relative_to_values)

    x = combine_heads(x)

    # Set last dim specifically.
    x.set_shape(x.shape.as_list()[:-1] + [total_value_depth])

    if vars_3d:
      o_var = tf.get_variable(
          "o", [num_heads, total_value_depth // num_heads, output_depth])
      o_var = tf.cast(o_var, x.dtype)
      o_var = tf.reshape(o_var, [total_value_depth, output_depth])
      x = tf.tensordot(x, o_var, axes=1)
    else:
      x = common_layers.dense(
          x, output_depth, use_bias=False, name="output_transform",
          layer_collection=layer_collection)

    if recurrent_memory is not None:
      x = recurrent_memory.post_attention(recurrent_memory_transaction, x)
    if additional_returned_value is not None:
      return x, additional_returned_value
    return x
