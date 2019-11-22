
def compute_attention_component(antecedent,
                                total_depth,
                                filter_width=1,
                                padding="VALID",
                                name="c"):
  """Computes attention compoenent (query, key or value).
  Args:
    antecedent: a Tensor with shape [batch, length, channels]
    total_depth: an integer
    filter_width: An integer specifying how wide you want the attention
      component to be.
    padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
    name: a string specifying scope name.
  Returns:
    c : [batch, length, depth] tensor
  """
  if filter_width == 1:
    return common_layers.dense(
        antecedent, total_depth, use_bias=False, name=name)
  else:
    return common_layers.conv1d(
        antecedent, total_depth, filter_width, padding, name=name)


def compute_qkv(query_antecedent,
                memory_antecedent,
                total_key_depth,
                total_value_depth,
                q_filter_width=1,
                kv_filter_width=1,
                q_padding="VALID",
                kv_padding="VALID"):
  """Computes query, key and value.
  Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels]
    total_key_depth: an integer
    total_value_depth: an integer
    q_filter_width: An integer specifying how wide you want the query to be.
    kv_filter_width: An integer specifying how wide you want the keys and values
    to be.
    q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
    kv_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
  Returns:
    q, k, v : [batch, length, depth] tensors
  """
  if memory_antecedent is None:
    memory_antecedent = query_antecedent
  q = compute_attention_component(query_antecedent, total_key_depth,
                                  q_filter_width, q_padding, "q")
  k = compute_attention_component(memory_antecedent, total_key_depth,
                                  kv_filter_width, kv_padding, "k")
  v = compute_attention_component(memory_antecedent, total_value_depth,
                                  kv_filter_width, kv_padding, "v")
  return q, k, v



