
#' Vanilla Luong attention scoring mechanism (multiplicative)
#' Expects query shape [batch, units] # Usually state vector from decoder
#' Expects key shape [batch, seqlen, units] # Usually return_states from encoder
#' @export
compute_luong_score <-
  function(query, keys, query_depth, scale = NULL, return_context = TRUE) {
    # stopifnot(query_depth == shape_list2(keys)[[3]])
    
    processed_query <-
      layer_dense(query, units = query_depth, use_bias = FALSE) %>%
      tf$expand_dims(axis = 1L)
    
    score <-
      tf$matmul(processed_query, keys, transpose_b = TRUE) %>%
      tf$transpose(list(0L, 2L, 1L))
    
    alignments <-
      tf$nn$softmax(if (is.null(scale)) score else scale * score)
    
    
    if (return_context) {
      context <-
        tf$keras$backend$sum(keys * alignments,
                             axis = -1L,
                             keepdims = FALSE)
      return(list(alignments, context))
    }
    
    alignments
  }



#' Vanilla Bahdanau attention mechanism without normalization (additive)
#' @export
compute_bahdanau_score <-
  function(query, keys, query_depth, attention_v = NULL) {
    
    if (is.null(attention_v))
      attention_v <- tf$Variable(tf$random$normal(shape = list(query_depth)))
    processed_query <-
      layer_dense(query, units = query_depth, use_bias = FALSE) %>%
      tf$expand_dims(1L)
    
    scores <-
      tf$reduce_sum(attention_v * tf$tanh(keys + processed_query), list(2L))
    
    alignments <- tf$nn$softmax(scores)
    
    alignments
  }

