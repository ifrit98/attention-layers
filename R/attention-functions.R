

# TODO: Make this an R6 layer?
#' Input query, key, and value matrices are used to compute dot product
#' attention. (Vaswani et al. 2017)
#' q: a Tensor with shape [batch, length_q,  depth_k]
#' k: a Tensor with shape [batch, length_kv, depth_k]
#' v: a Tensor with shape [batch, length_kv, depth_v]
#' @export
dot_product_attention_1d <-
  function(q,
           k,
           v,
           bias = NULL,
           dropout = 0,
           name = "dot_product_attention") {
    
    q_shape <- shape_list2(q)
    scalar  <-
      tf$math$rsqrt(tf$cast(q_shape[[length(q_shape)]], tf$float32))
    logits  <- tf$matmul(q * scalar, k, transpose_b = TRUE)
    
    if (!is.null(bias))
      logits <- logits + bias
    
    weights <- tf$nn$softmax(logits, name = "attention_weights")
    
    x <- tf$matmul(weights, v)
    
    x
  }




# TODO: hard_attention_k: integer, if > 0 triggers hard attention (pick top-k)
# TODO: Add callable option to attention_type
#' Multihead attention mechanism
#' 
#' With num_heads == 1, becomes simple dot product attention
#' @param query  Tensor [batch, seqlen, depth_q]
#' @param memory Tensor [batch, seqlen, depth_m]
#' @param bias Bias tensor passed to attention function
#' @param key_depth Specify units for key component
#' @param value_depth Specify units for value component
#' @param output_depth Specify feature dim of final output
#' @param num_heads Specify number of heads to break input space up by
#' @param dropout Float value to add dropout to attention function
#' @param attention_type Character value of attention type
#' @param vars_3d use 3-dimensional variables for input/output transformations
#' @export
multihead_attention <- function(query,
                                memory = NULL,
                                bias = NULL,
                                key_depth = 64L,
                                value_depth = 64L,
                                output_depth = 128L,
                                num_heads = 4L,
                                dropout = 0,
                                attention_type = "dot_product",
                                q_filter_width = 1L,
                                kv_filter_width = 1L,
                                q_padding = "same",
                                kv_padding = "same",
                                max_area_width = 1L,
                                max_area_height = 1L,
                                area_height = 1L,
                                area_key_mode = "mean",
                                area_value_mode = "sum",
                                vars_3d = FALSE) {
  stopifnot(key_depth %% num_heads == 0, value_depth %% num_heads == 0)
  
  vars_3d_num_heads <- if (vars_3d) num_heads else 0
  
  c(q, k, v) %<-% compute_qkv(query, 
                              memory, 
                              key_depth, 
                              value_depth, 
                              q_filter_width, 
                              kv_filter_width,
                              vars_3d_num_heads = vars_3d_num_heads)
  
  q <- split_heads(q, num_heads)
  k <- split_heads(k, num_heads)
  v <- split_heads(v, num_heads)
  
  key_depth_per_head <- key_depth %/% num_heads
  
  if (!vars_3d) 
    q %<>% `*`(key_depth_per_head^(-0.5))
  
  if (attention_type == "dot_product")
    x <- dot_product_area_attention_1d(q, k, v, bias, dropout)
  else if (attention_type == "dot_product_area")
    x <- dot_product_area_attention_1d(
      q, k, v, bias, dropout, max_area_width, max_area_height, area_height)
  else
    stop("No other attention types currently implemented...")
  
  x <- combine_heads(x)
  x <- 
    if (vars_3d)
      tf$Variable(
        tf$glorot_normal_initializer()(
          shape = list(num_heads,
                       as.integer(value_depth %/% num_heads),
                       output_depth),
          dtype = x$dtype
        ), trainable = TRUE,
        name = "output_kernel_3d") %>% 
    tf$reshape(list(value_depth, output_depth)) %>% 
    {tf$tensordot(x, ., axes = 1L)}
  else
    layer_dense(x, output_depth, use_bias = FALSE, name = "output_transform")
  
  x
}




#' Strided block local self-attention.
#'
#' The sequence is divided into blocks of length block_length. 
#' Attention for agiven query position can see all memory positions 
#' in the corresponding block and filter_width many positions to 
#' the left and right of the block.
#' q Tensor [batch, heads, length, depth_k]
#' k Tensor [batch, heads, length, depth_k]
#' v Tensor [batch, heads, length, depth_v]
#' Returns Tensor [batch, heads, length, depth_v]
#' @export
local_attention_1d <-
  function(q,
           k,
           v,
           block_length = 128L,
           filter_width = 100L,
           name = NULL){
    # Shape assertions go here
    q_shape <- shape_list2(q)
    
    c(batch, num_heads, original_length, original_depth) %<-% q_shape
    
    pad_to_multiple <- function(x, pad_length) {
      x_length <- shape_list2(x)[[3]]
      tf$pad(x, list(c(0L, 0L),
                     c(0L, 0L),
                     c(0L, -x_length %% pad_length),
                     c(0L, 0L)))
    }
    
    pad_l_and_r <- function(x, pad_length) {
      x_length <- shape_list2(x)[[3]]
      tf$pad(x, list(c(0L, 0L),
                     c(0L, 0L),
                     c(pad_length, pad_length),
                     c(0L, 0L)))
    }
    
    # Set up query blocks.
    # [batch, heads, blocks_q, block_length, depth_k]
    q <- pad_to_multiple(q, block_length)
    q <- reshape_by_blocks(q, shape_list2(q), block_length)
    
    total_query_blocks <- shape_list2(q)[[3]]
    
    
    blocks_per_filter_width <- as.integer(filter_width %/% block_length)
    remaining <- filter_width %% block_length
    
    k <- pad_to_multiple(k, block_length)
    v <- pad_to_multiple(v, block_length)
    k <- pad_l_and_r(k, filter_width + block_length - remaining)
    v <- pad_l_and_r(v, filter_width + block_length - remaining)
    k <- reshape_by_blocks(k, shape_list2(k), block_length)
    v <- reshape_by_blocks(v, shape_list2(v), block_length)
    
    total_kv_blocks <- shape_list2(k)[[3]]
    
    if (remaining) {
      left_partial_block_k <- tf$slice(
        k, list(0L, 0L, 0L, block_length - remaining, 0L),
        list(-1L, -1L, total_query_blocks, -1L, -1L)
      )
      left_partial_block_v <- tf$slice(
        k, list(0L, 0L, 0L, block_length - remaining, 0L),
        list(-1L, -1L, total_query_blocks, -1L, -1L)
      )
      right_partial_block_k = tf$slice(
        k, list(0L, 0L, total_kv_blocks - total_query_blocks, 0L, 0L),
        list(-1L, -1L, -1L, remaining, -1L)
      )
      right_partial_block_v = tf$slice(
        k, list(0L, 0L, total_kv_blocks - total_query_blocks, 0L, 0L),
        list(-1L, -1L, -1L, remaining, -1L)
      )
      
      slices <- list(c(left_partial_block_k, left_partial_block_v),
                     c(right_partial_block_k, right_partial_block_v))
    }
    
    # Prepare the rest of the blocks
    first_block_index <- if (remaining) 1L else 0L
    attention_blocks  <- 2 * blocks_per_filter_width + 1L
    
    n <- first_block_index:attention_blocks + first_block_index
    
    blocks <- lapply(1:n, function(i) {
      block_k <- tf$slice(k, list(0L, 0L, i, 0L, 0L),
                          list(-1L, -1L, total_query_blocks, -1L, -1L))
      block_v <- tf$slice(k, list(0L, 0L, i, 0L, 0L),
                          list(-1L, -1L, total_query_blocks, -1L, -1L))
      c(block_k, block_v)
    })
    
    slices <- append(slices, blocks)
    
    k <- tf$concat(lapply(slices, function(b) b[[1]]), axis = 3L)
    v <- tf$concat(lapply(slices, function(b) b[[2]]), axis = 3L)
    
    attention_bias <- tf$expand_dims(embedding_to_padding(k) * -1e9, axis = -2L)
    shape_v <- shape_list2(v)
    depth_v <- shape_v[[length(shape_v)]]
    
    output <- 
      dot_product_attention_1d(q, k, v, attention_bias, name = "local_1d") %>% 
      tf$reshape(list(batch, num_heads, original_length, depth_v))
    
    # Remove the padding if introduced.
    output <- tf$slice(output, 
                       list(0L, 0L, 0L, 0L),
                       list(-1L, -1L, original_length, -1L))
    
    output$set_shape(list(batch, num_heads, original_length, depth_v))
    
    output
  }

