#' Pools for an area in features_2d.
.pool_one_shape <-
  function(features_2d,
           area_width,
           area_height,
           batch,
           width,
           height,
           depth,
           fn = tf$reduce_max,
           name = NULL) {
    
    images <- list()
    
    i <- 1L
    for (y_shift in seq(0L, area_height-1L)) {
      image_height <- tf$maximum(height - area_height + 1L + y_shift, 0L)
      for (x_shift in seq(0L, area_width-1L)) {
        image_width <- tf$maximum(width - area_width + 1L + x_shift, 0L)
        area <- 
          features_2d[ , y_shift:image_height, x_shift:image_width, , 
                       style = "python"]
        flatten_area <- tf$reshape(area, list(batch, -1L, depth, 1L))
        images[[i]]  <- flatten_area
        
        i <- i + 1L
      }
    }
    
    image_tensor <- tf$concat(images, axis = 3L)
    max_tensor   <- fn(image_tensor, axis = 3L)

    max_tensor
  }



#' Pools for each area based on a given pooling function (fn)
#' @export
#'
#' @param features a Tensor in a shape of [batch_size, height * width, depth]
#' @param max_area_width the max width allowed for an area.
#' @param max_area_height the max height allowed for an area.
#' @param fn the TF function for the pooling.
#' @param name the namescope.
#' 
#' @return pool_results: A Tensor of shape [batch_size, num_areas, depth]
#' @return area_heights: A Tensor of shape [batch_size, num_areas, 1]
#' @return area_widths:  A Tensor of shape [batch_size, num_areas, 1]
basic_pool <-
  function(features = tf$random$normal(shape = list(8L, 512L, 32L)),
           max_area_width,
           max_area_height = 1L,
           height = 1L,
           fn = tf$reduce_max,
           name = NULL) {
      feature_shape <- shape_list2(features)
      batch         <- feature_shape[[1]]
      length        <- feature_shape[[length(feature_shape) - 1L]]
      depth         <- feature_shape[[length(feature_shape)]]
      height        <- as.integer(height)
      width         <- as.integer(length %/% height)
      
      c(width, height) %<-%
        validate_area_parameters(width, height, feature_shape)
      
      # if (is.null(max_area_width) || is.null(max_area_height))
      #   c(max_area_width, max_area_height) %<-% c(width %/% 2, height %/% 2)
      
      features_2d <-
        tf$reshape(features, list(batch, height, width, depth))
      
      height_list <- list()
      width_list  <- list()
      pool_list   <- list()
      size_tensor <-
        tf$ones_like(features_2d[, , , 0L, style = "python"], dtype = tf$int32)
      
      i <- 1L
      for (area_height in seq(0L, max_area_height - 1L)) {
        for (area_width in seq(0L, max_area_width - 1L)) {
          pool_tensor = .pool_one_shape(
            features_2d,
            area_width  = area_width  + 1L,
            area_height = area_height + 1L,
            batch       = batch,
            width       = width,
            height      = height,
            depth       = depth,
            fn          = fn
          )
          
          pool_list[[i]] <-
            tf$reshape(pool_tensor, list(batch, -1L, depth))
          
          h <- 
            size_tensor[, area_height:NULL, area_width:NULL, style = "python"] * 
            tf$cast((area_height + 1L), tf$int32)
          
          w <- 
            size_tensor[, area_height:NULL, area_width:NULL, style = "python"] * 
            tf$cast((area_width + 1), tf$int32)
          
          height_list[[i]] <- tf$reshape(h, list(batch, -1L))
          width_list[[i]]  <- tf$reshape(w, list(batch, -1L))

          i <- i + 1L
        }
      }

      pool_results <- tf$concat(pool_list, axis = 1L)
      area_heights <-
        tf$expand_dims(tf$concat(height_list, axis = 1L), 2L)
      area_widths  <-
        tf$expand_dims(tf$concat(width_list, axis = 1L), 2L)
      
    c(pool = pool_results, heights = area_heights, widths = area_widths)
  }


#' Compute area sums for features
#' @param features: a Tensor in a shape of [batch_size, height * width, depth].
#' @param max_area_width the max width allowed for an area.
#' @param max_area_height the max height allowed for an area. (default for 1D case)
#' @param height the height of the image. (default for 1D case)
#' @param name the namescope.
#' @return sum_image
#' @return area_heights
#' @return area_widths
.compute_sum_image <- 
  function(features = tf$random$normal(shape = list(8L, 512L, 32L)), 
           max_area_width, max_area_height = 1L, .height = 1L) {
    features_shape  <- shape_list2(features)
    batch           <- features_shape[[1]]
    length          <- features_shape[[length(features_shape)-1L]]
    depth           <- features_shape[[length(features_shape)]]
    .width          <- length %/% .height
    
    c(.width, .height) %<-% 
      validate_area_parameters(.width, .height, features_shape)

    features_2d <- tf$reshape(features, list(batch, .height, .width, depth))
    
    width_cum <- 
      tf$cumsum(features_2d, axis = -2L, name = "compute_integral_h")
    
    integral_image <- 
      tf$cumsum(width_cum, axis = -3L, name = "compute_integral_v")
    
    padded_image <- 
      tf$pad(integral_image, list(c(0L, 0L),
                                  c(1L, 0L),
                                  c(1L, 0L),
                                  c(0L, 0L)), constant_values = 0L)
    
    length.out      <- max_area_width * max_area_height
    height_list     <- vector("list", length.out)
    width_list      <- vector("list", length.out)
    dst_images      <- vector("list", length.out)
    src_images_diag <- vector("list", length.out)
    src_images_h    <- vector("list", length.out)
    src_images_v    <- vector("list", length.out)
    
    image_shape <- shape_list2(padded_image)
    size_tensor <- tf$ones(shape = image_shape[1:length(image_shape)-1],
                           dtype = tf$int32)
    i <- 1L
    for (height in seq(0L, max_area_height-1L)) { 
      for (width in seq(0L, max_area_width-1L)) {
        
        dst_images[[i]] <-
          padded_image[, `(height + 1):`, `(width + 1):`, , style="python"] %>%
          tf$reshape(list(batch, -1L, depth))

        src_images_diag[[i]] <-
          padded_image[, `:-height - 1`, `:-width - 1`, , style="python"] %>%
          tf$reshape(list(batch, -1L, depth))

        src_images_h[[i]] <-
          padded_image[, `(height + 1):`, `:-width - 1`, , style = "python"] %>%
          tf$reshape(list(batch, -1L, depth))
        
        src_images_v[[i]] <-
          padded_image[, `:-height - 1`, `width + 1:`, , style = "python"] %>%
          tf$reshape(list(batch, -1L, depth))

        height_list[[i]] <-
          tf$reshape(size_tensor[, `height + 1:`, `width + 1:`, style = "python"] *
                       (height + 1L),
                     list(batch, -1L))
        width_list[[i]] <-
          tf$reshape(size_tensor[, `height + 1:`, `width + 1:`, style = "python"] *
                       (height + 1L),
                     list(batch, -1L))
        
        # print(paste("dst:     ", dst_images[[i]]))
        # print(paste("src_diag:", src_images_diag[[i]]))
        # print(paste("src_v:   ", src_images_v[[i]]))
        # print(paste("src_h:   ", src_images_h[[i]]))
        # print("")

        i <- i + 1L
      }
    }
    
    sum_image <- tf$subtract(
      tf$concat(dst_images, axis = 1L) + tf$concat(src_images_diag, axis = 1L),
      tf$concat(src_images_v, axis = 1L) + tf$concat(src_images_h, axis = 1L))
    
    area_heights <- tf$expand_dims(tf$concat(height_list, axis = 1L), 2L)
    area_widths  <- tf$expand_dims(tf$concat(width_list, axis = 1L), 2L)
    
    c(sum = sum_image, heights = area_heights, widths = area_widths)
  }



#' Computes features for each area.
#' @return area_mean: A Tensor of shape [batch_size, num_areas, depth]
#' @return area_std: A Tensor of shape [batch_size, num_areas, depth]
#' @return area_sum: A Tensor of shape [batch_size, num_areas, depth]
#' @return area_heights: A Tensor of shape [batch_size, num_areas, 1]
#' @return area_widths: A Tensor of shape [batch_size, num_areas, 1]
compute_area_features <-
  function(features,
           max_area_width = NULL,
           max_area_height = NULL,
           epsilon = 1e-6) {
    c(area_sum, area_heights, area_widths) %<-% 
      .compute_sum_image(features, max_area_width, max_area_height)

    c(area_sq_sum, unused1, unused2) %<-% 
      .compute_sum_image(tf$pow(features, 2L), 
                         max_area_width, 
                         max_area_height)
    
    sizes <- 
      tf$multiply(area_heights, area_widths) %>% 
      tf$cast(dtype = tf$float32)
    
    area_mean     <- tf$math$divide(area_sum, sizes)
    sq_area_mean  <- tf$math$divide(area_sq_sum, sizes)
    area_variance <- tf$subtract(sq_area_mean, tf$pow(area_mean, 2L))
    area_std      <- tf$sqrt(tf$abs(area_variance) + epsilon)
    
    c(mean = area_mean, stddev = area_std, sum = area_sum, 
      heights = area_heights, widths = area_widths)
  }



#' Computes the key for each area.
#' 
#' @param features a Tensor in a shape of [batch_size, height * width, depth].
#' @param max_area_width: the max width allowed for an area.
#' @param max_area_height: the max height allowed for an area.
#' @param height: the height of the image.
#' @param mode: whether to combine different area features or only use
#' the vector mean of each area, which can be "mean", "concat", "sum",
#' "sample_concat", and "sample_sum".
#' @return Tensor of shape [batch, num_areas, depth]
compute_area_key <-
  function(features,
           max_area_width,
           max_area_height = 1L,
           height = 1L,
           mode = "sample_concat",
           hidden_activation = "relu",
           training = TRUE,
           name = NULL) {
    
    # TODO: Error in modes: c("concat", "max_concat")
    
    stopifnot(mode %in% c("mean", "max", "concat", 
                          "sum", "sample", "sample_concat", 
                          "sample_sum", "max_concat"))
    
    c(area_mean, area_std, unused, area_heights, area_widths) %<-% 
      compute_area_features(features, max_area_width, max_area_height, height)
    
    if (mode == "mean") 
      return(area_mean)
    else if (mode == "max") {
      c(area_max, unused, unused2) %<-% 
        basic_pool(features, max_area_width, max_area_height, height)
      return(area_max)
    }
    else if (mode == "sample") {
      if (training)
        area_mean <- 
          area_mean + (area_std * tf$random_normal(tf$shape(area_std)))
      return(area_mean)
    }
    
    depth <- tail(shape_list2(area_mean), 1)[[1]]

    height_embed <- tf$nn$embedding_lookup(
      params = tf$Variable(
        tf$zeros(shape = list(max_area_height, depth %/% 2)), 
        name = "area_height_emb"),
      ids = area_heights[, , 0, style = "python"] - 1L
    )
    
    width_embed <- tf$nn$embedding_lookup(
      params = tf$Variable(
        tf$zeros(shape = list(max_area_width, depth %/% 2)), 
        name = "area_width_emb"),
      ids = area_heights[, , 0, style = "python"] - 1L
    )
    
    size_embed <- tf$concat(list(height_embed, width_embed), -1L)
    
    if (mode == "concat") 
      feature_concat <- tf$concat(list(area_mean, area_std, size_embed), -1L)
    else if (mode == "max_concat") {
      area_max <-
        basic_pool(features, max_area_width, max_area_height, height)[[1]]
      feature_concat <- tf$concat(list(area_max, size_embed), -1L)
    }
    else if (mode == "sum") 
      feature_concat <- size_embed + area_mean + area_std
    else if (mode == "sample_concat") {
      if (training)
        area_mean <- 
          area_mean + (area_std * tf$random$normal(tf$shape(area_std)))
      feature_concat <- area_mean + size_embed
    }
    else if (mode == "sample_sum") {
      if (training)
        area_mean <- area_mean * (area_std * tf$random$normal(tf$shape(area_std)))
      feature_concat <- area_mean + size_embed
    }
    else
      stop(sprintf("Unsupported area key mode %s", mode))
    
    feature_hidden <- 
      layer_dense(feature_concat, depth, activation = hidden_activation)
    
    # Shape issue with calling keras_layer vs tf dense layer?
    if (mode %in% c("concat", "max_concat"))
      area_key <- tf$layers$dense(feature_hidden, depth)
    else
      area_key <- layer_dense(feature_hidden, depth)
    
    area_key
  }

modes <- c("mean", "max", "concat", 
           "sum", "sample", "sample_concat", 
           "sample_sum", "max_concat")

#' Dot product area attention
#' 
#' @param q Tensor with shape [..., length_q, depth_k].
#' @param k Tensor with shape [..., length_kv, depth_k]. 
#'   Leading dimensions must match with q.
#' @param v: Tensor with shape [..., length_kv, depth_v] 
#'   Leading dimensions must match with q.
#'   
#' Returns Tensor with shape [..., length_q, depth_v].
dot_product_area_attention_1d <- function(q,
                                          k,
                                          v,
                                          bias = NULL,
                                          dropout = 0, 
                                          max_area_width = 1L,
                                          max_area_height = 1L,
                                          area_height = 1L,
                                          area_key_mode = "mean",
                                          area_value_mode = "sum",
                                          top_k_areas = 0L,
                                          name = NULL,
                                          training = TRUE) {
  
  mem_shape <- shape_list2(k)
  batch     <- mem_shape[[1]]
  head_size <- mem_shape[[2]]
  length    <- mem_shape[[3]]
  depth     <- mem_shape[[4]]

  k_area <- compute_area_key(
    tf$reshape(k, list(-1L, length, depth)),
    max_area_width  = max_area_width,
    max_area_height = max_area_height,
    mode            = area_key_mode,
    training        = training)
  
  if (area_value_mode == "mean")
    v_area <- compute_area_features(tf$reshape(v, list(-1L, length, depth)),
                                    max_area_width, max_area_height)[[1]]
    # c(v_area, unused, unused2, unused3, unused4) %<-% basic_pool(
  else if (area_value_mode == "max")
    v_area <- basic_pool(tf$reshape(v, list(-1L, length, depth)),
                         max_area_width,
                         max_area_height,
                         height,
                         fn = tf$reduce_max)[[1]]
    # c(v_area, unused, unused2, unused3, unused4) %<-% 
  else if (area_value_mode == "sum") {
    v_area <- compute_area_features(tf$reshape(v, list(-1L, length, depth)),
                                    max_area_width,
                                    max_area_height)[[3]]
    # c(unused, unused2, v_area, unused3, unused4) %<-% 
  }
  else stop(paste("Unsuported area value mode", mode))
  
  k <- tf$reshape(k_area, list(batch, head_size, -1L, depth))
  v <- tf$reshape(v_area, list(batch, head_size, -1L, depth))
  
  logits <- tf$matmul(q, k, transpose_b = TRUE)
  
  if (!is.null(bias)) {
    bias <- cast_like(bias, logits)
    bias_shape <- shape_list2(bias)
    mem_length <- bias_shape[[length(bias_shape)]]
    bias_value <- tf$reshape(
      tf$to_float(tf$less(bias, -1L)), list(-1L, mem_length, -1L))
    
    c(unused, unuse2, padding_sum, unused3, unused4) %<-% 
      compute_area_features(bias_value, max_area_width, max_area_height)
    
    bias <- tf$where(
      tf$cast(tf$to_int32(padding_sum), tf$bool),
      tf$fill(tf$shape(padding_sum), -Inf),
      tf$zeros_like(padding_sum, dtype = tf$float32))
    
    bias <- 
      tf$reshape(bias,
                 list(bias_shape[[1]], bias_shape[[2]], bias_shape[[3]], -1L))
    
    logits <- logits + bias
  }
  
  weights <- tf$nn$softmax(logits, name = "attention_weights")
  
  if (top_k_areas > 0) {
    tf$logging$info("area_attention top_k_areas=%d", top_k_areas)
    
    top_k <- tf$minimum(tail(shape_list2(weights), 1), top_k_areas)
    c(top_weights, unused) %<-% tf$nn$top_k(weights, k = top_k)
    
    min_values <- tf$reduce_min(top_weights, -1L, keepdims = TRUE)
    weights    <- tf$where(tf$greater_equal(weights, min_values),
                           weights, tf$zeros_like(weights))
    weights    <- tf$div(weights, tf$reduce_sum(weights, -1L, keepdims = TRUE))
  }
  
  weights <- layer_dropout(weights, dropout)
  
  tf$matmul(weights, v)
}



layer_multihead_attention <- function(query,
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
                                      vars_3d = TRUE) {
  
  layer_lambda(list(query, memory), function(x) {
    
    stopifnot(key_depth %% num_heads == 0, value_depth %% num_heads == 0)
    
    if (typeof(x) == "list" & length(x) > 1) # if (any(grepl("list", class(x)))) 
      c(query, memory) %<-% x
    else 
      query <- x
    
    vars_3d_num_heads <- if (vars_3d) num_heads else 0

    c(q, k, v) %<-% compute_qkv(query = query, 
                                memory = memory, 
                                key_depth = key_depth, 
                                value_depth = value_depth, 
                                q_filter_width = q_filter_width, 
                                vars_3d_num_heads = vars_3d_num_heads)

    q <- split_heads(q, num_heads)
    k <- split_heads(k, num_heads)
    v <- split_heads(v, num_heads)
    
    key_depth_per_head <- key_depth %/% num_heads
    
    if (!vars_3d) 
      q %<>% `*`(key_depth_per_head^(-0.5))
    
    if (attention_type == "dot_product")
      if (max_area_width > 1 || max_area_height > 1)
        x <- dot_product_area_attention_1d(
          q,
          k,
          v,
          bias,
          dropout,
          max_area_width = max_area_width,
          max_area_height = max_area_height,
          area_height = area_height,
          area_key_mode = area_key_mode,
          area_value_mode = area_value_mode
        )
      else
        x <- dot_product_attention_1d(q, k, v, bias, dropout)
    else
      stop(paste("attention_types other than (dot_product, dot_product_area)",
                 "currently unimplemented..."))
    
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
        
        # tf$compat$v1$get_variable(
        #   name  = "output_kernel_3d",
        #   shape = list(num_heads,
        #                as.integer(value_depth %/% num_heads),
        #                output_depth),
        #   initializer = tf$glorot_normal_initializer
        # ) %>% 
        # tf$cast(x$dtype) %>% 
        # tf$reshape(list(value_depth, output_depth)) %>% 
        # {tf$tensordot(x, ., axes = 1L)}
      else  
        layer_dense(x, output_depth, use_bias = FALSE, name = "output_kernel")
        # tf$matmul(x,
        #           tf$get_variable(
        #             name  = "output_kernel",
        #             shape = list(x_shape[[length(x_shape)]], output_depth),
        #             dtype = x$dtype,
        #             trainable = TRUE
        #           ))
    
    x
    
  }, name = "multihead_attention")
}




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
#' TODO: hard_attention_k: integer, if > 0 triggers hard attention (pick top-k)
#' @export
#' TODO: Add callable option to attention_type
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
  # x_shape <- shape_list2(x)

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

      # tf$compat$v1$get_variable(
      #   name  = "output_kernel_3d",
      #   shape = list(num_heads,
      #                as.integer(value_depth %/% num_heads),
      #                output_depth),
      #   initializer = tf$glorot_normal_initializer
      # ) %>% 
      # tf$cast(x$dtype) %>% 
      # tf$reshape(list(value_depth, output_depth)) %>% 
      # {tf$tensordot(x, ., axes = 1L)}
    else
      layer_dense(x, output_depth, use_bias = FALSE, name = "output_transform")
  
  x
}



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



#' @export
layer_dot_product_attention_1d <-
  function(q,
           k,
           v,
           bias = NULL,
           dropout = 0,
           name = "dot_product_attention") {
    layer_lambda(c(q, k, v), function(x) {
      c(q, k, v) %<-% x
      
      q_shape <- shape_list2(q)
      
      scalar <-
        tf$math$rsqrt(tf$cast(q_shape[[length(q_shape)]], tf$float32))
      logits <- tf$matmul(q * scalar, k, transpose_b = TRUE)
      
      if (!is.null(bias))
        logits <- logits + bias
      
      weights <- tf$nn$softmax(logits, name = "attention_weights")
      
      x <- tf$matmul(weights, v)
      
      x
    }, name = name)
  }


# TODO: make lambda R6 layer
# Expecting shape(x) == (batch, maxtime, units)
# Ref: https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/
# layers/common_attention.py#L5020
self_attention_simple <-
  function(x,
           filter_depth = 32L,
           output_depth = 64L,
           num_parts = 3L,
           dropout = 0,
           share_kv = TRUE) {
  x_shape <- shape_list2(x)
      
  c(q, k, v) %<-% create_qkv(x, filter_depth, num_parts, share_kv)

  bias <- NULL
  x <- dot_product_attention_1d(q, k, v, bias, dropout)
  x <- tf$reshape(x, list(x_shape[[1]], x_shape[[2]], filter_depth))
  x <- layer_dense(x, 
                   units = output_depth, 
                   use_bias = FALSE, 
                   name = "output_transform")
  
  x
}


#' @export
layer_self_attention_simple <- 
  function(x,
           filter_depth = 32L,
           output_depth = 64L,
           num_parts = 3L,
           dropout = 0,
           share_kv = TRUE) {
    layer_lambda(x, function(x) {
      x_shape <- shape_list2(x)
      
      c(q, k, v) %<-% create_qkv(x, filter_depth, num_parts, share_kv)
      
      bias <- NULL
      x <- dot_product_attention_1d(q, k, v, bias, dropout)
      x <- tf$reshape(x, list(x_shape[[1]], x_shape[[2]], filter_depth))
      x <- layer_dense(x, output_depth, use_bias = FALSE, name = "output_layer")
      
      x
    }, name = "self_attention_simple")
  }



#' antecedent: Tensor with shape [batch, length, channels]
#' depth: specifying projection layer depth
#' filter_width: how wide should the attention component be
#' padding: must be in: c("valid", "same", "left")
.compute_attention_component <- function(antecedent,
                                         depth,
                                         filter_width = 1L,
                                         padding = 'same',
                                         name = 'c',
                                         vars_3d_num_heads = 0L) {
  layer_lambda(x, function(x) {

    if (vars_3d_num_heads > 0) {
      stopifnot(filter_width == 1)
      
      input_shape <- shape_list2(antecedent)
      input_depth <- input_shape[[length(input_shape)]]
      depth_per_head <- depth %/% vars_3d_num_heads
      stddev <- input_depth^(-0.5)
      
      if ("q" %in% name) stddev %<>% `*`(depth_per_head^(-0.5))
      
      # TODO: Add tf$variable_scope?
      var <- tf$Variable(
        tf$random$normal(
          shape = list(
            input_depth,
            vars_3d_num_heads,
            as.integer(depth %/% vars_3d_num_heads)
          ),
          stddev = stddev,
          dtype = antecedent$dtype,
          name = name
        ),
        name = name,
        trainable = TRUE
      )
      
      # var <- tf$compat$v1$get_variable(
      #   name,
      #   shape = list(
      #     input_depth,
      #     vars_3d_num_heads,
      #     as.integer(depth %/% vars_3d_num_heads)
      #   ),
      #   
      #   initializer = tf$random_normal_initializer(stddev = stddev))
      
      var %<>% 
        tf$cast(dtype = antecedent$dtype) %>% 
        tf$reshape(shape = list(input_depth, depth))
      
      return(tf$tensordot(antecedent, var, axes = 1L))
    }
    
    out <- 
      if (filter_width == 1L) 
        layer_dense(antecedent, depth, use_bias = FALSE, name = name)
    else
      layer_conv_1d(antecedent, depth, filter_width, 
                    padding = padding, name = name)
    
    out
  }, name = "compute_attention_component")
}



#' @export
layer_compute_qkv <- function(query,
                              memory = NULL,
                              key_depth = 64L,
                              value_depth = 64L,
                              q_filter_width = 1L,
                              kv_filter_width = 1L,
                              q_padding = 'same',
                              kv_padding = 'same',
                              vars_3d_num_heads = 0L) {
  
  x <- if(!is.null(memory)) c(query, memory) else query
  
  layer_lambda(x, function(x) {
    
    if (typeof(x) == "list")
      c(query, memory) %<-% x
    else 
      query <- x
    
    if (is.null(memory))
      memory <- query
    q <- compute_attention_component(query,
                                      key_depth,
                                      q_filter_width,
                                      q_padding,
                                      name = "q",
                                      vars_3d_num_heads = vars_3d_num_heads)
    
    k <- compute_attention_component(memory,
                                      key_depth,
                                      kv_filter_width,
                                      kv_padding,
                                      name = "k",
                                      vars_3d_num_heads = vars_3d_num_heads)
    
    v <- compute_attention_component(memory,
                                      key_depth,
                                      kv_filter_width,
                                      kv_padding,
                                      name = "v",
                                      vars_3d_num_heads = vars_3d_num_heads)
    
    c(q, k, v)
    
  }, name = "layer_compute_qkv")
}




#' @export
layer_create_qkv <- 
  function(x, filter_depth, num_parts = 1L, share_kv = FALSE) {
    layer_lambda(x, function(x) {
      x_shape    <- shape_list2(x)
      part_depth <- as.integer(floor(filter_depth / num_parts))
      
      if (!share_kv) {
        combined <- layer_dense(
          x, filter_depth * 3L, use_bias = FALSE, name = "qkv_transform")
        
        c(q, k, v) %<-% tf$split(combined, 3L, axis = 2L)
      }
      else {
        q <- layer_dense(
          x, filter_depth, use_bias = FALSE, name = "q_transform")
        
        kv_combined <-
          layer_dense(
            tf$concat(list(x, x), axis = 1L),
            filter_depth,
            use_bias = FALSE,
            name = "kv_transform")
        
        c(k, v) %<-% 
          tf$split(kv_combined, list(x_shape[[2]], x_shape[[2]]), axis = 1L)
      }
      
      q <- q * tf$pow(tf$cast(part_depth, tf$float32), tf$constant(-0.5))
      
      c(q, k, v)
      
    }, name = "create_qkv")
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
layer_local_attention_1d <- function(q,
                                     k,
                                     v,
                                     block_length = 1024L,
                                     filter_width = 100L,
                                     name = "local_attention_1d") {
  layer_lambda(x, function(x) {
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
      dot_product_attention_1d(
        q, k, v, attention_bias, name = "local_1d") %>% 
      tf$reshape(list(batch, num_heads, original_length, depth_v))
    
    # Remove the padding if introduced.
    output <- tf$slice(output, 
                       list(0L, 0L, 0L, 0L),
                       list(-1L, -1L, original_length, -1L))
    
    output$set_shape(list(batch, num_heads, original_length, depth_v))
    
    output
  }, name = name)
}



# TODO: Convert above layer lambda to R6 class or custom keras model
# Via tensor2tensor framework
# Strided block local self-attention.
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/
# common_attention.py#L3118
LocalSelfAttentionTF <- R6::R6Class(
  "LocalSelfAttentionTF",
  
  inherit = KerasLayer,
  
  public = list(
    initialize = function() {},
    
    build = function() {},
    
    call = function(x, mask = NULL) {
      # Score by attention type
      
      # Pass through activation to get alignments
      
      # 
    },
    
    compute_output_shape = function() {}
    
  )
)


layer_local_self_attentionTF <-
  function(object,
           units = 32L,
           attention_width = 3L,
           attention_type = "additive",
           return_attention = FALSE,
           mask = FALSE,
           kernel_initializer = 'glorot_normal',
           bias_initializer = 'zeros') {
    
  create_layer(LocalSelfAttentionTF,
               object, 
               list(units = as.integer(units),
                    attention_width = as.integer(attention_width),
                    attention_type = attention_type,
                    return_attention = return_attention,
                    mask = mask,
                    kernel_initializer = tf$keras$initializers$get(kernel_initializer), 
                    bias_initializer = tf$keras$initializer$get(bias_initializer)
               )
  )
}





LocalSelfAttention <- R6::R6Class(
  "LocalSelfAttention",
  
  inherit = KerasLayer,
  
  public = list(
    
    units = NULL,
    attention_width = NULL,
    attention_type = NULL,
    use_attention_bias = NULL,
    kernel_initializer = NULL,
    kernel_regularizer = NULL,
    bias_initializer = NULL,
    bias_regularizer = NULL,
    Wt = NULL,
    Wx = NULL,
    Wa = NULL,
    bh = NULL,
    ba = NULL,
    
    initialize = function(units,
                          attention_width,
                          attention_type,
                          use_attention_bias,
                          kernel_initializer,
                          kernel_regularizer,
                          bias_initializer,
                          bias_regularizer) {
      self$units              <- units
      self$attention_width    <- attention_width
      self$attention_type     <- attention_type
      self$use_attention_bias <- use_attention_bias
      self$kernel_initializer <- kernel_initializer
      self$kernel_regularizer <- kernel_regularizer
      self$bias_initializer   <- bias_initializer
      self$bias_regularizer   <- bias_regularizer
    },
    
    build_additive_attention = function(channels) {
      self$Wt <- self$add_weight(
        shape = list(channels, self$units),
        initializer = self$kernel_initializer,
        regularizer = self$kernel_regularizer,
        name = "Wt"
      )
      
      self$Wx <- self$add_weight(
        shape = list(channels, self$units),
        initializer = self$kernel_initializer,
        name = "Wx"
      )
      
      self$Wa <- self$add_weight(
        shape = list(self$units, 1L),
        initializer = self$kernel_initializer,
        name = "Wa"
      )
      
      if (self$use_attention_bias) {
        self$bh <- self$add_weight(
          shape = list(self$units),
          initializer = self$bias_initializer,
          name = "bh"
        )
      
        self$ba <- self$add_weight(
          shape = list(1L),
          initializer = self$bias_initializer,
          name = "ba"
        )
      }
    },
    
    build_multiplicative_attention = function(channels) {
      self$Wa <- self$add_weight(
        shape = list(channels, channels),
        initializer = self$kernel_initializer,
        name = "Wa"
      )
      
      if (self$use_attention_bias)      
        self$ba <- self$add_weight(
          shape = list(1L),
          initializer = self$bias_initializer,
          name = "ba"
        )
    },
    
    build = function(input_shape) {
      channels <- input_shape[[length(input_shape)]]
      
      if (!self$attention_type %in% c("additive", "multiplicitive"))
        stop("attention_type must be one of: 'additive', 'multiplicative'")
      
      if (self$attention_type == "additive")
        self$build_additive_attention(channels)
      else
        self$build_multiplicative_attention(channels)
      
    },
    
    call = function(x, mask = NULL) {
      seqlen <- shape_list2(x)[[2]]
      
      score <- 
        if (self$attention_type == "additive")
          self$additive_score(x)
        else 
          self$multiplicative_score(x)
      
      # Localize
      lower <- 
        tf$range(0L, seqlen) - as.integer(self$attention_width / 2L) %>% 
        tf$expand_dims(-1L)
      
      upper <- lower + self$attention_width
      indices <- tf$expand_dims(tf$range(0L, seqlen), axis = 0L)

      # Mask out anything wider than attention_width and apply scores
      emission <- 
        score * 
        tf$cast(lower <= indices, tf$float32) * 
        tf$cast(upper > indices, tf$float32) 
  
      sum <- tf$keras$backend$sum(emission, axis = -1L, keepdims = TRUE)
      
      attention <- emission / (sum + tf$keras$backend$epsilon())

      v <- tf$matmul(attention, x)
      
      v
    },
    
    additive_score = function(x) {
      shape  <- shape_list2(x)
      batch  <- shape[[1]]
      seqlen <- shape[[2]]
      
      q <- tf$expand_dims(tf$matmul(x, self$Wt), 2L)
      k <- tf$expand_dims(tf$matmul(x, self$Wx), 1L)
      
      h <- tf$tanh(q + k + if (!is.null(self$bh)) self$bh else tf$constant(0L))

      e <- tf$reshape(tf$matmul(h, self$Wa) + 
                        if (!is.null(self$ba)) self$ba
                        else tf$constant(0L), 
                      list(batch, seqlen, seqlen))
      e
    },
    
    multiplicative_score = function(x) {

      score <- tf$keras$backend$batch_dot(
        tf$matmul(x, self$Wa),
        tf$transpose(x, perm = list(0L, 2L, 1L))
      )
      
      if (!is.null(self$ba))
        score <- score + self$ba
      
      score
    },
    
    compute_output_shape = function(input_shape) {
      output_shape <- input_shape
      if (self$return_attention)
        output_shape <- list(output_shape,
                             list(input_shape[[1]],
                                  output_shape[[2]],
                                  input_shape[[2]]))
      output_shape
    }
    
  )
)


layer_local_self_attention <- function(object,
                                       units,
                                       attention_width,
                                       attention_type = "additive",
                                       use_attention_bias = TRUE,
                                       kernel_initializer = 'glorot_uniform',
                                       kernel_regularizer = NULL,
                                       bias_initializer = 'zeors',
                                       bias_regularizer = NULL) {
  create_layer(
    LocalSelfAttention,
    object,
    list(
      units = as.integer(units),
      attention_width = as.integer(attention_width),
      attention_type = attention_type,
      use_attention_bias = use_attention_bias,
      kernel_initializer = tf$keras$initializers$get(kernel_initializer),
      kernel_regularizer = tf$keras$regularizers$get(kernel_regularizer),
      bias_initializer = tf$keras$initializers$get(bias_initializer),
      bias_regularizer = tf$keras$initializers$get(bias_initializer)
    )
  )
}



