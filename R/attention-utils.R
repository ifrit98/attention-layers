
#' Maybe set max_area_height/width parameters if invalid.
validate_area_parameters <- function(width, height, features_shape) {
  pairs <- calculate_divisor_pairs(features_shape[[2]])
  
  if (is.null(width) || is.null(height)) {
    n <- length(pairs) %/% 2
    print(sprintf(
      "Setting (width, height) to (%s, %s)",
      pairs[[n]][[1]],
      pairs[[n]][[2]]
    ))
    height <- pairs[[n]][[1]]
    width  <- pairs[[n]][[2]]
  }
  
  if(!list(c(width, height)) %in% pairs) {
    stop(paste(
      "(width, height) must be a pair that",
      "divides (length) evenly, got: "), paste0("(", width, ", ", height, ")"),
      "\nPlease select (height) from one of the following values:\n",
      list(unlist(pairs)[seq(length(pairs) * 2, by = -2)]))
  }
  
  c(as.integer(width), as.integer(height))
}


#' Calculates the padding mask based on which embeddings are all zero.
#' 
#' emb Tensor with shape [..., depth]
#' 
#' Returns:
#'   a float Tensor with shape [...]. Each element is 1 if its 
#'   corresponding embedding vector is all zero, and is 0 otherwise.
embedding_to_padding <- function(emb) {
  emb_sum <- tf$reduce_sum(tf$abs(emb), axis = -1L)
  tf$to_float(tf$equal(emb_sum, 0))
}


#' Reshape input by splitting length over blocks of memory_block_size.
#'
#' x Tensor [batch, heads, length, depth]
#' x_shape tf$TensorShape of x
#' memory_block_size Integer to dividing length by
#' Return 
#'   Tensor [batch, heads, length %/% memory_block_size, memory_block_size, depth]
reshape_by_blocks <- function(x, x_shape, memory_block_size) {
  x <- tf$reshape(x,
                  list(x_shape[[1]], x_shape[[2]], 
                       as.integer(x_shape[[3]] %/% memory_block_size), 
                       memory_block_size, x_shape[[4]]))
  x
}



#' Reshape x so that the last dimension becomes two dimensions.
split_last_dimension <- function(x, n) {
  x_shape <- shape_list2(x)
  
  n <- as.integer(n)
  m <- x_shape[[length(x_shape)]]
  
  stopifnot(m %% n == 0)
  
  out <- 
    tf$reshape(x, c(x_shape[-length(x_shape)], list(n, as.integer(m %/% n))))
  
  out
}



#' Split channels (dimension 2) into multiple heads (becomes dimension 1).
#' x Tensor shape: [batch, length, channels]
#' num_heads integer
split_heads <- function(x, num_heads) {
  out <- tf$transpose(split_last_dimension(x, num_heads), 
                      perm = list(0L, 2L, 1L, 3L))
  out
}



#' Reshape x so that the last two dimension become one.
combine_last_two_dimensions <- function(x) {
  x_shape <- shape_list2(x)
  c(a, b) %<-% x_shape[-c(1:(length(x_shape)-2))]
  
  tf$reshape(x, c(x_shape[c(1,2)], as.integer(a * b)))
}



#' Inverse of split_heads.
combine_heads <- function(x) {
  combine_last_two_dimensions(tf$transpose(x, list(0L, 2L, 1L, 3L)))  
}




# TODO: make this an R6 layer?
#' Takes input tensor of shape [batch, seqlen, channels] and
#' creates query, key, and value tensors to pass to attention
#' mechanisms downstream.
#' 
#' query shape [batch, seqlen, filter_depth]
#' key shape   [batch, seqlen, filter_depth]
#' value shape [batch, seqlen, filter_depth]
#' @export
create_qkv <- function(x, filter_depth, num_parts = 1L, share_kv = FALSE) {
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
}



#' query  [batch, length_q, channels]
#' memory [batch, length_m, channels] (optional, usually RNN hidden states)
#' return [batch, length_q, *_depth] (q, k ,v) tensors
compute_qkv <-
  function(query,
           memory = NULL,
           key_depth = 64L,
           value_depth = 64L,
           q_filter_width = 1L,
           kv_filter_width = 1L,
           q_padding = 'same',
           kv_padding = 'same',
           vars_3d_num_heads = 0L) {
    
    if (is.null(memory))
      memory <- query
    q <- compute_attention_component(query,
                                     key_depth,
                                     q_filter_width,
                                     q_padding,
                                     "q",
                                     vars_3d_num_heads)
    
    k <- compute_attention_component(memory,
                                     key_depth,
                                     kv_filter_width,
                                     kv_padding,
                                     "k",
                                     vars_3d_num_heads)
    
    v <- compute_attention_component(memory,
                                     value_depth,
                                     kv_filter_width,
                                     kv_padding,
                                     "v",
                                     vars_3d_num_heads)
    
    c(q, k, v)
  }



#' antecedent: Tensor with shape [batch, length, channels]
#' depth: specifying projection layer depth
#' filter_width: how wide should the attention component be
#' padding: must be in: c("valid", "same", "left")
compute_attention_component <- function(antecedent,
                                        depth,
                                        filter_width = 1L,
                                        padding = 'same',
                                        name = 'c',
                                        vars_3d_num_heads = 0L) {
  if (vars_3d_num_heads > 0) {
    stopifnot(filter_width == 1)
    
    input_shape    <- shape_list2(antecedent)
    input_depth    <- input_shape[[length(input_shape)]]
    stddev         <- input_depth ^ (-0.5)
    depth_per_head <- depth %/% vars_3d_num_heads
    
    if ("q" %in% name)
      stddev %<>% `*`(depth_per_head ^ (-0.5))
    
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
      name = name
    )
    
    # var <- tf$compat$v1$get_variable(
    #   name = name,
    #   shape = list(
    #     input_depth,
    #     vars_3d_num_heads,
    #     as.integer(depth %/% vars_3d_num_heads)
    #   ),
    #   initializer = tf$random_normal_initializer(stddev = stddev),
    #   dtype = antecedent$dtype
    # )
    
    var <- var %>% tf$reshape(shape = list(input_depth, depth))
    
    return(tf$tensordot(antecedent, var, axes = 1L))
  }
  
  out <- 
    if (filter_width == 1L) 
      layer_dense(antecedent, depth, use_bias = FALSE, name = name)
  else
    layer_conv_1d(antecedent, depth, filter_width, padding = padding, name = name)
  
  out
}


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
    
    images <- vector("list", area_height * area_width)
    
    i <- 1L
    for (y_shift in seq(0L, area_height-1L)) {
      img_height <- tf$maximum(height - area_height + 1L + y_shift, 0L)
      for (x_shift in seq(0L, area_width-1L)) {
        img_width <- tf$maximum(width - area_width + 1L + x_shift, 0L)
        area <- 
          features_2d[ , y_shift:img_height, x_shift:img_width, , style = "python"]
        flatten_area <- tf$reshape(area, list(batch, -1L, depth, 1L))
        images[[i]]  <- flatten_area
        
        i <- i + 1L
      }
    }
    
    img_tensor <- tf$concat(images, axis = 3L)
    max_tensor <- fn(img_tensor, axis = 3L)

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
  function(features,
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
  function(features, 
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
    
    stopifnot(mode %in% c("mean", "max", "concat", 
                          "sum", "sample", "sample_concat", 
                          "sample_sum", "max_concat"))
    if (mode %in% c("concat", "max_concat"))
      warning(sprintf("Mode '%s' uses tf$layers$dense and is deprecated", mode))
    
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
          area_mean + (area_std * tf$random$normal(tf$shape(area_std)))
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
