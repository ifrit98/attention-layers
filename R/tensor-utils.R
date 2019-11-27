

#' Grabs list of tensor dims statically, where possible.
#' @export
shape_list <- 
  function(x) {
    
    x <- tf$convert_to_tensor(x)
    
    dims <- x$get_shape()$dims
    if (is.null(dims)) return(tf$shape(x))
    
    sess <- tf$keras$backend$get_session()
    
    shape <- tf$shape(x)$eval(session = sess)
    
    ret <- vector('list', length(dims))
    
    purrr::map2(dims, shape, function(x, y) {
      dim <- x
      
      if (is.null(dim)) 
        dim <- y

      dim
      
    })
  }


#' Can we cheat and call value on Dimension 
#' class object without getting into trouble?
#' @export
shape_list2 <- 
  function(x) {
    
    x <- tf$convert_to_tensor(x)
    
    dims <- x$get_shape()$dims    
    if (is.null(dims)) return(tf$shape(x))
    
    dims <- purrr::map(dims, ~.$value)
    
    sess <- tf$keras$backend$get_session()
    shape <- tf$shape(x)$eval(session = sess)
    
    ret <- vector('list', length(dims))
    
    purrr::map2(dims, shape, function(x, y) {
      dim <- x
      
      if (is.null(dim)) 
        dim <- y
      
      dim
      
    })
  }


#' Cast x to y's detype if necessary
#' @export
cast_like <- function(x, y) {
  x <- tf$convert_to_tensor(x)
  y <- tf$convert_to_tensor(y)
  
  if (x$dtype == y$dtype) return(x)
  
  cast_x <- tf$cast(x, y$dtype)

  if (cast_x$device != x$device) {
    x_name <- "(eager Tensor)"
    
    try(x_name <- x$name, silent = TRUE)
  
    tf$logging$warning("Cast for %s may induce copy from '%s' to '%s'",
                       x_name,
                       x$device,
                       cast_x$device)
  }
  cast_x
}



#' Creates a sparse eye_matrix tensor
#' @export
sparse_eye <- function(size, axis = 1L) {
  size <- as.integer(size)
  
  indices <-
    tf$cast(tf$stack(list(tf$range(size), tf$range(size)), axis = axis), 
            tf$int64)
  
  values <- tf$ones(size)
  
  dense_shape <- 
    list(tf$cast(size, tf$int64), tf$cast(size, tf$int64))
  
  sparse_tensor <-
    tf$SparseTensor(indices     = indices,
                    values      = values,
                    dense_shape = dense_shape)
  
  sparse_tensor
}
