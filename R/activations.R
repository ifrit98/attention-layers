#' Gaussian Error Linear Unit.
#' This is a smoother version of the RELU.
#' Original paper: https://arxiv.org/abs/1606.08415 
#' Args:
#'   x: float Tensor to perform activation.
#'Returns:
#'  x with the GELU activation applied.
#' @export
gelu <- function(x) {
    
  cdf = 0.5 * (1 + tf$tanh(
    (sqrt(2 / pi) * (x + 0.044715 * tf$pow(x, 3L)))))
  
  act <- x * cdf
  
  act
}


#' Bipolar ReLU as in https://arxiv.org/abs/1709.04054
#' @export
brelu <- function(x) {
  x_shape <- shape_list2(x)
  c(x1, x2) %<-%
    tf$split(tf$reshape(x, c(x_shape[1:(length(x_shape) - 1)], 
                             list(-1L, 2L))), 2L, axis = -1L)
  y1 <-  tf$nn$relu(x1)
  y2 <- -tf$nn$relu(-x2)
  
  tf$reshape(tf$concat(list(y1, y2), axis = -1L), x_shape)
  
}


#' Bipolar ELU as in https://arxiv.org/abs/1709.04054.
#' @export
belu <- function(x) {
  x_shape <- shape_list2(x)
  c(x1, x2) %<-%
    tf$split(tf$reshape(x, c(x_shape[1:(length(x_shape) - 1)], 
                           list(-1L, 2L))), 2L, axis = -1L)
  y1 <- tf$nn$elu(x1)
  y2 <- -tf$nn$elu(-x2)
  
  tf$reshape(tf$concat(list(y1, y2), axis = -1L), x_shape)
}


#' NALU as in https://arxiv.org/abs/1808.00508
#' @export
nalu <- function(x, depth, epsilon = 1e-30, name = NULL, reuse = NULL) {
  with(tf$variable_scope(name, default_name = "nalu", values = list(x), reuse = reuse)) {
    x_shape <- shape_list2(x)
  }
  x_flat <- tf$reshape(x, list(-1L, x_shape[length(x_shape)]))
  gw     <- tf$get_variable("w", list(x_shape[length(x_shape)], depth))
  g      <- tf$nn$sigmoid(tf$matmul(x_flat, gw))
  g      <- tf$reshape(g, c(x_shape[1:(length(x_shape)-1)], depth))
  a      <- nac(x, depth, name = "nac_lin")
  log_x  <- tf$log(tf$abs(x) + epsilon)
  m      <- nac(log_x, depth, name = "nac_log")
  
  out <- g * a + (1 - g) * tf$exp(m)
  
  out
}


#' NAC as in https://arxiv.org/abs/1808.00508
#' @export
nac <- function(x, depth, name = NULL, reuse = NULL) {
  with(tf$variable_scope(name, default_name = "nac", values = list(x), reuse = reuse)) {
    x_shape <- shape_list2(x)
  }
  w        <- tf$get_variable("w", list(x_shape[[length(x_shape)]], depth))
  m        <- tf$get_variable("m", list(x_shape[[length(x_shape)]], depth))
  w        <- tf$tanh(w) * tf$nn$sigmoid(m)
  x_flat   <- tf$reshape(x, list(-1L, x_shape[[length(x_shape)]]))
  res_flat <- tf$matmul(x_flat, w)
  
  tf$reshape(res_flat, x_shape[1L:(length(x)-1L)])
}

