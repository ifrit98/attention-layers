#' Calculate all divisor pairs for a given number
#' @export
calculate_divisor_pairs <- function(x) {
  pairs <- list()
  d     <- get_divisors(x)
  n     <- 1
  
  for (i in d) {
    for (j in d) {
      y <- i * j
      if (y == x) {
        pairs[[n]] <- c(i, j)
        n <- n + 1
      }
    }
  }
  pairs
}


#' Get all divisors of a given number
#' @export
get_divisors <- function(x) {
  i   <- 1L
  j   <- 1L
  div <- c() 
  while (i <= x) {
    if (x %% i == 0L) {
      div[j] <- i
      j <- j + 1L
    }
    i <- i + 1L
  }
  div
}


#' Greatest common divisor
#' @export
gcd <- function(x, y) {
  while(y) {
    temp = y
    y = x %% y
    x = temp
  }
  return(x)
}

#' WTF
product <- function(from, to, .f, x = 1L) {
  .x <- x
  for (j in from:to) {
    .x <- .x * .f(j, .x)
  }
  .x
}


#' Powers of 2 up to a given number.
#' 
#' Takes log2 of n and calculates 2^x, where 
#' x ranges from 1:log2(n)
#' @export
pow2_up_to <- function(n) {
  x <- floor(log2(n))
  powers_of_2(x)
}


#' List powers of 2 up to a given exponent, x
#' @export
powers_of_2 <- function(x) {
  x <- as.integer(x)
  l <- seq(0L, x)
  vapply(l, function(p) 2^p, 0)
}


# Equivalency in tensor extraction 
# x[, 0:-1:2, ,style = "python"] == x[, NULL:NULL:2,]

# Missing arguments for python syntax are valid, but they must be supplied 
# as NULL or whole expression must by backticked.
# x[, `2:`,]
# x[, `::2`] 
# x[, NULL:NULL:2] 

#    R             python
# x[all_dims()] == x[...]

# x[-1, , ] # Last row

# x[all_dims(), , style = "python"] == x[all_dims(), NULL:NULL, style = "python"]
