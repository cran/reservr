% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/tf_util.R
\name{as_params}
\alias{as_params}
\title{Convert TensorFlow tensors to distribution parameters recursively}
\usage{
as_params(x)
}
\arguments{
\item{x}{possibly nested list structure of \code{tensorflow.tensor}s}
}
\value{
A nested list of vectors suitable as distribution parameters
}
\description{
Convert TensorFlow tensors to distribution parameters recursively
}
\examples{
if (interactive()) {
  tf_params <- list(
    probs = k_matrix(t(c(0.5, 0.3, 0.2))),
    shapes = k_matrix(t(c(1L, 2L, 3L)), dtype = "int32"),
    scale = keras3::as_tensor(1.0, keras3::config_floatx())
  )
  params <- as_params(tf_params)
  dist <- dist_erlangmix(vector("list", 3L))
  dist$sample(10L, with_params = params)
}

}
