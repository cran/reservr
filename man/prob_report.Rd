% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/prob_report.R
\name{prob_report}
\alias{prob_report}
\title{Determine probability of reporting under a Poisson arrival Process}
\usage{
prob_report(
  dist,
  intervals,
  expo = NULL,
  with_params = list(),
  .tolerance = .Machine$double.eps^0.5,
  .max_iter = 100L,
  .try_compile = TRUE
)
}
\arguments{
\item{dist}{A reporting delay Distribution, or a compiled interval probability function.}

\item{intervals}{A data frame with columns \code{xmin}, \code{xmax}, \code{tmin}, \code{tmax}.
Claims occur within \verb{[xmin, xmax]} and be reported within \verb{[tmin, tmax]}.}

\item{expo}{Poisson intensity. If given, must be a vectorised function that
yields the intensity of the claim arrival process at a specified time.
\code{expo = NULL} is equivalent to a constant intensity function. \code{expo} is only
relevant up to a multiplicative constant.}

\item{with_params}{Parameters of \code{dist} to use. Can be a parameter set with
different values for each interval. If \code{dist} is a compiled interval probability function, \code{with_params} can be a
matrix instead.}

\item{.tolerance}{Absolute element-wise tolerance.}

\item{.max_iter}{Maximum number of iterations. The number of
integration intervals will be at most \code{length(lower) * .max_iter}. Therefor the maximum
number of function evaluations per integration interval will be
\code{15 * .max_iter}.}

\item{.try_compile}{Try compiling the distributions probability function to speed up integration?}
}
\value{
A vector of reporting probabilities, with one entry per row of \code{intervals}.
}
\description{
Determines the probability that claims occuring under a Poisson process with
arrival intensity \code{expo} and reporting delay distribution \code{dist} during the
time between \code{t_min} and \code{t_max} are reported between \code{tau_min} and
\code{tau_max}.
}
\details{
The reporting probability is given by

P(x + d in [tmin, tmax] | x in [xmin, xmax])
= E(P(x + d in [tmin, tmax] | x) | x in [xmin, xmax]) /
P(x in [xmin, xmax])
= int_[xmin, xmax] expo(x) P(x + d in [tmin, tmax]) dx
= int_[xmin, xmax] expo(x) P(d in [tmin - x, tmax - x]) dx /
int_[xmin, xmax] expo(x) dx

\code{prob_report} uses \code{\link[=integrate_gk]{integrate_gk()}} to compute the two integrals.
}
\examples{
dist <- dist_exponential()
ints <- data.frame(
  xmin = 0,
  xmax = 1,
  tmin = seq_len(10) - 1.0,
  tmax = seq_len(10)
)
params <- list(rate = rep(c(1, 0.5), each = 5))

prob_report(dist, ints, with_params = params)

}
