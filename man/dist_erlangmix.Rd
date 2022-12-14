% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/dist_erlangmix.R
\name{dist_erlangmix}
\alias{dist_erlangmix}
\title{Erlang Mixture distribution}
\usage{
dist_erlangmix(shapes, scale = NULL, probs = NULL)
}
\arguments{
\item{shapes}{Shape parameters, a trunc_erlangmix fit, or \code{NULL} as a
placeholder.}

\item{scale}{Common scale parameter, or \code{NULL} as a placeholder.}

\item{probs}{Mixing probabilities, or \code{NULL} as a placeholder.}
}
\value{
An \code{ErlangMixtureDistribution} object.
}
\description{
Erlang Mixture distribution
}
\examples{
params <- list(scale = 1.0, probs = list(0.5, 0.3, 0.2), shapes = list(1L, 2L, 3L))
dist <- dist_erlangmix(vector("list", 3L))
x <- dist$sample(20, with_params = params)
d_emp <- dist_empirical(x, positive = TRUE)

plot_distributions(
  empirical = d_emp,
  theoretical = dist,
  with_params = list(
    theoretical = params
  ),
  .x = seq(1e-4, 5, length.out = 100)
)

}
\seealso{
Other Distributions: 
\code{\link{Distribution}},
\code{\link{dist_bdegp}()},
\code{\link{dist_beta}()},
\code{\link{dist_binomial}()},
\code{\link{dist_blended}()},
\code{\link{dist_dirac}()},
\code{\link{dist_discrete}()},
\code{\link{dist_empirical}()},
\code{\link{dist_exponential}()},
\code{\link{dist_gamma}()},
\code{\link{dist_genpareto}()},
\code{\link{dist_lognormal}()},
\code{\link{dist_mixture}()},
\code{\link{dist_negbinomial}()},
\code{\link{dist_normal}()},
\code{\link{dist_pareto}()},
\code{\link{dist_poisson}()},
\code{\link{dist_translate}()},
\code{\link{dist_trunc}()},
\code{\link{dist_uniform}()},
\code{\link{dist_weibull}()}
}
\concept{Distributions}
