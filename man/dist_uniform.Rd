% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/dist_uniform.R
\name{dist_uniform}
\alias{dist_uniform}
\title{Uniform distribution}
\usage{
dist_uniform(min = NULL, max = NULL)
}
\arguments{
\item{min}{Lower limit, or \code{NULL} as a placeholder.}

\item{max}{Upper limit, or \code{NULL} as a placeholder.}
}
\value{
A \code{UniformDistribution} object.
}
\description{
See \link[stats:Uniform]{stats::Uniform}
}
\details{
Both parameters can be overridden with
\code{with_params = list(min = ..., max = ...)}.
}
\examples{
d_unif <- dist_uniform(min = 0, max = 1)
x <- d_unif$sample(100)
d_emp <- dist_empirical(x)

plot_distributions(
  empirical = d_emp,
  theoretical = d_unif,
  estimated = d_unif,
  with_params = list(
    estimated = inflate_params(
      fitdistrplus::fitdist(x, distr = "unif")$estimate
    )
  ),
  .x = seq(0, 1, length.out = 100)
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
\code{\link{dist_erlangmix}()},
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
\code{\link{dist_weibull}()}
}
\concept{Distributions}
