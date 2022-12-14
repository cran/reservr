% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/dist_discrete.R
\name{dist_discrete}
\alias{dist_discrete}
\title{Discrete Distribution}
\usage{
dist_discrete(size = NULL, probs = NULL)
}
\arguments{
\item{size}{Number of classes parameter (integer). Required if \code{probs} is \code{NULL}.}

\item{probs}{Vector of probabilties parameter, or \code{NULL} as a placeholder.}
}
\value{
A \code{DiscreteDistribution} object.
}
\description{
A full-flexibility discrete distribution with values from 1 to \code{size}.
}
\details{
Parameters can be overridden with
\code{with_params = list(probs = ...)}.
}
\examples{
d_discrete <- dist_discrete(probs = list(0.5, 0.25, 0.15, 0.1))
x <- d_discrete$sample(100)
d_emp <- dist_empirical(x)

plot_distributions(
  empirical = d_emp,
  theoretical = d_discrete,
  estimated = d_discrete,
  with_params = list(
    estimated = list(
      size = max(x),
      probs = as.list(unname(table(x)) / 100)
    )
  ),
  .x = 0:max(x)
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
\code{\link{dist_uniform}()},
\code{\link{dist_weibull}()}
}
\concept{Distributions}
