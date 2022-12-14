% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/dist_translate.R
\name{dist_translate}
\alias{dist_translate}
\title{Tranlsated distribution}
\usage{
dist_translate(dist = NULL, offset = NULL, multiplier = 1)
}
\arguments{
\item{dist}{An underlying distribution, or \code{NULL} as a placeholder.}

\item{offset}{Offset to be added to each observation, or \code{NULL} as a placeholder.}

\item{multiplier}{Factor to multiply each observation by, or \code{NULL} as a placeholder.}
}
\value{
A \code{TranslatedDistribution} object.
}
\description{
Tranlsated distribution
}
\examples{
d_norm <- dist_normal(mean = 0, sd = 1)
d_tnorm <- dist_translate(dist = d_norm, offset = 1)
plot_distributions(d_norm, d_tnorm, .x = seq(-2, 3, length.out = 100))

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
\code{\link{dist_trunc}()},
\code{\link{dist_uniform}()},
\code{\link{dist_weibull}()}
}
\concept{Distributions}
