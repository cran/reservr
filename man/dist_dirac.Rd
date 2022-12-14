% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/dist_dirac.R
\name{dist_dirac}
\alias{dist_dirac}
\title{Dirac (degenerate point) Distribution}
\usage{
dist_dirac(point = NULL)
}
\arguments{
\item{point}{The point with probability mass 1.}
}
\value{
A \code{DiracDistribution} object.
}
\description{
A degenerate distribution with all mass at a single point.
}
\details{
The parameter can be overridden with
\code{with_params = list(point = ...)}.
}
\examples{
d_dirac <- dist_dirac(1.5)
d_dirac$sample(2L)
d_dirac$sample(2L, list(point = 42.0))

}
\seealso{
Other Distributions: 
\code{\link{Distribution}},
\code{\link{dist_bdegp}()},
\code{\link{dist_beta}()},
\code{\link{dist_binomial}()},
\code{\link{dist_blended}()},
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
\code{\link{dist_uniform}()},
\code{\link{dist_weibull}()}
}
\concept{Distributions}
