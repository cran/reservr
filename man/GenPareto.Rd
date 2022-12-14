% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/gpd.R
\name{GenPareto}
\alias{GenPareto}
\alias{rgpd}
\alias{dgpd}
\alias{pgpd}
\alias{qgpd}
\title{The Generalized Pareto Distribution (GPD)}
\usage{
rgpd(n = 1L, u = 0, sigmau = 1, xi = 0)

dgpd(x, u = 0, sigmau = 1, xi = 0, log = FALSE)

pgpd(q, u = 0, sigmau = 1, xi = 0, lower.tail = TRUE, log.p = FALSE)

qgpd(p, u = 0, sigmau = 1, xi = 0, lower.tail = TRUE, log.p = FALSE)
}
\arguments{
\item{n}{integer number of observations.}

\item{u}{threshold parameter (minimum value).}

\item{sigmau}{scale parameter (must be positive).}

\item{xi}{shape parameter}

\item{x, q}{vector of quantiles.}

\item{log, log.p}{logical; if \code{TRUE}, probabilities/densities
\code{p} are given as \code{log(p)}.}

\item{lower.tail}{logical; if \code{TRUE} (default), probabilities are
\eqn{P(X \le x)}, otherwise \eqn{P(X > x)}.}

\item{p}{vector of probabilities.}
}
\value{
\code{rgpd} generates random deviates.

\code{dgpd} gives the density.

\code{pgpd} gives the distribution function.

\code{qgpd} gives the quantile function.
}
\description{
These functions provide information about the generalized Pareto distribution
with threshold \code{u}. \code{dgpd} gives the density, \code{pgpd} gives the distribution
function, \code{qgpd} gives the quantile function and \code{rgpd} generates random
deviates.
}
\details{
If \code{u}, \code{sigmau} or \code{xi} are not specified, they assume the default values of
\code{0}, \code{1} and \code{0} respectively.

The generalized Pareto distribution has density

\deqn{f(x) = 1 / \sigma_u (1 + \xi z)^(- 1 / \xi - 1)}

where \eqn{z = (x - u) / \sigma_u} and \eqn{f(x) = exp(-z)} if
\eqn{\xi} is 0.
The support is \eqn{x \ge u} for \eqn{\xi \ge 0} and
\eqn{u \le x \le u - \sigma_u / \xi} for \eqn{\xi < 0}.

The Expected value exists if \eqn{\xi < 1} and is equal to

\deqn{E(X) = u + \sigma_u / (1 - \xi)}

k-th moments exist in general for \eqn{k\xi < 1}.
}
\examples{

x <- rgpd(1000, u = 1, sigmau = 0.5, xi = 0.1)
xx <- seq(-1, 10, 0.01)
hist(x, breaks = 100, freq = FALSE, xlim = c(-1, 10))
lines(xx, dgpd(xx, u = 1, sigmau = 0.5, xi = 0.1))

plot(xx, dgpd(xx, u = 1, sigmau = 1, xi = 0), type = "l")
lines(xx, dgpd(xx, u = 0.5, sigmau = 1, xi = -0.3), col = "blue", lwd = 2)
lines(xx, dgpd(xx, u = 1.5, sigmau = 1, xi = 0.3), col = "red", lwd = 2)

plot(xx, dgpd(xx, u = 1, sigmau = 1, xi = 0), type = "l")
lines(xx, dgpd(xx, u = 1, sigmau = 0.5, xi = 0), col = "blue", lwd = 2)
lines(xx, dgpd(xx, u = 1, sigmau = 2, xi = 0), col = "red", lwd = 2)

}
\references{
\url{https://en.wikipedia.org/wiki/Generalized_Pareto_distribution}
}
