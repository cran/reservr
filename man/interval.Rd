% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/interval.R
\name{interval}
\alias{interval}
\alias{is.Interval}
\title{Intervals}
\usage{
interval(
  range = c(-Inf, Inf),
  ...,
  include_lowest = closed,
  include_highest = closed,
  closed = FALSE,
  integer = FALSE,
  read_only = FALSE
)

is.Interval(x)
}
\arguments{
\item{range}{The interval boundaries as a sorted two-element numeric vector.}

\item{...}{First argument is used as the endpoint if \code{range} has length 1.
Additional arguments, or any if \code{range} has length 2, cause a warning and
will be ignored.}

\item{include_lowest}{Is the lower boundary part of the interval?}

\item{include_highest}{Is the upper boundary part of the interval?}

\item{closed}{Is the interval closed?}

\item{integer}{Is the interval only over the integers?}

\item{read_only}{Make the interval object read-only?}

\item{x}{An object.}
}
\value{
\code{interval} returns an \code{Interval}.
\code{is.Interval} returns \code{TRUE} if \code{x} is an \code{Interval}, \code{FALSE} otherwise.
}
\description{
Intervals
}
\examples{
# The real line
interval()

# Closed unit interval
interval(c(0, 1), closed = TRUE)
# Alternative form
interval(0, 1, closed = TRUE)

# Non-negative real line
interval(c(0, Inf), include_lowest = TRUE)
}
\seealso{
interval-operations
}
