## ----setup, include = FALSE---------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## -----------------------------------------------------------------------------
library(reservr)
set.seed(1L)
# Instantiate an unspecified normal distribution
norm <- dist_normal()
x <- norm$sample(n = 10L, with_params = list(mean = 3, sd = 1))

set.seed(1L)
norm2 <- dist_normal(sd = 1)
x2 <- norm2$sample(n = 10L, with_params = list(mean = 3))

# the same RVs are drawn because the distribution parameters and the seed were the same
stopifnot(identical(x, x2))

## -----------------------------------------------------------------------------
norm$density(x, with_params = list(mean = 3, sd = 1))
dnorm(x, mean = 3, sd = 1)
norm$density(x, log = TRUE, with_params = list(mean = 3, sd = 1)) # log-density
norm$is_discrete_at(x, with_params = list(mean = 3, sd = 1))

# A discrete distribution with mass only at point = x[1].
dd <- dist_dirac(point = x[1])
dd$density(x)
dd$is_discrete_at(x)

## -----------------------------------------------------------------------------
norm$diff_density(x, with_params = list(mean = 3, sd = 1))

## -----------------------------------------------------------------------------
norm$probability(x, with_params = list(mean = 3, sd = 1))
pnorm(x, mean = 3, sd = 1)

dd$probability(x)
dd$probability(x, lower.tail = FALSE, log.p = TRUE)

## -----------------------------------------------------------------------------
norm$diff_probability(x, with_params = list(mean = 3, sd = 1))

## -----------------------------------------------------------------------------
norm$hazard(x, with_params = list(mean = 3, sd = 1))
norm$hazard(x, log = TRUE, with_params = list(mean = 3, sd = 1))

## -----------------------------------------------------------------------------
# Fit with mean, sd free
fit1 <- fit(norm, x)
# Fit with mean free
fit2 <- fit(norm2, x)
# Fit with sd free
fit3 <- fit(dist_normal(mean = 3), x)

# Fitted parameters
fit1$params
fit2$params
fit3$params

# log-Likelihoods can be computed on
AIC(fit1$logLik)
AIC(fit2$logLik)
AIC(fit3$logLik)

# Convergence checks
fit1$opt$message
fit2$opt$message
fit3$opt$message

## -----------------------------------------------------------------------------
params <- list(mean = 30, sd = 10)
x <- norm$sample(100L, with_params = params)
xl <- floor(x)
xr <- ceiling(x)

cens_fit <- fit(norm, trunc_obs(xmin = xl, xmax = xr))
print(cens_fit)

## -----------------------------------------------------------------------------
params <- list(mean = 30, sd = 10)
x <- norm$sample(100L, with_params = params)
tl <- runif(length(x), min = 0, max = 20)
tr <- runif(length(x), min = 0, max = 60) + tl

# truncate_obs() also truncates observations.
# if data is already truncated, use trunc_obs(x = ..., tmin = ..., tmax = ...) instead.
trunc_fit <- fit(norm, truncate_obs(x, tl, tr))
print(trunc_fit)

attr(trunc_fit$logLik, "nobs")

## -----------------------------------------------------------------------------
# Plot fitted densities
plot_distributions(
  true = norm,
  fit1 = norm,
  fit2 = norm2,
  fit3 = dist_normal(3),
  .x = seq(-2, 7, 0.01),
  with_params = list(
    true = list(mean = 3, sd = 1),
    fit1 = fit1$params,
    fit2 = fit2$params,
    fit3 = fit3$params
  ),
  plots = "density"
)

# Plot fitted densities, c.d.f.s and hazard rates
plot_distributions(
  true = norm,
  cens_fit = norm,
  trunc_fit = norm,
  .x = seq(0, 60, length.out = 101L),
  with_params = list(
    true = list(mean = 30, sd = 10),
    cens_fit = cens_fit$params,
    trunc_fit = trunc_fit$params
  )
)

# More complex distributions
plot_distributions(
  bdegp = dist_bdegp(2, 3, 10, 3),
  .x = c(seq(0, 12, length.out = 121), 1.5 - 1e-6),
  with_params = list(
    bdegp = list(
      dists = list(
        list(), list(), list(
          dists = list(
            list(
              dist = list(
                shapes = as.list(1:3),
                scale = 2.0,
                probs = list(0.2, 0.5, 0.3)
              )
            ),
            list(
              sigmau = 0.4,
              xi = 0.2
            )
          ),
          probs = list(0.7, 0.3)
        )
      ),
      probs = list(0.15, 0.1, 0.75)
    )
  )
)

