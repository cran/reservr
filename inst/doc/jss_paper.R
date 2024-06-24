## ----setup, include=FALSE-----------------------------------------------------
library(reservr)
options(prompt = 'R> ', continue = '+ ')
keras_available <- reticulate::py_module_available("tensorflow.keras")

## -----------------------------------------------------------------------------
trunc_obs(1.3)

## -----------------------------------------------------------------------------
set.seed(123)
N <- 1000L
x <- rnorm(N)
is_censored <- rbinom(N, size = 1L, prob = 0.8) == 1L

c_lower <- runif(sum(is_censored), min = -2.0, max = 0.0)
c_upper <- c_lower + runif(sum(is_censored), min = 0, max = 1.0)

x_lower <- x
x_upper <- x

x_lower[is_censored] <- dplyr::case_when(
  x[is_censored] <= c_lower ~ -Inf,
  x[is_censored] <= c_upper ~ c_lower,
  TRUE ~ c_upper
)
x_upper[is_censored] <- dplyr::case_when(
  x[is_censored] <= c_lower ~ c_lower,
  x[is_censored] <= c_upper ~ c_upper,
  TRUE ~ Inf
)

t_lower <- runif(N, min = -2.0, max = 0.0)
t_upper <- runif(N, min = 0.0, max = 2.0)

is_observed <- t_lower <= x & x <= t_upper

obs <- trunc_obs(
  xmin = pmax(x_lower, t_lower)[is_observed],
  xmax = pmin(x_upper, t_upper)[is_observed],
  tmin = t_lower[is_observed],
  tmax = t_upper[is_observed]
)

## -----------------------------------------------------------------------------
obs[8L:12L, ]

## -----------------------------------------------------------------------------
nrow(obs)

## -----------------------------------------------------------------------------
sum(is.na(obs$x))

## -----------------------------------------------------------------------------
dist <- dist_normal(sd = 1.0)

## ----error = TRUE-------------------------------------------------------------
dist$sample(1L)

## -----------------------------------------------------------------------------
set.seed(10L)
dist$sample(1L, with_params = list(mean = 0.0))
set.seed(10L)
dist$sample(1L, with_params = list(mean = 0.0, sd = 2.0))

## -----------------------------------------------------------------------------
set.seed(10L)
dist$sample(3L, with_params = list(mean = 0.0:2.0, sd = 0.5))

## -----------------------------------------------------------------------------
dist <- dist_normal(sd = 1.0)
mix <- dist_mixture(dists = list(dist_normal(), NULL))

dist$default_params
mix$default_params
str(dist$get_placeholders())
str(mix$get_placeholders())
str(dist$param_bounds)
str(mix$param_bounds)
str(dist$get_param_bounds())
str(mix$get_param_bounds())
str(dist$get_param_constraints())
str(mix$get_param_constraints())
dist$get_components()
mix$get_components()

## -----------------------------------------------------------------------------
dist <- dist_normal()
flatten_params_matrix(dist$get_placeholders())
denscmp <- dist$compile_density()

if (requireNamespace("bench", quietly = TRUE)) {
  bench::mark(
    dist$density(-2:2, with_params = list(mean = 0.0, sd = 1.0)),
    denscmp(-2:2, matrix(c(0.0, 1.0), nrow = 5L, ncol = 2L, byrow = TRUE)),
    dnorm(-2:2, mean = rep(0.0, 5L), sd = rep(1.0, 5L))
  )
}

## -----------------------------------------------------------------------------

dist1 <- dist_normal(mean = -1.0, sd = 1.0)
dist2 <- dist_exponential(rate = 1.0)
distb <- dist_blended(
  dists = list(dist1, dist2),
  breaks = list(0.0),
  bandwidths = list(1.0),
  probs = list(0.5, 0.5)
)

## -----------------------------------------------------------------------------
distt1 <- dist_trunc(dist1, min = -Inf, max = 0.0)
distt2 <- dist_trunc(dist2, min = 0.0, max = Inf)

distb1 <- distb$clone()
distb1$default_params$probs <- list(1.0, 0.0)
distb2 <- distb$clone()
distb2$default_params$probs <- list(0.0, 1.0)

## ----echo = FALSE-------------------------------------------------------------
library(ggplot2)

distb_data <- tibble::tibble(
  dist = list(dist1, dist2, distt1, distt2, distb1, distb2),
  component = rep(c("Normal", "Exp"), times = 3L),
  transformation = factor(rep(c("original", "truncated", "blended"), each = 2L), levels = c("original", "truncated", "blended"))
)
distb_data$points <- lapply(distb_data$dist, function(d) {
  xb <- seq(-5, 5, length.out = 101)
  tibble::tibble(
    x = xb,
    density = d$density(xb)
  )
})
distb_data$dist <- NULL
distb_data <- tidyr::unnest(distb_data, points)

blending_annotation <- geom_vline(
  xintercept = c(0, -1, 1),
  linetype = c(1L, 3L, 3L),
  color = "black"
)

ggplot(
  distb_data[distb_data$density > 0, ],
  aes(x = x, y = density, color = component, linetype = transformation)
) +
  geom_line() +
  blending_annotation +
  scale_x_continuous(name = NULL)

ggplot(
  tibble::tibble(
    x = seq(-5, 5, length.out = 101),
    density = distb$density(x)
  ),
  aes(x = x, y = density)
) +
  geom_line() +
  blending_annotation +
  scale_x_continuous(name = NULL)

## -----------------------------------------------------------------------------
dist <- dist_normal(sd = 1.0)
the_fit <- fit(dist, obs)
str(the_fit)

## -----------------------------------------------------------------------------
plot_distributions(
  true = dist,
  fitted = dist,
  empirical = dist_empirical(0.5 * (obs$xmin + obs$xmax)),
  .x = seq(-5, 5, length.out = 201),
  plots = "density",
  with_params = list(
    true = list(mean = 0.0, sd = 1.0),
    fitted = the_fit$params
  )
)

## -----------------------------------------------------------------------------
dist <- dist_erlangmix(list(NULL, NULL, NULL))
params <- list(
  shapes = list(1L, 4L, 12L),
  scale = 2.0,
  probs = list(0.5, 0.3, 0.2)
)

set.seed(1234)
x <- dist$sample(100L, with_params = params)

set.seed(32)
init_true <- fit_dist_start(dist, x, init = "shapes",
                            shapes = as.numeric(params$shapes))
init_fan <- fit_dist_start(dist, x, init = "fan", spread = 3L)
init_kmeans <- fit_dist_start(dist, x, init = "kmeans")
init_cmm <- fit_dist_start(dist, x, init = "cmm")
rbind(
  flatten_params(init_true),
  flatten_params(init_fan),
  flatten_params(init_kmeans),
  flatten_params(init_cmm)
)

set.seed(32)
str(fit(dist, x, init = "shapes", shapes = as.numeric(params$shapes)))
fit(dist, x, init = "fan", spread = 3L)$logLik
fit(dist, x, init = "kmeans")$logLik
fit(dist, x, init = "cmm")$logLik

## ----eval = keras_available---------------------------------------------------
set.seed(1431L)
keras3::set_random_seed(1432L)

dataset <- tibble::tibble(
  x = runif(100, min = 10, max = 20),
  y = 2 * x + rnorm(100)
)

## -----------------------------------------------------------------------------
dist <- dist_normal(sd = 1.0)

## ----eval = keras_available---------------------------------------------------
nnet_input <- keras3::keras_input(shape = 1L, name = "x_input")
nnet_output <- nnet_input

## ----eval = keras_available---------------------------------------------------
nnet <- tf_compile_model(
  inputs = list(nnet_input),
  intermediate_output = nnet_output,
  dist = dist,
  optimizer = keras3::optimizer_adam(learning_rate = 0.1),
  censoring = FALSE,
  truncation = FALSE
)
nnet$dist
nnet$model

## ----eval = keras_available---------------------------------------------------
nnet_fit <- fit(
  nnet,
  x = dataset$x,
  y = dataset$y,
  epochs = 100L,
  batch_size = 100L,
  shuffle = FALSE,
  verbose = FALSE
)

## ----eval = keras_available---------------------------------------------------
# Fix weird behavior of keras3
nnet_fit$params$epochs <- max(nnet_fit$params$epochs, length(nnet_fit$metrics$loss))
plot(nnet_fit)

## ----eval = keras_available---------------------------------------------------
pred_params <- predict(nnet, data = list(keras3::as_tensor(dataset$x, keras3::config_floatx())))

lm_fit <- lm(y ~ x, data = dataset)

dataset$y_pred <- pred_params$mean
dataset$y_lm <- predict(lm_fit, newdata = dataset, type = "response")

library(ggplot2)
ggplot(dataset, aes(x = x, y = y)) +
  geom_point() +
  geom_line(aes(y = y_pred), color = "blue") +
  geom_line(aes(y = y_lm), linetype = 2L, color = "green")

## ----eval = keras_available---------------------------------------------------
coef_nnet <- rev(as.numeric(nnet$model$get_weights()))
coef_lm <- unname(coef(lm_fit))

str(coef_nnet)
str(coef_lm)

## ----eval = keras_available---------------------------------------------------
set.seed(1219L)
tensorflow::set_random_seed(1219L)
keras3::config_set_floatx("float32")

dist <- dist_exponential()
ovarian <- survival::ovarian
dat <- list(
  y = trunc_obs(
    xmin = ovarian$futime,
    xmax = ifelse(ovarian$fustat == 1, ovarian$futime, Inf)
  ),
  x = list(
    age = keras3::as_tensor(ovarian$age, keras3::config_floatx(), shape = nrow(ovarian)),
    flags = k_matrix(ovarian[, c("resid.ds", "rx", "ecog.ps")] - 1.0)
  )
)

## ----eval = keras_available---------------------------------------------------
nnet_inputs <- list(
  keras3::keras_input(shape = 1L, name = "age"),
  keras3::keras_input(shape = 3L, name = "flags")
)

## ----eval = keras_available---------------------------------------------------
hidden1 <- keras3::layer_concatenate(
  keras3::layer_normalization(nnet_inputs[[1L]]),
  nnet_inputs[[2L]]
)
hidden2 <- keras3::layer_dense(
  hidden1,
  units = 5L,
  activation = keras3::activation_relu
)
nnet_output <- keras3::layer_dense(
  hidden2,
  units = 5L,
  activation = keras3::activation_relu
)

nnet <- tf_compile_model(
  inputs = nnet_inputs,
  intermediate_output = nnet_output,
  dist = dist,
  optimizer = keras3::optimizer_adam(learning_rate = 0.01),
  censoring = TRUE,
  truncation = FALSE
)
nnet$model

## ----eval = keras_available---------------------------------------------------
str(predict(nnet, dat$x))
global_fit <- fit(dist, dat$y)
tf_initialise_model(nnet, params = global_fit$params, mode = "zero")
str(predict(nnet, dat$x))

## ----eval = keras_available---------------------------------------------------
nnet_fit <- fit(
  nnet,
  x = dat$x,
  y = dat$y,
  epochs = 100L,
  batch_size = nrow(dat$y),
  shuffle = FALSE,
  verbose = FALSE
)

nnet_fit$params$epochs <- max(nnet_fit$params$epochs, length(nnet_fit$metrics$loss))
plot(nnet_fit)

ovarian$expected_lifetime <- 1.0 / predict(nnet, dat$x)$rate

## ----echo = FALSE, eval = keras_available-------------------------------------
ggplot(ovarian, aes(x = age, y = expected_lifetime, color = factor(rx))) +
  geom_point() +
  geom_hline(yintercept = 1.0 / global_fit$params$rate, color = "blue", linetype = "dotted") +
  scale_color_discrete(name = "treatment group")

## ----echo = FALSE, eval = keras_available-------------------------------------
ggplot(ovarian[order(ovarian$futime), ], aes(x = seq_len(nrow(ovarian)))) +
  geom_linerange(aes(ymin = futime, ymax = ifelse(fustat == 1, futime, Inf)), show.legend = FALSE) +
  geom_point(aes(y = futime, shape = ifelse(fustat == 1, "observed", "censored"))) +
  geom_point(aes(y = expected_lifetime, shape = "predicted"), color = "blue") +
  geom_hline(yintercept = 1.0 / global_fit$params$rate, color = "blue", linetype = "dotted") +
  coord_flip() +
  scale_x_continuous(name = "subject", breaks = NULL) +
  scale_y_continuous(name = "lifetime") +
  scale_shape_manual(name = NULL, values = c(observed = "circle", censored = "circle open", predicted = "cross")) +
  guides(shape = guide_legend(override.aes = list(color = c("black", "black", "blue"))))

