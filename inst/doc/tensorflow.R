## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup--------------------------------------------------------------------
library(reservr)
library(tensorflow)
library(keras)
library(tibble)
library(ggplot2)

## ----sim-data-----------------------------------------------------------------
if (keras::is_keras_available()) {
  set.seed(1431L)
  tensorflow::set_random_seed(1432L)

  dataset <- tibble(
    x = runif(100, min = 10, max = 20),
    y = 2 * x + rnorm(100)
  )

  ggplot2::qplot(x, y, data = dataset)

  # Specify distributional assumption of OLS:
  dist <- dist_normal(sd = 1.0) # OLS assumption: homoskedasticity

  # Optional: Compute a global fit
  global_fit <- fit(dist, dataset$y)

  # Define a neural network
  nnet_input <- layer_input(shape = 1L, name = "x_input")
  # in practice, this would be deeper
  nnet_output <- nnet_input

  optimizer <- if (packageVersion("keras") >= "2.6.0") {
    optimizer_adam(learning_rate = 0.1)
  } else {
    optimizer_adam(lr = 0.1)
  }

  nnet <- tf_compile_model(
    inputs = list(nnet_input),
    intermediate_output = nnet_output,
    dist = dist,
    optimizer = optimizer,
    censoring = FALSE, # Turn off unnecessary features for this problem
    truncation = FALSE
  )

  nnet_fit <- fit(nnet, x = dataset$x, y = dataset$y, epochs = 100L, batch_size = 100L, shuffle = FALSE)

  plot(nnet_fit)

  pred_params <- predict(nnet, data = list(k_constant(dataset$x)))

  lm_fit <- lm(y ~ x, data = dataset)

  dataset$y_pred <- pred_params$mean
  dataset$y_lm <- predict(lm_fit, newdata = dataset, type = "response")

  ggplot(dataset, aes(x = x, y = y)) %+%
    geom_point() %+%
    geom_line(aes(y = y_pred)) %+%
    geom_line(aes(y = y_lm), linetype = 2L)

  coef_nnet <- rev(as.numeric(nnet$model$get_weights()))
  coef_lm <- coef(lm_fit)

  print(coef_nnet)
  print(coef_lm)
}

