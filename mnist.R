# Following the mnist example from https://keras.rstudio.com/
# Helsinki university course "Computer age statistical inference"
# Tuomo Nieminen 05/2018

# Prequisities
# ----
# install.packages("keras")
# keras::install_keras()


# 1 load the MNIST data
# ----

rm(list =ls())
library(keras)

mnist <- dataset_mnist()
X <- mnist$train$x
y <- mnist$train$y

# 2 Explore the data
# ----

# see what we have
str(X) # images array (n, 28, 28)
str(y) # labels (n)

table(y) # distribution of labels

# plot some of the images
which <- "CHANGE ME!"
image(X[which, , ], col = grey(seq(1,0, length = 256)))


# 3 Reshaping
# ----

# reshape the n, 28, 28 array to a n, 28*28 matrix
X_ <- array_reshape(X, c(nrow(X), 784))

# rescale to 0...1
X_ <- X_ / 255

# hot-one encode y
y_ <- to_categorical(y)
head(y_)


# 4 Train and test split
# ----

n <- nrow(X)
train_indices <- sample(1:n, n*0.8)

x_train <- X_[train_indices , ]
y_train <- y_[train_indices, ]

x_test <- X_[-train_indices , ]
y_test <- y_[-train_indices, ]


# 5 Build a neural network
# ----

# 5.a 
# define the structure of a network
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 64, 
              activation = 'relu', 
              input_shape =  c(784),
              kernel_regularizer = regularizer_l2()
              ) %>% 
  layer_dense(units = 10, activation = 'softmax')


# 5.b 
# define the loss and optimizer algorithm
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_sgd(),
  metrics = c('accuracy')
)

# 5.c
# fit the model
history <- model %>% fit(
  x_train, y_train, 
  epochs = 20, 
  batch_size = 480, 
  validation_data = list(x_test, y_test)
)


# 6 performance evaluation
# ----

# see the training history
history
plot(history)

# Use the model to predict labels on the test data
predicted_proba <- model %>% predict(x_test)
predicted_label <- max.col(predictions_proba)
true_label <- max.col(y_test)

# confusion matrix
confusion <- table(predicted_label, true_label) 
confusion

# accuracy
acc <- sum(diag(confusion)) / sum(confusion)
acc
history

