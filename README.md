# MUDAC2018

### Open file

```{r}
read.csv(filename, header)
read.table()
```

### Create data partition using sample

```{r}
# 1
data_idx = sample(nrow(Default), 5000)
data_trn = Default[data_idx, ]
data_tst = Default[-data_idx, ]
```

```{r}
# 2
library(caret)
data_idx_2 = createDataPartition(y, p = 0.8, list = F)
data_trn_2 = Default[data_idx_2, ]
data_tst_2 = Default[-data_idx_2, ]
```

### NA handle

```{r}
na.omit(Default)
na.exclude(Default)
na.pass(Default)
na.mean(Default)
```

### Helper function

```{r}
calc_acc = function(actual, predicted) {
  mean(actual == predicted)
}

get_best_result = function(caret_fit) {
  best = which(rownames(caret_fit$results) == rownames(caret_fit$bestTune))
  best_result = caret_fit$results[best, ]
  rownames(best_result) = NULL
  best_result
}
```


### Logistic Regression

```{r}
model_glm = glm(default ~ balance, data = data_trn, family = "binomial")
coef(model_glm)
# give probability
head(predict(model_glm, type = "response"))

# glm using caret
default_glm_mod = train(
  form = default ~ .,
  data = default_trn,
  trControl = trainControl(method = "cv", number = 5),
  method = "glm",
  family = "binomial"
)
default_glm_mod$finalModel
```

### Naive Bayes

```{r}
library(e1071)
iris_nb = naiveBayes(Species ~ ., data = iris_trn)
iris_nb
```

### KNN

```{r}
default_knn_mod = train(
  default ~ .,
  data = default_trn,
  method = "knn",
  trControl = trainControl(method = "cv", number = 5),
  preProcess = c("center", "scale"),
  tuneGrid = expand.grid(k = seq(1, 101, by = 2))
)
```


### LDA

```{r}
library(MASS)
iris_lda = lda(Species ~ ., data = iris_trn)
iris_lda_trn_pred = predict(iris_lda, iris_trn)$class
iris_lda_tst_pred = predict(iris_lda, iris_tst)$class
iris_lda_flat = lda(Species ~ ., data = iris_trn, prior = c(1, 1, 1) / 3)
```

### QDA

```{r}
iris_qda = qda(Species ~ ., data = iris_trn)
iris_qda
```

### Ridge

```{r}
library(glmnet)
data(Hitters, package = "ISLR")
Hitters = na.omit(Hitters)

# no cv
par(mfrow = c(1, 2))
fit_ridge = glmnet(X, y, alpha = 0)
plot(fit_ridge)
plot(fit_ridge, xvar = "lambda", label = TRUE)

# cv
fit_ridge_cv = cv.glmnet(X, y, alpha = 0)
plot(fit_ridge_cv)
coef(fit_ridge_cv)
coef(fit_ridge_cv, s = "lambda.min")

# penalty term using minimum lambda
sum(coef(fit_ridge_cv, s = "lambda.min")[-1] ^ 2)

# predict using 1-SE rule lambda, default behavior
predict(fit_ridge_cv, X)

# calcualte "train error"
mean((y - predict(fit_ridge_cv, X)) ^ 2)
```

### Lasso

```{r}
# no cv
par(mfrow = c(1, 2))
fit_lasso = glmnet(X, y, alpha = 1)
plot(fit_lasso)
plot(fit_lasso, xvar = "lambda", label = TRUE)

# cv
fit_lasso_cv = cv.glmnet(X, y, alpha = 1)
plot(fit_lasso_cv)

# fitted coefficients, using 1-SE rule lambda, default behavior
coef(fit_lasso_cv)

# fitted coefficients, using minimum lambda
coef(fit_lasso_cv, s = "lambda.min")

# penalty term using minimum lambda
sum(coef(fit_lasso_cv, s = "lambda.min")[-1] ^ 2)
```

### Broom for interpreting glmnet result

```{r}
library(broom)
# the output from the commented line would be immense
# fit_lasso_cv
tidy(fit_lasso_cv)
# the two lambda values of interest
glance(fit_lasso_cv) 
```

### Elastic Net

```{r}
library(caret)
library(glmnet)
data(Hitters, package = "ISLR")
Hitters = na.omit(Hitters)
#We then fit an elastic net with a tuning grid.
def_elnet_int = train(
  default ~ . ^ 2, data = default_trn,
  method = "glmnet",
  trControl = cv_5,
  tuneLength = 10
)
get_best_result(def_elnet_int)

# or tune alpha
for (i in seq(0.1, 0.9, 0.3)) {
  credit_elastic = cv.glmnet(credit_X_trn, credit_Y_trn, alpha = i)
  credit_elastic_pred = ifelse(predict(credit_elastic, credit_X_tst, s = 'lambda.min') > 0.5, 1, 0)
  credit_acc = calc_class_acc(credit_Y_tst, credit_elastic_pred)
  if (credit_acc > credit_acc_big) {
    credit_elastic_big = credit_elastic
    credit_acc_big = credit_acc
    credi_alpha_big = i
  }
}
coef(credit_elastic_big, s='lambda.min')
credit_acc_big
```

### Tree

```{r}
seat_tree = rpart(Sales ~ ., data = seat_trn)
rpart.plot(seat_tree)
seat_tree_tst_pred = predict(seat_tree, seat_tst, type = "class")
table(predicted = seat_tree_tst_pred, actual = seat_tst$Sales)
```

### Boost tree

```{r}
gbm_grid =  expand.grid(interaction.depth = 1:5,
                        n.trees = (1:6) * 500,
                        shrinkage = c(0.001, 0.01, 0.1),
                        n.minobsinnode = 10)

set.seed(42)
sim_gbm_mod = train(
  y ~ .,
  data = sim_trn,
  trControl = trainControl(method = "cv", number = 5),
  method = "gbm",
  tuneGrid = gbm_grid,
  verbose = FALSE
)

booston_boost = gbm(medv ~ ., data = boston_trn, distribution = "gaussian",
                    n.trees = 5000, interaction.depth = 4, shrinkage = 0.01)
booston_boost
tibble::as_tibble(summary(booston_boost))
plot(seat_gbm_tune)

```


### Random Forest

```{r}
library(randomForest)
oob = trainControl(method = "oob")
cv_5 = trainControl(method = "cv", number = 5)
rf_grid =  expand.grid(mtry = 1:dim(seat_trn) - 1)
# bagging
seat_bag = randomForest(Sales ~ ., data = seat_trn, mtry = 10,
                        importance = TRUE, ntrees = 500)
seat_bag

# no bagging
seat_forest = randomForest(Sales ~ ., data = seat_trn, mtry = 3, importance = TRUE, ntrees = 500)
seat_forest

set.seed(825)
seat_rf_tune = train(Sales ~ ., data = seat_trn,
                     method = "rf",
                     trControl = oob,
                     verbose = FALSE,
                     tuneGrid = rf_grid)
seat_rf_tune
seat_rf_tune$bestTune

```

### Plot

```{r}
plot(default ~ balance, data = data_trn_lm, 
     col = "darkorange", pch = "|", ylim = c(-0.2, 1),
     main = "Using Linear Regression for Classification")
abline(h = 0, lty = 3)
abline(h = 1, lty = 3)
abline(h = 0.5, lty = 2)
abline(model_lm, lwd = 3, col = "dodgerblue")
```

### Kable

```{r}
classifiers = c("LDA", "LDA, Flat Prior", "QDA", "Naive Bayes")

train_err = c(
  calc_class_err(predicted = iris_lda_trn_pred,      actual = iris_trn$Species),
  calc_class_err(predicted = iris_lda_flat_trn_pred, actual = iris_trn$Species),
  calc_class_err(predicted = iris_qda_trn_pred,      actual = iris_trn$Species),
  calc_class_err(predicted = iris_nb_trn_pred,       actual = iris_trn$Species)
)

test_err = c(
  calc_class_err(predicted = iris_lda_tst_pred,      actual = iris_tst$Species),
  calc_class_err(predicted = iris_lda_flat_tst_pred, actual = iris_tst$Species),
  calc_class_err(predicted = iris_qda_tst_pred,      actual = iris_tst$Species),
  calc_class_err(predicted = iris_nb_tst_pred,       actual = iris_tst$Species)
)

results = data.frame(
  classifiers,
  train_err,
  test_err
)

colnames(results) = c("Method", "Train Error", "Test Error")
```

```{r}
knitr::kable(results)
knitr::kable(head(sim_gbm_mod$results), digits = 3)
```

### Tune
- Bagging: Actually just a subset of Random Forest with mtry =  p
- Random Forest: mtry
- Boosting: n.trees, interaction.depth, shrinkage, n.minobsinnode
