# Train Text Analytics



# Install all required packages.
#install.packages(c("ggplot2", "e1071", "caret", "quanteda", 
#                   "irlba", "randomForest"))


spam.raw <- read.csv("spam.csv", stringsAsFactors = FALSE, fileEncoding = "UTF-16")

spam.raw <- spam.raw[, 1:2]
names(spam.raw) <- c("Label", "Text")

spam.raw$Label <- as.factor(spam.raw$Label)

spam.raw$TextLength <- nchar(spam.raw$Text)

library(caret)

set.seed(32984)
indexes <- createDataPartition(spam.raw$Label, times = 1,
                               p = 0.7, list = FALSE)

set.seed(48743)
cv.folds <- createMultiFolds(train$Label, k = 10, times = 3)

cv.cntrl <- trainControl(method = "repeatedcv", number = 10,
                         repeats = 3, index = cv.folds)

#-----------------------------------------------------------------------                               

train <- spam.raw[indexes,]
test <- spam.raw[-indexes,]

library(quanteda)

train.tokens <- tokens(train$Text, what = "word", 
                       remove_numbers = TRUE, remove_punct = TRUE,
                       remove_symbols = TRUE, remove_hyphens = TRUE)
train.tokens <- tokens_tolower(train.tokens)
train.tokens <- tokens_select(train.tokens, stopwords(), 
                              selection = "remove")

train.tokens <- tokens_wordstem(train.tokens, language = "english")


term.frequency <- function(row) {
        row / sum(row)
}

inverse.doc.freq <- function(col) {
        corpus.size <- length(col)
        doc.count <- length(which(col > 0))
        
        log10(corpus.size / doc.count)
}

tf.idf <- function(x, idf) {
        x * idf
}

train.tokens <- tokens_ngrams(train.tokens, n = 1:2)

train.tokens.dfm <- dfm(train.tokens, tolower = FALSE)
train.tokens.matrix <- as.matrix(train.tokens.dfm)

train.tokens.df <- apply(train.tokens.matrix, 1, term.frequency)

train.tokens.idf <- apply(train.tokens.matrix, 2, inverse.doc.freq)

train.tokens.tfidf <-  apply(train.tokens.df, 2, tf.idf, 
                             idf = train.tokens.idf)

train.tokens.tfidf <- t(train.tokens.tfidf)

incomplete.cases <- which(!complete.cases(train.tokens.tfidf))
train.tokens.tfidf[incomplete.cases,] <- rep(0.0, ncol(train.tokens.tfidf))

train.tokens.tfidf.df <- cbind(Label = train$Label, data.frame(train.tokens.tfidf))
names(train.tokens.tfidf.df) <- make.names(names(train.tokens.tfidf.df))

library(irlba)

start.time <- Sys.time()

train.irlba <- irlba(t(train.tokens.tfidf), nv = 300, maxit = 600)

total.time <- Sys.time() - start.time
total.time

sigma.inverse <- 1 / train.irlba$d
u.transpose <- t(train.irlba$u)
document <- train.tokens.tfidf[1,]
document.hat <- sigma.inverse * u.transpose %*% document

train.svd <- data.frame(Label = train$Label, train.irlba$v)
train.svd$TextLength <- train$TextLength


library(doSNOW)

cl <- makeCluster(10, type = "SOCK")
registerDoSNOW(cl)

start.time <- Sys.time()

rf.cv.2 <- train(Label ~ ., data = train.svd, method = "rf",
                 trControl = cv.cntrl, tuneLength = 7, 
                 importance = TRUE)

stopCluster(cl)

total.time <- Sys.time() - start.time
total.time

# Load results from disk.
load("rf.cv.2.RData")

# Check the results.
rf.cv.2

# Drill-down on the results.
confusionMatrix(train.svd$Label, rf.cv.2$finalModel$predicted)

# How important was the new feature?
library(randomForest)
varImpPlot(rf.cv.1$finalModel)
varImpPlot(rf.cv.2$finalModel)


