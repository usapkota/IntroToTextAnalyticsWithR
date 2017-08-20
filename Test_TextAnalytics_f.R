# Test TextAnalytics final

# Tokenization.
test.tokens <- tokens(test$Text, what = "word", 
                      remove_numbers = TRUE, remove_punct = TRUE,
                      remove_symbols = TRUE, remove_hyphens = TRUE)

# Lower case the tokens.
test.tokens <- tokens_tolower(test.tokens)

# Stopword removal.
test.tokens <- tokens_select(test.tokens, stopwords(), 
                             selection = "remove")

# Stemming.
test.tokens <- tokens_wordstem(test.tokens, language = "english")

# Add bigrams.
test.tokens <- tokens_ngrams(test.tokens, n = 1:2)

# Convert n-grams to quanteda document-term frequency matrix.
test.tokens.dfm <- dfm(test.tokens, tolower = FALSE)

# Explore the train and test quanteda dfm objects.
train.tokens.dfm
test.tokens.dfm

# Ensure the test dfm has the same n-grams as the training dfm.
#
# NOTE - In production we should expect that new text messages will 
#        contain n-grams that did not exist in the original training
#        data. As such, we need to strip those n-grams out.
#
test.tokens.dfm <- dfm_select(test.tokens.dfm, features = train.tokens.dfm)
test.tokens.matrix <- as.matrix(test.tokens.dfm)
test.tokens.dfm




# With the raw test features in place next up is the projecting the term
# counts for the unigrams into the same TF-IDF vector space as our training
# data. The high level process is as follows:
#      1 - Normalize each document (i.e, each row)
#      2 - Perform IDF multiplication using training IDF values

# Normalize all documents via TF.
test.tokens.df <- apply(test.tokens.matrix, 1, term.frequency)
str(test.tokens.df)

# Lastly, calculate TF-IDF for our training corpus.
test.tokens.tfidf <-  apply(test.tokens.df, 2, tf.idf, idf = train.tokens.idf)
dim(test.tokens.tfidf)
View(test.tokens.tfidf[1:25, 1:25])

# Transpose the matrix
test.tokens.tfidf <- t(test.tokens.tfidf)

# Fix incomplete cases
summary(test.tokens.tfidf[1,])
test.tokens.tfidf[is.na(test.tokens.tfidf)] <- 0.0
summary(test.tokens.tfidf[1,])




# With the test data projected into the TF-IDF vector space of the training
# data we can now to the final projection into the training LSA semantic
# space (i.e. the SVD matrix factorization).
test.svd.raw <- t(sigma.inverse * u.transpose %*% t(test.tokens.tfidf))




# Lastly, we can now build the test data frame to feed into our trained
# machine learning model for predictions. First up, add Label and TextLength.
test.svd <- data.frame(Label = test$Label, test.svd.raw, 
                       TextLength = test$TextLength)


# # Next step, calculate SpamSimilarity for all the test documents. First up, 
# # create a spam similarity matrix.
# test.similarities <- rbind(test.svd.raw, train.irlba$v[spam.indexes,])
# test.similarities <- cosine(t(test.similarities))
# 
# test.svd$SpamSimilarity <- rep(0.0, nrow(test.svd))
# spam.cols <- (nrow(test.svd) + 1):ncol(test.similarities)
# for(i in 1:nrow(test.svd)) {
#         test.svd$SpamSimilarity[i] <- mean(train.similarities[i, spam.cols])  
# }


# Now we can make predictions on the test data set using our trained mighty 
# random forest.
preds <- predict(rf.cv.2, test.svd)

# Drill-in on results
confusionMatrix(preds, test.svd$Label)

