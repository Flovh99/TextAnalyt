setwd("C:/Users/marij/OneDrive/Documenten/Data Science Master/Text Mining")

# Load required libraries
options(stringsAsFactors = F)
library(ggplot2)
library(tokenizers)
library(tibble)
library(tidyverse)
library(tidytext)
library(SnowballC)
library(tm)
library(stringi)
library(ggrepel)
library(wordcloud)
library(quanteda)
library(smacof)
library(ggfortify)
library(ggthemes)
library(factoextra)
library(dplyr)
library(sentimentr)
library(qdap)
library(wordcloud)
library(syuzhet)
library(NMF)
library(tidyr)
library(mgsub)
library(slam)

##### PART 1

# Load data
disney <- read.csv(file = "DisneylandReviews.csv", sep = ",", header = TRUE)

# Inspect data
str(disney)
head(disney)
disney$Rating <- as.factor(disney$Rating)
disney$Branch <- as.factor(disney$Branch)
summary(nchar(disney$Review_Text))

# Remove error messages in text
disney <- disney[nchar(disney$Review_Text) > 50, ]

# Convert date to date object
disney$Year_Month <- as.Date(paste0(disney$Year_Month, "-01"))

Sys.setlocale('LC_ALL','C') # set aspects of the internationalization of the program

# Replace numbers and certain characters
disney$Review_Text <- as.character(disney$Review_Text) %>%
  tolower() %>%
  {gsub(":( |-|o)*\\("," SADSMILE ", .)} %>%        # Find :( or :-( or : ( or :o(
  {gsub(":( |-|o)*\\)"," HAPPYSMILE ", .)} %>%      # Find :) or :-) or : ) or :o)
  {gsub("(\"| |\\$)-+\\.-+"," NUMBER", .)} %>%      # Find numbers
  {gsub("([0-9]+:)*[0-9]+ *am"," TIME_AM", .)} %>%  # Find time AM
  {gsub("([0-9]+:)*[0-9]+ *pm"," TIME_PM", .)} %>%  # Find time PM
  {gsub("-+:-+","TIME", .)} %>%                     # Find general time
  {gsub("\\$ ?[0-9]*[\\.,]*[0-9]+"," DOLLARVALUE ", .)} %>%   # Find Dollar values
  {gsub("\\€ ?[0-9]*[\\.,]*[0-9]+"," EUROVALUE ", .)} %>%   # Find Euro values
  {gsub("[0-9]*[\\.,]*[0-9]+"," NUMBER ", .)} %>%  # Find remaining numbers
  {gsub("-"," ", .)} %>%                           # Remove all -
  {gsub("&"," and ", .)} %>%                       # Find general time
  {gsub("\"+"," ", .)} %>%                         # Remove all "
  {gsub("\\|+"," ", .)} %>%                        # Remove all |
  {gsub("_+"," ", .)} %>%                          # Remove all _
  {gsub(";+"," ", .)} %>%                          # Remove excess ;
  {gsub(" +"," ", .)} %>%                          # Remove excess spaces
  {gsub("\\.+","\\.", .)}                          # Remove excess .

# Create word vector by splitting on spaces and punctuation
review_words <- disney %>% unnest_tokens(word, Review_Text, to_lower = T)
print("Number of words total: ")
print(nrow(review_words)) # 5,537,502 words

# Count how often each word occurs
counts <- review_words %>%
  count(word, sort = TRUE)
print("Number of unique words: ")
nrow(counts) # 75,365 unique words

# Show top 50 words
head(counts, 50) # Mostly stop words!

# Show bottom 50 words
tail(counts, 50)


# Remove stop words
data(stop_words)
review_words_nostop <- review_words %>%
  anti_join(stop_words)
counts <- review_words_nostop %>%
  count(word, sort = TRUE)


# Clean + stem words
#counts[900:1100, ]
#run_slow <- F
#if (run_slow) {
j <- 1
for (j in 1:nrow(disney)) {
  stemmed_rev <- anti_join(disney[j, ] %>% unnest_tokens(word, Review_Text, drop = F, to_lower = T),
                           stop_words)
  stemmed_rev <- wordStem(stemmed_rev[, "word"], language = "porter")
  disney[j, "Review_Text"] <- paste(stemmed_rev, collapse = " ")
}
save(disney, file = "disney_stemmed.RData")
#}


#cut text into words by splitting on spaces and punctuation
review_words <- disney %>% unnest_tokens(word, Review_Text,to_lower=TRUE) 
print("number of words")
nrow(disney)

#Count the number of times each word occurs
counts <- review_words %>%count(word, sort=TRUE) # sort = TRUE for sorting in descending order of n. 
print("number of unique words after stemming and without stop words")
nrow(counts)



### Remove most infrequent words from the data
infrequent <- counts %>% filter(n<0.01*nrow(disney))
toremove <- infrequent


j<-1 
for (j in 1:nrow(disney)) {
  stemmed_Review<-  anti_join((disney[j,] %>% unnest_tokens(word,Review_Text,to_lower=TRUE) ),toremove)
  
  disney[j,"Review_Text"]<- paste((stemmed_Review[,"word"]),collapse = " ")
  
}


# remove duplicates
disney_unique <- disney[!duplicated(disney$Review_ID), ]

saveRDS(disney_unique, file = "disney_stemmed_new.RData")

##### PART 2

### MDS ###
# Create a document term matrix (DTM) of the reviews 
tokenized_reviews <- tokens(corpus(disney_unique, docid_field = "Review_ID", text_field = "Review_Text"))

# Create a sparse feature co-occurrence matrix fcm(): measures co-occurrences 
co_occurrence_matrix <- fcm(x = tokenized_reviews, context = "document", count = "frequency", tri=FALSE)

# Create a matrix with number of documents with each word on the diagonal
reviews_dfm <- dfm(tokenized_reviews) # get document frequency matrix
counts <- colSums(as.matrix(reviews_dfm)) 
co_occurrence_matrix <- as.matrix(co_occurrence_matrix)
diag(co_occurrence_matrix) <- counts

sortedcount <- counts%>% sort(decreasing=TRUE)
sortednames <- names(sortedcount)

# Create co-occurrence matrix showing first 15 rows and columns
co_occurrence_matrix <- co_occurrence_matrix[sortednames,sortednames]
co_occurrence_matrix[1:7,1:7]

# Convert similarities to distances
distances <- sim2diss(co_occurrence_matrix, method = "cooccurrence") 
distances[1:20,1:7]

# Run the routine that finds the best matching coordinates in a 2D mp given the distances
MDS_map <- smacofSym(distances) # Transform similarity values to distances
# Plot words in a map based on the distances
ggplot(as.data.frame(MDS_map$conf), aes(D1, D2, label = rownames(MDS_map$conf))) +
  geom_text(check_overlap = TRUE) + theme_minimal(base_size = 15) + xlab('') + ylab('') +
  scale_y_continuous(breaks = NULL) + scale_x_continuous(breaks = NULL)
# the conf element in the MDS output contains the coordinates with as names D1 and D2.

### PCA ###

# Cast the data frame to a TermDocumentMatrix (TDM)
review_tdm <- disney_unique %>% unnest_tokens(word,Review_Text) %>%count(word,Review_ID,sort=TRUE) %>%ungroup()%>%cast_tdm(word,Review_ID,n)

counts <- rowSums(as.matrix(review_tdm)) 
sortedcount <- counts%>% sort(decreasing=TRUE)
nwords<-200
sortednames <- names(sortedcount[1:nwords])

review_dtm <- t(review_tdm)

pca_results <- prcomp(review_dtm, scale = FALSE, rank. = 40) # rank specifies maximum number of principal components
pca_results_backup <- pca_results

fviz_screeplot(pca_results,ncp=40)

ncomp<-4

j<-1 # For first dimension
toplist <- abs(pca_results$rotation[,j]) %>% sort(decreasing=TRUE) %>% head(10)
topwords <- (names(toplist)) # Save most important words
for (j in 2:ncomp){
  toplist <- abs(pca_results$rotation[,j]) %>% sort(decreasing=TRUE) %>% head(10)
  topwords <-cbind( topwords , (names(toplist))) # Add most important words of the dimension
}

# Display
topwords

# Get factor loadings (relatedness) and do rotation to new coordinate system
rawLoadings     <- pca_results$rotation[sortednames,1:ncomp] %*% diag(pca_results$sdev, ncomp, ncomp)
rotated <- varimax(rawLoadings) # rotate loading matrix
pca_results$rotation <- rotated$loadings 
pca_results$x <- scale(pca_results$x[,1:ncomp]) %*% rotated$rotmat 

j<-1 # For first dimension
toplist <- abs(pca_results$rotation[,j]) %>% sort(decreasing=TRUE) %>% head(10)
topwords <- (names(toplist)) # Save most important words
for (j in 2:ncomp){
  toplist <- abs(pca_results$rotation[,j]) %>% sort(decreasing=TRUE) %>% head(10)
  topwords <-cbind( topwords , (names(toplist))) # Add most important words of the dimension
}

# Display
topwords

# Use smaller dataset: Keep only 200 reviews to plot
pca_results_backup <- pca_results 
pca_results_small <- pca_results
pca_results_small$x <- pca_results_small$x[sample(nrow(pca_results_small$x), 200),] 
pca_results <- pca_results_small
#pca_results <- pca_results_backup

# Plot PCA results after rotation - variables
axeslist <- c(1, 2)
fviz_pca_var(pca_results, axes=axeslist 
             ,geom.var = c("arrow", "text")
             ,col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"), # colors to use
             repel = TRUE     # Avoid text overlapping
)

# Plot PCA results after rotation - individuals
axeslist=c(1,2)
fviz_pca_ind(pca_results, axes = axeslist,
             col.ind = "cos2", # Color by the quality of representation
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),  # colors to use
             repel = FALSE,     # text can be overlapping
             #             geom = "point" # shows only points and no lables
)


# Plot PCA results in Biplot
axeslist=c(3,4)
biplot34 <- fviz_pca_biplot(pca_results, repel = TRUE, axes=axeslist,
                            col.ind = "cos2" # Color by the quality of representation
                            ,col.var = "contrib" # Color by contributions to the PC
                            ,geom = "point" # shows only points and no labels
)

saveRDS(biplot34, file = "Biplot34")

# first restore full set of reviews
pca_results_backup -> pca_results 
pca_sample <- sample(nrow(pca_results$x), 500)
pca_results$x <- pca_results$x[pca_sample,] 
groups <- as.factor(disney_unique[pca_sample, 'Rating' ])
groups_34 <- fviz_pca_ind(pca_results, axes=c(3,4), # Different dimensions
                          col.ind = groups, # color by groups
                          #             palette = c("#00AFBB",  "#FC4E07"),
                          addEllipses = TRUE, # Concentration ellipses (draws ellipses around the individuals)
                          ellipse.type = "confidence",
                          legend.title = "Groups",
                          repel = TRUE
                          ,               geom = "point" # shows only points and no labels
)

saveRDS(groups_34, file = "groups34")

##### PART 3

# Load data
disney_part34 <- read.csv(file = "DisneylandReviews.csv", sep = ",", header = TRUE)
disney_part34$Branch <- as.factor(disney_part34$Branch)

disney_part34$Description <- disney_part34$Review_Text

disney_part34 <- disney_part34[1:2500, ]

# - - - CREATE DATASET WITHOUT STEMMING AND INCLUDING PUNCTUATION

# Remove error messages in text
disney_part34 <- disney_part34[nchar(disney_part34$Description) > 50, ]

# Convert date to date object
disney_part34$Year_Month <- as.Date(paste0(disney_part34$Year_Month, "-01"))

Sys.setlocale('LC_ALL','C') # set aspects of the internationalization of the program

reviews_df <- disney_part34 %>%
  rename(
    Review = Review_Text
  )

# Replace numbers and certain characters
reviews_df$Description <- as.character(reviews_df$Description)  %>% 
  tolower() %>% 
  {gsub(":( |-|o)*\\("," SADSMILE ", .)} %>%        # Find :( or :-( or : ( or :o(
  {gsub(":( |-|o)*\\)"," HAPPYSMILE ", .)} %>%      # Find :) or :-) or : ) or :o)
  {gsub("(\"| |\\$)-+\\.-+"," NUMBER", .)} %>%      # Find numbers
  {gsub("([0-9]+:)*[0-9]+ *am"," TIME_AM", .)} %>%  # Find time AM
  {gsub("([0-9]+:)*[0-9]+ *pm"," TIME_PM", .)} %>%  # Find time PM
  {gsub("-+:-+","TIME", .)} %>%                     # Find general time
  {gsub("\\$ ?[0-9]*[\\.,]*[0-9]+"," DOLLARVALUE ", .)} %>%   # Find Dollar values
  {gsub("[0-9]*[\\.,]*[0-9]+"," NUMBER ", .)} %>%  # Find remaining numbers
  {gsub("-"," ", .)} %>%                           # Remove all -
  {gsub("&"," and ", .)} %>%                       # Find general time
  {gsub("\"+"," ", .)} %>%                         # Remove all "
  {gsub("\\|+"," ", .)} %>%                        # Remove all |
  {gsub("_+"," ", .)} %>%                          # Remove all _
  {gsub(";+"," ", .)} %>%                          # Remove excess ;
  {gsub(" +"," ", .)} %>%                          # Remove excess spaces
  {gsub("\\.+","\\.", .)}                          # Remove excess .


### - - - DATASET CREATED


### - - - OBTAIN TOTAL SENTIMENT - Frequency of positive/negative words (after removing stopwords)

# Remove stopwords from Review_text and count the number of times each word occurs
all_words <- reviews_df[,] %>%
  unnest_tokens("Description", output = "word") %>%
  anti_join(stop_words, by = "word") %>% # Remove stopwords
  count(word, sort = TRUE) %>% # count number of times each word occurs
  filter(n>100) # only do for words that occur more than 100 times

# Determine sentiment score
all_words$sentiment <- sentiment(all_words$word , lexicon::hash_sentiment_huliu)$sentiment


# Create plot
all_words %>%
  filter(n > 100) %>% # only do for words that occur more than 250 times
  filter(sentiment != 0) %>% # remove words with a sentiment score of 0
  mutate(n = ifelse(sentiment < 0, -n, n)) %>% # create negative count for words with negative sentiment score
  mutate(word = reorder(word, n)) %>% # apply sorting 
  mutate(Sentiment = ifelse(sentiment >0 , "Postive","Negative")) %>% # give labels to positive and negative sentiment scores
  ggplot(aes(word, n, fill = Sentiment)) + # Create a bar plot
  geom_col() +
  coord_flip() +
  labs(y = "Contribution to \"total\" sentiment", x = "Word (min freq = 250)")



### - - - SENTIMENT PER SENTENCE

# Cut the reviews into list of sentences
sentences <- get_sentences(reviews_df$Description)

# Determine sentiment scores of the sentences 
sentence_scores <- sentiment(get_sentences(reviews_df$Description), amplifier.weight = 0.0, but.weight = 0.0)

# Make a dataframe of all sentences
all_sentences <- as.data.frame(unlist(sentences[])) 
colnames(all_sentences) = "sentences" # Give name to the column

# Add sentiment score to the sentences
all_sentences$sentiment <- sentence_scores$sentiment 

# Give sentences an id
all_sentences$sentence_id <- c(1:dim(all_sentences)[1]) 




### - - - CREATE SENTIMENT FUNCTION
# Obtain SENTIMENT per REVIEW
reviews_df$sentiment <- sentiment_by(reviews_df$Description,  amplifier.weight = 0.0, but.weight = 0.0, n.before = 4, lexicon::hash_sentiment_huliu)$ave_sentiment

Figure_2 <- ggplot(reviews_df, aes(x = sentiment, y = ..density..)) + theme_gdocs() +
  geom_histogram(binwidth = .25, fill = "darkred", colour = "grey", size = .2) + 
  geom_density(size = 0.75) + ggtitle("Figure 2: Sentiment Distribution")
saveRDS(Figure_2, "Figure_2.rds")

pos_comments <- subset(reviews_df$Description, reviews_df$sentiment > 0)
neg_comments <- subset(reviews_df$Description, reviews_df$sentiment < 0)

pos_terms <- paste(pos_comments,collapse = " ")
neg_terms <- paste(neg_comments,collapse = " ")
all_terms <- c(pos_terms,neg_terms)
all_corpus <- VCorpus(VectorSource(all_terms))

# TDM using common preprocessing steps, explore the impact of matrix weighting
sentiment <- TermDocumentMatrix(all_corpus, control=
                                  list(weighting=weightTfIdf, removePunctuation = TRUE,
                                       stopwords=stopwords(kind='en')))
sentiment_matrix <- as.matrix(sentiment)
colnames(sentiment_matrix) <- c('positive','negative')

# Construct comparison cloud, use scaling of the polarity scores
comparison.cloud(sentiment_matrix, max.words=30,
                 colors=c('darkgreen','darkred'))



### EXTRA
# Get positive reviews: reviews with rating 4 or higher
positive <- reviews_df %>% filter(reviews_df[,"Rating"] >= 4)

# Get negative reviews: reviews with rating 2 or lower
negative <- reviews_df %>% filter(reviews_df[,"Rating"] <=2)

# Determine the average sentiment scores in positive reviews and negative reviews
cat("sentiment", mean(negative[, "sentiment"]), " ", mean(positive[, "sentiment"]), "\n")


Figure_1 <- ggplot( ,aes(sentiment)) +
  geom_density(aes(fill = "Happy"),   data = positive, alpha = 0.5) +
  geom_density(aes(fill = "Unhappy"), data = negative, alpha = 0.5) +
  scale_colour_manual("Sentiment", values = c("green", "red"), aesthetics = "fill")
+ ggtitle("Figure 1: Rating Distribution")
saveRDS(Figure_1, file = 'Figure_1.rds')





### - - - CREATE POLARITY FUNCTION - Approximate the sentiment (polarity) of text by grouping variable(s).
# Experiment with the settings of the negation.list and amplification.list arguments to identify how this affects 
# the identified sentiment

data("negation.words")
data("amplification.words")
data("deamplification.words")

negation.adj <- c("ain't", "aren't", "can't", "couldn't", "didn't", "doesn't")
amplification.adj <- c("acute", "acutely", "certain", "certainly", "colossal", "colossally")



# Adjusted negation words, what happens with the identified sentiment?
d_polarity_negation <- polarity(reviews_df$Description, negators = negation.adj)
Figure_3 <- ggplot(d_polarity_negation$all, aes(x = polarity, y = ..density..)) + theme_gdocs() +
  geom_histogram(binwidth = .25, fill = "darkred", colour = "grey", size = .2) + 
  geom_density(size = 0.75) + ggtitle("Figure 3: Adjusted Negators")
saveRDS(Figure_3, file = "Figure_3.rds")

# Adjusted amplification words, what happens with the identified sentiment?
d_polarity_ampl <- polarity(reviews_df$Description, amplifiers = amplification.adj)
Figure_4 <- ggplot(d_polarity_ampl$all, aes(x = polarity, y = ..density..)) + theme_gdocs() +
  geom_histogram(binwidth = .25, fill = "darkred", colour = "grey", size = .2) + 
  geom_density(size = 0.75) + ggtitle("Figure 4: Adjusted Amplifiers")
saveRDS(Figure_4, file = 'Figure_4.rds')

# Polarity with DEFAULT settings / complete lists 
d_polarity <- polarity(reviews_df$Description)
ggplot(d_polarity$all, aes(x = polarity, y = ..density..)) + theme_gdocs() +
  geom_histogram(binwidth = .25, fill = "darkred", colour = "grey", size = .2) + 
  geom_density(size = 0.75)


# Polarity scores are not centered at 0 which means that on average each review 
# has at least one positive word in it, there is a social norm for people to be nice
# acknowledge effort, or find sth to be positive about --> grade inflation = people are
# mixing positive words alongside negative ones 


# If you want the comparison cloud not to be scaled, remove "scale()" but the output will be less informative. 
reviews_df$polarity <- scale(d_polarity$all$polarity)

pos_comments <- subset(reviews_df$Description, reviews_df$polarity > 0)
neg_comments <- subset(reviews_df$Description, reviews_df$polarity < 0)



# Collapse pos_comments and neg_comments into 2 distinct documents. This creates a single
# character vector = all.terms. Then VCorpus contains 2 documents with all the positive
# and negative reviews

pos_terms <- paste(pos_comments,collapse = " ")
neg_terms <- paste(neg_comments,collapse = " ")
all_terms <- c(pos_terms,neg_terms)
all_corpus <- VCorpus(VectorSource(all_terms))

# TDM using common preprocessing steps, explore the impact of matrix weighting
all_tdm <- TermDocumentMatrix(all_corpus, control=
                              list(weighting=weightTfIdf, removePunctuation = TRUE,
                                   stopwords=stopwords(kind='en')))
all_tdm_matrix <- as.matrix(all_tdm)
colnames(all_tdm_matrix) <- c('positive','negative')

# Construct comparison cloud, use scaling of the polarity scores 
png("Figure_5.png", width = 800, height = 600)
Figure_5 <- comparison.cloud(all_tdm_matrix, max.words=25,
                 colors=c('darkgreen','darkred'), main = "Figure 5: Word Cloud of Polarity")
dev.off()

# Final results!
# Compare the previous output to the polarity function where the negation words 
# and amplification words have been adjusted














# Classify emotions
# Anger, fear, sadness, disgust, surprise, joy
# Main function in the package calculates a score for each of these emotional states and then selects
# the specific emotion with the highest score from the 6


# Creating NRC emotion keys
nrc_sentiment_key <- syuzhet:::nrc %>%
  dplyr::filter(
    sentiment %in% c('positive', 'negative'),
    lang == 'english'
  ) %>% # Use positive and negative sentiments
  dplyr::select(-lang) %>% # Select language
  mutate(value = ifelse(sentiment == 'negative', value * -1, value)) %>% # Multiply negative sentiment scores by -1
  dplyr::group_by(word) %>%
  dplyr::summarize(y = mean(value)) %>% # Get mean
  sentimentr::as_key()

nrc_anger_key <- syuzhet:::nrc %>%
  dplyr::filter(
    sentiment %in% c('anger'),
    lang == 'english'
  ) %>% # Use sentiments: anger
  dplyr::select(-lang) %>% # Select language
  mutate(value = ifelse(sentiment == 'negative', value * -1, value)) %>%
  dplyr::group_by(word) %>%
  dplyr::summarize(y = mean(value)) %>% # Get mean
  sentimentr::as_key()

nrc_joy_key <- syuzhet:::nrc %>%
  dplyr::filter(
    sentiment %in% c('joy'),
    lang == 'english'
  ) %>% # Use sentiments: joy
  dplyr::select(-lang) %>% # Select language
  mutate(value = ifelse(sentiment == 'negative', value * -1, value)) %>%
  dplyr::group_by(word) %>%
  dplyr::summarize(y = mean(value)) %>% # Get mean
  sentimentr::as_key()




# Get emotion classification of all reviews
reviews_classified <- as.data.frame(get_nrc_sentiment(reviews_df[,"Description"]))
reviews_classified[,"Description"] <- reviews_df[,"Description"]

max_col_index <- max.col(reviews_classified[, 1:8])
max_col_name <- colnames(reviews_classified)[max_col_index]
reviews_classified$best_fit <- max_col_name


ggplot(reviews_classified, aes(x=best_fit)) +
  geom_bar(aes(y=..count.., fill=best_fit)) +
  labs(x="emotion categories",
       y="Disneyland HongKong")+theme_gdocs() +
  theme(legend.position="none")


reviews_df$emotions <-(reviews_classified$best_fit)
emotion.reviews <-split(reviews_df$Description,
                        reviews_df$emotions)
emotion.reviews <-lapply(emotion.reviews,
                         paste,collapse=" ")
emotion.reviews <-do.call(c,emotion.reviews)
emotion.reviews <-VCorpus(VectorSource(
  emotion.reviews))
all.tdm <-TermDocumentMatrix(emotion.reviews,control=
                               list(weighting=weightTfIdf, removePunctuation = TRUE,
                                    stopwords=stopwords(kind='en')))

all.tdm.m <- as.matrix(all.tdm)
reviews_df$emotions <- factor(reviews_df$emotions)
colnames(all.tdm.m) <- levels(reviews_df$emotions)

png("Figure_6.png", width = 800, height = 600)
Figure_6 <- comparison.cloud(all.tdm.m, title.size = 1.5, title.bg.colors = "lightblue", main = "Word Cloud of Emotion Classification")
dev.off()

##### PART 4


######### NMF ################
#take subset for efficiency
saved <- disney_unique
disney_unique <- saved[1:2500,]

#Create a Corpus object and get DTM (and pre-process the text)
corpus <- Corpus(VectorSource(unlist(disney_unique[, "Review_Text"])))

# preprocess and create DTM
#Stemming and Stopwords removal has already been done
dtm <- DocumentTermMatrix(corpus, control = list(stemming = FALSE, stopwords = FALSE, minWordLength = 3, removeNumbers = TRUE, removePunctuation = TRUE))
dim(dtm)

#Show some statistics on the frequency of words, also count number of words with freq<25
summary(col_sums(dtm))
lowFreq <- (col_sums(dtm)<25)
sum(lowFreq)


#Filter low frequency words (and remove empty docs)
dtm <- dtm[, !lowFreq]
dtm <- dtm[row_sums(dtm) > 0,]
summary(col_sums(dtm))
n <- nrow(dtm)
dim(dtm)
#saved_dtm <- dtm
#dtm <- saved_dtm

# Get current data as a term-document-matrix
maxcols <- ncol(dtm)
V <- t(as.matrix(dtm[,1:maxcols]))  # NMF is usually defined on term-document-matrix

#Obtain NMF with different methods (for 2 factors) using smart starting values
nfactors <- 2
#nmf_multi <- nmf(V, nfactors, method=list("snmf/l","snmf/r","brunet","lee"), seed="nndsvd", .options='vt')
#saveRDS(nmf_multi, file = "nmf_multi")

nmf_multi <- readRDS(file = "nmf_multi")
plot(nmf_multi)
compare(nmf_multi)


#Compare fits of the different NMF solutions
for (m in nmf_multi){
  print(mean((V-fitted(m))^2))
  plot(m)
} 

#Use ood method with different ranks
#nmf_rank = list()
#for (n in seq(2,14,2)){
#  nmf <- nmf(V, rank=n, method=list("lee"), seed="nndsvd", .options='vt')
#  fit <- fitted(nmf)
#  mean(abs(V-fit))
#  nmf_rank = append(nmf_rank, nmf)
#}

#saveRDS(nmf_rank, file = "nmf_rank")

nmf_rank <- readRDS(file = "nmf_rank")

compare(nmf_rank)


#Make plot of error as a function of the number of factors (error of course decreases)
r = c()
error = c()
for (m in nmf_rank){
  r = rbind(r, ncol(m@fit@W))
  error = rbind(error, mean((V-fitted(m))^2))
}
plot(r,error)

#Conclusion: after the first 7 factors there is not much explanatory power left
plotC.1 <- svd(V)
saveRDS(plotC.1, file = "plotC1")
plot(plotC1$d[1:40])

#Choose same method Lee and use rank. Set amount of Factors to 7.
nfactors <- 7
#res <- nmf(V, rank=nfactors, method="lee", seed="nndsvd", .options='v')
#save_res <- res
#saveRDS(save_res, file = "save_res")
save_res <- readRDS("save_res")
res <- save_res
W <- basis(res)
H <- coef(res)

#Visualize the basis and coefficient matrices
heatmap(W, Colv=NA, main="W",scale="row")
heatmap(H, Rowv=NA, main="H",scale="column")

#Show W and H for a selection of documents and terms

heatmap(H[,1:15], Colv=NA, Rowv=NA, main="H[,1:15]",scale="column")


heatmap(W[apply(W, 1, max) > 0,], Colv=NA, main="W, selected words with highest scores",scale="none")

#Select terms loading high for each factor
relW <- W/(rep(1,nrow(W))%*%t(col_sums(W)))
Wtable <- data.frame(topic= c(1:nfactors), t(relW))
Wtable <- gather(Wtable, term, score, -topic)

text_top_terms <- Wtable %>%
  group_by(topic) %>%
  top_n(10, score) %>%
  ungroup() %>%
  arrange(topic, -score)

plot_topics <- text_top_terms %>%
  filter(topic <= 7) %>%
  mutate(term = reorder_within(term, score, topic)) %>%
  ggplot(aes(term, score, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip()+
  scale_x_reordered()

#saveRDS(plot_topics, file = "PlotC.2")
#Code to find (rescaled) topic weights per document
docScores <- H/(rep(1,nfactors) %*% t(col_sums(H)))
docScores[,1:10]
