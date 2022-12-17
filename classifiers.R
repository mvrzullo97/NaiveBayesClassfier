#--- Libraries

library(tm)
library(openNLP)
library(NLP)
library(textstem)
library(naivebayes)
library(caret)

set.seed(1)
# loading dataset
Data <- read.csv(file="insert_path/emails.csv")
colnames(Data)<-c("text","label")

# choose the algortihms
multinomial <- TRUE
bernoulli <- TRUE
lab0 <- "ham"
lab1 <- "spam"
c0 <- lab0
c1 <- lab1

Data$label[which(Data$label == 0)] <- c0
Data$label[which(Data$label == 1)] <- c1
table(Data$label)

N<-dim(Data)[1]

# define a subset of dataset
n <- 1000
subs<-sort(sample(1:N,n,replace=FALSE))
X<-Data[subs,]

p_spam<-length(which(X$label=="spam"))/n
p_ham<-1-p_spam

# pre-processing step for data clean-up
Stopwords_Data<-read.csv(file="insert_path/stopwords.csv")
colnames(Stopwords_Data)<-c("stopwords")
removed_words<-Stopwords_Data$stopwords

corpus<-Corpus(VectorSource(X$text))
corpus[]$content<-lemmatize_strings(corpus[]$content)
corpus <- tm_map(corpus, tolower)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, removed_words)
corpus <- tm_map(corpus, stripWhitespace)

label <- X$label
clean_text <- corpus[]$content
clean_data <- data.frame(label,clean_text) 


# from corpus to Document Term Matrix
Xdtm <- DocumentTermMatrix(corpus)
dim(Xdtm)
nwrd<-ncol(Xdtm)

# Training and Test sets
perc <- 70
ntrn<-floor((n*perc)/100)
train<-sort(sample(1:n, ntrn, replace=FALSE))

Xtrain_dtm<-Xdtm[train,]
Xtest_dtm<-Xdtm[-train,]
dim(Xtrain_dtm)
dim(Xtest_dtm)

# to extract all terms in all documents
min_occ <- 0
dictio0<-findFreqTerms(Xtrain_dtm, min_occ)

dictio<-dictio0
a<-corpus[train]
Train_dtm<-DocumentTermMatrix(a, control = list(dictionary = dictio))
a<-corpus[-train]
dim(Train_dtm)
Test_dtm<-DocumentTermMatrix(a, control = list(dictionary = dictio))
dim(Test_dtm)

# from occurrences to presence/absence 
Train_bin<-ceiling(as.matrix(Train_dtm)/max(as.matrix(Train_dtm)))
Test_bin<-ceiling(as.matrix(Test_dtm)/max(as.matrix(Test_dtm)))

i_c1<-which(X$label[train]==lab1)
i_c0<-which(X$label[train]==lab0)

Train_c1<-Train_bin[i_c1,]
colnames(Train_c1)<-dictio0
row.names(Train_c1)<-i_c1

Train_c0<-Train_bin[i_c0,]
colnames(Train_c0)<-dictio0
row.names(Train_c0)<-i_c0

# Prior e likelihood
Sum_c1 <- colSums(Train_c1)
Sum_c0 <- colSums(Train_c0)
n_c1<-length(i_c1)
n_c0<-length(i_c0)
pi_c1<-n_c1/ntrn
pi_c0<-n_c0/ntrn

# define Laplace smoothing 
alfa<-0.001

# calculate the conditional probabilities
prob_c1<-(Sum_c1+alfa)/(n_c1+2*alfa)
prob_c0<-(Sum_c0+alfa)/(n_c0+2*alfa)
df_prob<- data.frame(dictio0, Sum_c0, prob_c0, Sum_c1, prob_c1)
colnames(df_prob)<-c("word","n_c0","p_c0","n_c1","p_c1")
rownames(df_prob) <- c(1:ncol(Train_bin))

# log-ratio ->
lgpr <- abs(log10(prob_c0/prob_c1))
x<-lgpr

# to define a waste percent
q<-quantile(x,probs=seq(0,1,0.05))
lambda <- q[7]
rmvd<-which(x<=lambda)
dictio<-dictio0[-rmvd]

# dtm updated with new dictio
a<-corpus[train]
Train_dtm<-DocumentTermMatrix(a, control = list(dictionary = dictio))
a<-corpus[-train]
Test_dtm<-DocumentTermMatrix(a, control = list(dictionary = dictio))


label_tr <- factor(X[train,]$label)
label_test <- factor(X[-train,]$label)

if (bernoulli){
  Train_bin<-ceiling(as.matrix(Train_dtm)/max(as.matrix(Train_dtm)))
  Test_bin<-ceiling(as.matrix(Test_dtm)/max(as.matrix(Test_dtm)))
  out.nb<- bernoulli_naive_bayes(Train_bin, y=label_tr, laplace = alfa)
  summary(out.nb)
  pred.test<-predict(out.nb,newdata = Test_bin)
  true.class<-label_test
  A<-table(true.class, pred.test)
  bern_nb <- confusionMatrix(A)
} 

if (multinomial){
  Train_dtm <- as.matrix(Train_dtm)
  Test_dtm <- as.matrix(Test_dtm)
  out.multi<- multinomial_naive_bayes(Train_dtm, y=label_tr, laplace = alfa)
  summary(out.multi)
  pred.test<-predict(out.multi,newdata = Test_dtm)
  true.class<-label_test
  B<-table(true.class, pred.test)
  multi_nb <- confusionMatrix(B)
}

print(paste("Accuracy Bernoulli NB -> ",signif(bern_nb$overall[1],digits = 3), sep = ""))
print(paste("Accuracy Multinomial NB -> ",signif(multi_nb$overall[1],digits = 3), sep = ""))
