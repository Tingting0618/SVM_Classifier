# SVM and KNN Classifier

### Support Vector Machine (SVM) classifier
- SVMs are sensitive to the feature scales
- Unlike Logistic Regression classifiers, SVM classifiers do not output probabilities for each class.

  - Linear SVM Classification
    - hard margin classification (not working well when there are outliers) 
    - soft margin classification (not working well when classes are not linearly separable)
       - hyperparameters C is controling the balance between keeping the street as large as possible and limiting the margin violations. If overfit, then try to reduce C. 
  - Nonlinear SVM Classification
    - One approach to handling nonlinear datasets is to add more features, such as polynomial features; in some cases this can result in a linearly separable dataset.
    - That said, at a low polynomial degree, this method cannot deal with very complex datasets, and with a high polynomial degree it creates a huge number of features, making the model too slow.
  - Fortunately, the kernel trick makes it possible to get the same result as if you had added many polynomial features, even with very high-degree polynomials, without actually having to add them.
    - Polynomial Kernel
    - Gaussian RBF Kernel 

### KNeighbors (KNN) classifier

**Concept:**

- start with a dataset with known data/label 
- use PCA to project features into PC1 and PC2
- import a new data, count what the nearest k neighbors' clusters
- the highest vote will become the new cluster for that new obs

**Requirements for kNN**
- Generally, k gets decided on the square root of the number of data points. But a large k value has benefits which include reducing the variance due to the noisy data; the side effect being developing a bias due to which the learner tends to ignore the smaller patterns which may have useful insights
- Data Normalization - It is to transform all the feature data in the same scale (0 to 1) else it will give more weightage to the data which is higher in value irrespective of scale/unit.
- Installation of “Class” library to implement in R.
- More neighbors means less impacted by noise, but could miss smaller patterns/clusters.
- Parameter tuning (k, i.e., how many neighbors to look at) 

![Rplot](https://user-images.githubusercontent.com/44503223/128354662-9cbb8851-c82c-4ed2-a1f1-fe5a05abc3b5.png)

```r
# reading csv data files from defined directory as file has already downloaded and stored in the directory
gc <- read.csv("index.csv") 

## Taking back-up of the input file, in case the original data is required later
gc_bkup <- gc
head (gc) # To check top 6 values of all the variables in the data set.

# Age (numeric)
# Sex (text: male, female)
# Job (numeric: 0 - unskilled and non-resident, 1 - unskilled and resident, 2 - skilled, 3 - highly skilled)
# Saving accounts (text - little, moderate, quite rich, rich)
# Credit amount (numeric, in DM)
# Duration (numeric, in month)
# Purpose (text: car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others)

str(gc) 
# understanding data structure can see all the variables are integers including 'Creditability' which is our response variable.


gc_subset <- gc[c('Creditability','Age..years.','Sex...Marital.Status','Occupation','Account.Balance','Credit.Amount','Length.of.current.employment','Purpose')]
head(gc_subset)

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x))) } # creating a normalize function for easy convertion.

gc_subset_norm<- as.data.frame(lapply(gc_subset[,2:8], normalize)) # lapply creates list that is why it is converted to dataframe and it applies defined fundtion (which is 'normalize') to all the list values which is here column 2 to 8 as first column is target/response.
head(gc_subset_norm)

set.seed(123)  # To get the same random sample
random_sample <- sample(1:nrow(gc_subset_norm),size=nrow(gc_subset_norm)*0.7,replace = FALSE) #random selection of 70% data.

train_gc <- gc_subset_norm[random_sample,] # 70% training data
test_gc <- gc_subset_norm[-random_sample,] # remaining 30% test data

#Now creating seperate dataframe for 'Creditability' feature which is our target.
train_gc_labels <- gc_subset[random_sample,1]
test_gc_labels  <- gc_subset[-random_sample,1]   

#install.packages(class) # to install class packages as it carries kNN function
library(class)          # to call class package

NROW(train_gc_labels)

#To identify optimum value of k, generally square root of total no of observations (700) which is 26.45 is taken, so will try with 26, 27 then will check for the optimal value of k.

knn_26 <-  knn(train=train_gc, test=test_gc, cl=train_gc_labels, k=26)
knn_27 <-  knn(train=train_gc, test=test_gc, cl=train_gc_labels, k=27)

## Let's calculate the proportion of correct classification for k = 26, 27 

score_26 <- 100 * sum(test_gc_labels == knn_26)/NROW(test_gc_labels)  # For knn = 26
score_27 <- 100 * sum(test_gc_labels == knn_27)/NROW(test_gc_labels)  # For knn = 27

table(knn_26 ,test_gc_labels)
table(knn_27 ,test_gc_labels)

library(caret)
confusionMatrix(knn_26 ,as.factor(test_gc_labels))
confusionMatrix(knn_27 ,as.factor(test_gc_labels))

i=1                          # declaration to initiate for loop
score=1                     # declaration to initiate for loop
for (i in 1:28){ 
  knn_i <-  knn(train=train_gc, test=test_gc, cl=train_gc_labels, k=i)
  score[i] <- 100 * sum(test_gc_labels == knn_i)/NROW(test_gc_labels)
  k=i  
  cat(k,'=',score[i],'\n')       # to print % accuracy 
}

plot(score, type="b", xlab="k-value",ylab="Accuracy Level")  
# to plot % accuracy wrt to k-value

knn_7 <-  knn(train=train_gc, test=test_gc, cl=train_gc_labels, k=7)
confusionMatrix(knn_7 ,as.factor(test_gc_labels))
```

## Learn More

For more information, please check out the [Project Portfolio](https://tingting0618.github.io).

## Reference

This repo is my learning journal following:
- Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition, by Aurélien Géron (O’Reilly). Copyright 2019 Kiwisoft S.A.S., 978-1-492-03264-9.
- kNN(k-Nearest Neighbour) Algorithm in R, by Kumar Skand https://rstudio-pubs-static.s3.amazonaws.com/316172_a857ca788d1441f8be1bcd1e31f0e875.html
