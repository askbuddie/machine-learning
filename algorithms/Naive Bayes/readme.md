# Naive Bayes from Scratch 


1. Implemented a simple Naive Bayes classifier to detect the sentiment of product reviews. Created a 7-fold cross validation to train and test your model. All the stop words were discarded like commas, fullstops, numbers, hyphens, brackets, exclamation marks and any other single/double letter words (such as a, an, the, be etc) which do not contribute to the sentiment of the text.
2. Use of laplace smoothening to avoid the problem of divison by zero.

## Implementation

Here is a brief description of the steps performed to implement the Naive Bayes model to classify the sentiment of product reviews:

1. **Converting the text file**: The dataset text file is converted to a pandas dataframe. Each sentence in the dataset goes through some preprocessing steps.
2. **7-fold Cross Validation**: Next, the dataframe is divided into 7 folds. For each of the 7 iterations, one of the set is considered as the testing set and the rest of the data is used in training.
3. **Count Vectorization**: An array of unique words is derived from the training data. Each sentence in the training set is now fitted onto this array of unique words, giving is a matrix that will be used for training.
4. **Fitting and evaluating the model**: Once we have a matrix representation for our training and testing data, we can fit the data into our model and use the Bayes theorem formula to derive the probabilities that a certain product review in our testing set is of positive sentiment. The predictions are compared to the actual labels for the testing data and accuracy is reported.
5. **Deriving the final accuracy**: Once we have the accuracies for each of the 7 folds (testing sets), we can take their mean and report it. This will be the accuracy of our model.


#### Note
Since the training data had a lot of spelling mistakes, we included some commented code that would use the "autocorrect" Python library to fix spelling mistakes (commented because libraries other than Numpy and Pandas are not allowed). Doing so increases the accuracy of the model.

