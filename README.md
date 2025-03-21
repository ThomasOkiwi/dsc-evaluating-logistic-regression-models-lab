## Evaluation Metrics
  
## Introduction
In this lesson, you'll learn about common evaluation metrics used to quantify the performance of classifiers!
## Objectives
You will be able to:
•	Evaluate classification models using the evaluation metrics appropriate for a specific problem
•	Define precision and recall
•	Define accuracy and F1 score
## Evaluation metrics for classification
Now that we've started discussing classification, it's time to examine comparing models to one other and choosing the models that have the best fit. Previously in regression, you were predicting values so it made sense to discuss error as a distance of how far off the estimates were from the actual values. However, in classifying a binary variable you are either correct or incorrect. As a result, we tend to deconstruct this as how many false positives versus false negatives there are in a model. In particular, there are a few different specific measurements when evaluating the performance of a classification algorithm.
Let's work through these evaluation metrics to understand what each metric tells us.
## Precision and recall
Precision and Recall are two of the most basic evaluation metrics available to us. Precision measures how precise the predictions are, while Recall indicates what percentage of the classes we're interested in were actually captured by the model.

 ![image](https://github.com/user-attachments/assets/ab00ebf0-f244-4f33-b97e-d9a5ea43dc67)

## Precision
The following formula shows how to use information found in a confusion matrix to calculate the precision of a model:
Precision=Number of True PositivesNumber of Predicted Positives
To reuse a previous analogy of a model that predicts whether or not a person has a certain disease, precision allows us to answer the following question:
"Out of all the times the model said someone had a disease, how many times did the patient in question actually have the disease?"
Note that a high precision score can be a bit misleading. For instance, let's say we take a model and train it to make predictions on a sample of 10,000 patients. This model predicts that 6000 patients have the disease when in reality, only 5500 have the disease. This model would have a precision of 91.6%. Now, let's assume we create a second model that only predicts that a person is sick when it's incredibly obvious. Out of 10,000 patients, this model only predicts that 5 people in the entire population are sick. However, each of those 5 times, it is correct. model 2 would have a precision score of 100%, even though it missed 5,495 cases where the patient actually had the disease! In this way, more conservative models can have a high precision score, but this doesn't necessarily mean that they are the best performing model!
## Recall
The following formula shows how we can use the information found in a confusion matrix to calculate the recall of a model:
Recall=Number of True PositivesNumber of Actual Total Positives
Following the same disease analogy, recall allows us to ask:
"Out of all the patients we saw that actually had the disease, what percentage of them did our model correctly identify as having the disease?"
Note that recall can be a bit of a tricky statistic because improving our recall score doesn't necessarily always mean a better model overall. For example, our model could easily score 100% for recall by just classifying every single patient that walks through the door as having the disease in question. Sure, it would have many False Positives, but it would also correctly identify every single sick person as having the disease!
## The relationship between precision and recall
As you may have guessed, precision and recall have an inverse relationship. As our recall goes up, our precision will go down, and vice versa. If this doesn't seem intuitive, let's examine this through the lens of our disease analogy.
A doctor that is overly obsessed with recall will have a very low threshold for declaring someone as sick because they are most worried about sick patients. Their precision will be quite low, because they classify almost everyone as sick, and don't care when they're wrong -- they only care about making sure that sick people are identified as sick.
A doctor that is overly obsessed with precision will have a very high threshold for declaring someone as sick, because they only declare someone as sick when they are completely sure that they will be correct if they declare a person as sick. Although their precision will be very high, their recall will be incredibly low, because a lot of people that are sick but don't meet the doctor's threshold will be incorrectly classified as healthy.
## Which metric is better?
A classic Data Science interview question is to ask "What is better -- more false positives, or false negatives?" This is a trick question designed to test your critical thinking on the topics of precision and recall. As you're probably thinking, the answer is "It depends on the problem!". Sometimes, our model may be focused on a problem where False Positives are much worse than False Negatives, or vice versa. For instance, detecting credit card fraud. A False Positive would be when our model flags a transaction as fraudulent, and it isn't. This results in a slightly annoyed customer. On the other hand, a False Negative might be a fraudulent transaction that the company mistakenly lets through as normal consumer behavior. In this case, the credit card company could be on the hook for reimbursing the customer for thousands of dollars because they missed the signs that the transaction was fraudulent! Although being wrong is never ideal, it makes sense that credit card companies tend to build their models to be a bit too sensitive, because having a high recall saves them more money than having a high precision score.
Take a few minutes and see if you can think of at least two examples each of situations where a high precision might be preferable to high recall, and two examples where high recall might be preferable to high precision. This is a common interview topic, so it's always handy to have a few examples ready!
## Accuracy and F1 score
The two most informative metrics that are often cited to describe the performance of a model are Accuracy and F1 score. Let's take a look at each and see what's so special about them.
## Accuracy
Accuracy is probably the most intuitive metric. The formula for accuracy is:
Accuracy=Number of True Positives + True NegativesTotal Observations
Accuracy is useful because it allows us to measure the total number of predictions a model gets right, including both True Positives and True Negatives.
Sticking with our analogy, accuracy allows us to answer:
"Out of all the predictions our model made, what percentage were correct?"
Accuracy is the most common metric for classification. It provides a solid holistic view of the overall performance of our model.
## F1 score
The F1 score is a bit more tricky, but also more informative. F1 score represents the Harmonic Mean of Precision and Recall. In short, this means that the F1 score cannot be high without both precision and recall also being high. When a model's F1 score is high, you know that your model is doing well all around.
The formula for F1 score is:
F1 score=2 Precision x RecallPrecision+Recall
To demonstrate the effectiveness of F1 score, let's plug in some numbers and compare F1 score with a regular arithmetic average of precision and recall.
Let's assume that the model has 98% recall and 6% precision.
Taking the arithmetic mean of the two, we get: 0.98+0.062=1.042=0.52
However, using these numbers in the F1 score formula results in:
F1 score=20.98∗0.060.98+0.06=20.05881.04=2(0.061152)=0.122304
or 12.2%!
As you can see, F1 score penalizes models heavily if it skews too hard towards either precision or recall. For this reason, F1 score is generally the most used metric for describing the performance of a model.
## Which metric to use?
The metrics that are most important to a project will often be dependent on the business use case or goals for that model. This is why it's very important to understand why you're doing what you're doing, and how your model will be used in the real world! Otherwise, you may optimize your model for the wrong metric!
In general, it is worth noting that it's a good idea to calculate all relevant metrics, when in doubt. In most classification tasks, you don't know which model will perform best when you start. The common workflow is to train each different type of classifier, and select the best by comparing the performance of each. It's common to make tables like the one below, and highlight the best performer for each metric:
 
## Calculate evaluation metrics with confusion matrices
Note that we can only calculate any of the metrics discussed here if we know the True Positives, True Negatives, False Positives, and False Negatives resulting from the predictions of a model. If we have a confusion matrix, we can easily calculate Precision, Recall and Accuracy -- and if we know precision and recall, we can easily calculate F1 score!
## Classification reports
Scikit-learn has a built-in function that will create a Classification Report. This classification report even breaks down performance by individual class predictions for your model. You can find the classification_report() function in the sklearn.metrics module, which takes labels and predictions and returns the precision, recall, F1 score and support (number of occurrences of each label in y_true) for the results of a model.
## Summary
In this lesson you were introduced to several metrics which can be used to evaluate classification models. In the following lab, you'll write functions to calculate each of these manually, as well as explore how you can use existing functions in scikit-learn to quickly calculate and interpret each of these metrics.



# Evaluating Logistic Regression Models - Lab

## Introduction

In regression, you are predicting continuous values so it makes sense to discuss error as a distance of how far off our estimates were. When classifying a binary variable, however, a model is either correct or incorrect. As a result, we tend to quantify this in terms of how many false positives versus false negatives we come across. In particular, we examine a few different specific measurements when evaluating the performance of a classification algorithm. In this lab, you'll review precision, recall, accuracy, and F1 score in order to evaluate our logistic regression models.


## Objectives 

In this lab you will: 

- Implement evaluation metrics from scratch using Python 



## Terminology review  

Let's take a moment and review some classification evaluation metrics:  


$$ \text{Precision} = \frac{\text{Number of True Positives}}{\text{Number of Predicted Positives}} $$    

$$ \text{Recall} = \frac{\text{Number of True Positives}}{\text{Number of Actual Total Positives}} $$  
  
$$ \text{Accuracy} = \frac{\text{Number of True Positives + True Negatives}}{\text{Total Observations}} $$

$$ \text{F1 score} = 2 * \frac{\text{Precision * Recall}}{\text{Precision + Recall}} $$


At times, it may be best to tune a classification algorithm to optimize against precision or recall rather than overall accuracy. For example, imagine the scenario of predicting whether or not a patient is at risk for cancer and should be brought in for additional testing. In cases such as this, we often may want to cast a slightly wider net, and it is preferable to optimize for recall, the number of cancer positive cases, than it is to optimize precision, the percentage of our predicted cancer-risk patients who are indeed positive.

## Split the data into training and test sets


```python
import pandas as pd
df = pd.read_csv('heart.csv')
df.head()
```

Split the data first into `X` and `y`, and then into training and test sets. Assign 25% to the test set and set the `random_state` to 0. 


```python
# Import train_test_split


# Split data into X and y
y = None
X = None

# Split the data into a training and a test set
X_train, X_test, y_train, y_test = None
```

## Build a vanilla logistic regression model

- Import and instantiate `LogisticRegression` 
- Make sure you do not use an intercept term and use the `'liblinear'` solver 
- Fit the model to training data


```python
# Import LogisticRegression


# Instantiate LogisticRegression
logreg = None

# Fit to training data
model_log = None
model_log
```

## Write a function to calculate the precision


```python
def precision(y, y_hat):
    # Your code here
    pass
```

## Write a function to calculate the recall


```python
def recall(y, y_hat):
    # Your code here
    pass
```

## Write a function to calculate the accuracy


```python
def accuracy(y, y_hat):
    # Your code here
    pass
```

## Write a function to calculate the F1 score


```python
def f1_score(y, y_hat):
    # Your code here
    pass
```

## Calculate the precision, recall, accuracy, and F1 score of your classifier 

Do this for both the training and test sets. 


```python
# Your code here
y_hat_train = None
y_hat_test = None
```

Great job! Now it's time to check your work with `sklearn`. 

## Calculate metrics with `sklearn`

Each of the metrics we calculated above is also available inside the `sklearn.metrics` module.  

In the cell below, import the following functions:

* `precision_score`
* `recall_score`
* `accuracy_score`
* `f1_score`

Compare the results of your performance metrics functions above with the `sklearn` functions. Calculate these values for both your train and test set. 


```python
# Your code here
```

Nicely done! Did the results from `sklearn` match that of your own? 

## Compare precision, recall, accuracy, and F1 score for train vs test sets

Calculate and then plot the precision, recall, accuracy, and F1 score for the test and training splits using different training set sizes. What do you notice?


```python
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
training_precision = []
testing_precision = []
training_recall = []
testing_recall = []
training_accuracy = []
testing_accuracy = []
training_f1 = []
testing_f1 = []

for i in range(10, 95):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= None) # replace the "None" here
    logreg = LogisticRegression(fit_intercept=False, C=1e20, solver='liblinear')
    model_log = None
    y_hat_test = None
    y_hat_train = None 
    
    # Your code here

```

Create four scatter plots looking at the train and test precision in the first one, train and test recall in the second one, train and test accuracy in the third one, and train and test F1 score in the fourth one. 

We already created the scatter plot for precision: 


```python
# Train and test precision
plt.scatter(list(range(10, 95)), training_precision, label='training_precision')
plt.scatter(list(range(10, 95)), testing_precision, label='testing_precision')
plt.legend()
plt.show()
```


```python
# Train and test recall
```


```python
# Train and test accuracy
```


```python
# Train and test F1 score
```

## Summary

Nice! In this lab, you calculated evaluation metrics for classification algorithms from scratch in Python. Going forward, continue to think about scenarios in which you might prefer to optimize one of these metrics over another.
