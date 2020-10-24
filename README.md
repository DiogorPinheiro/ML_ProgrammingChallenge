# Programming Challenge for the KTH Course DD2421 - Machine Learning 

## The Problem
Welcome to the DD2421 ML Challenge. In short, you must build and train a classifier given a labeled dataset and then use it to infer the labels of a given unlabeled evaluation dataset. You must then submit the inferred labels in a specified format, which will be compared to the ground truth.

## The Code

### Data Analysis
Without any context, it is important to understand the data. For that, I used a heatmap diagram to check the correlation between features, which would be useful for dimensionality reduction in case some features are heavily correlated.

IMAGE

X3 and X4 have a strong negative correlation (almost opposites of one another), so one possible solution would be to remove one of them. However, the model accuracy wasn't really affected by this, as well as the processing time, so there was no advantage in removing it.

### Data Cleaning
The usual process, transform all nulls and '?' cells into NaN and replace them with the columns's mean value. For columns with a missing Boolean, it would be possible to apply a generative algorithm to "predict" this values but the easy solution was just to remove those rows (3/4 in total).

### Feature Engineering
Several approaches were considered, including applying a Deep Feature Synthesis (DFS). Using DFS with multiplication operation between columns, the number of features increased to 34. To reduce this number, Principal Component Analysis (PCA) was used but to no avail.
All in all, no changes in the features were made.

### Models
The final model was a Voting Classifier, which combined the classifiers: LightGBM, ExtraTrees and XGBoost.

| Classifier        | Accuracy
| ------------- |:-------------:|
| KNN    | right-aligned |
| Naive Bayes      | centered |     
| Gradient Boosting | are neat  |  
| Random Forest   | right-aligned |
| SVM     | centered    |  
| XGBoost | 85.6% |
| Adaboost   | right-aligned |
| Extra Trees      | 84.2%  |   
| Light Gradient Boosting | are neat |
| Voting Classifier | 87.5% |
