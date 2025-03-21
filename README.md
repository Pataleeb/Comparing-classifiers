# Comparing-classifiers
Scikit-learn

We perform divorce classification using a dataset containing responses from participants who completed a personal information form and a divorce predictors scale. The dataset includes 170 participants and 54 real-valued attributes (predictor variables).

The last column of the CSV file is the target label y, where:

1 indicates "divorce"
0 indicates "no divorce"

Each column represents a feature (predictor variable), and each row corresponds to one participant (sample).

We will compare the performance of the following classifiers: Naive Bayes, Logistic Regression, and K-Nearest Neighbors (KNN). The dataset will be split into 80% for training and 20% for testing.

The original dataset can be found at: (https://archive.ics.uci.edu/ml/datasets/Divorce+Predictors+data+set)

First, we report the testing accuracy for each of the three classifiers.
Next, we perform Principal Component Analysis (PCA) to project the data into a two-dimensional space and rebuild the classifiers using the transformed data.
