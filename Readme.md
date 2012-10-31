kaggle-stackoverflow
=====

Machine learning set-up for 
https://www.kaggle.com/c/predict-closed-questions-on-stack-overflow

Usage:

```bash
$ cd src/
$ ./runme.py
```

which will extract 35 handcrafted features as well as their combinations from the training data 
( using the nltk for tokenization and stemming ), 
train an ensemble of classifiers on the feature matrix 
( Random Forest, Linear Discriminant Analysis, and Gradient Boosting, all from the scikit-learn toolkit), 
and make predictions for the private leaderboard data.

Note: feature generation requires the nltk corpora ( download should be prompted on first run )
