kaggle-stackoverflow
=====

Machine learning set-up for 
https://www.kaggle.com/c/predict-closed-questions-on-stack-overflow

usage:

```bash
$ cd src/
$ ./runme.py
```

which will extract 35 handcrafted features as well as their combinations from the training data, train an ensemble of classifiers 
on the feature matrix ( random forest, linear discriminant analysis, and gradient boosting), and make predictions for the public leaderboard data.

Note: feature generation requires the nltk corpora ( download should be prompted on first run )
