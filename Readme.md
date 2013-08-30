## kaggle-stackoverflow


Machine learning set-up for the [Stackoverflow competition at Kaggle]
(https://www.kaggle.com/c/predict-closed-questions-on-stack-overflow) in 2012

( scored in the top 25% )

### Usage:

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
