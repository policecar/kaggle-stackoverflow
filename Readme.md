## kaggle-stackoverflow


Machine learning set-up for the [Stackoverflow competition at Kaggle]
(https://www.kaggle.com/c/predict-closed-questions-on-stack-overflow) in 2012

( scored in the top 25% )

### Usage:

```bash
$ cd src/
$ ./runme.py
```

which will extract 35 handcrafted features and their combinations from the training data 
( using the [NLTK](http://nltk.org/) for tokenization and stemming ), 
train an ensemble of classifiers on the feature matrix 
( [Random Forest](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), 
[Linear Discriminant Analysis](http://scikit-learn.org/stable/modules/generated/sklearn.lda.LDA.html), and 
[Gradient Boosting](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html), 
all from the [scikit-learn toolkit](http://scikit-learn.org)), 
and make predictions for the private leaderboard data.

Note: feature generation requires some of the [NLTK corpora](http://nltk.org/data) 
( download should be prompted on first run )
