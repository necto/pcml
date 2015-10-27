1) Tried a trivial logistic regression with normalized features. Tested on a untouched test set (0.3 fraction of the original training set). Does not work: 402 misspredictions out of 450 (a lot more than 50%, amazing)
2) Tried a penalized logistic regression using cross validation to choose lambda. Independently on lambda, I've got very high level of misspredication (constant on lambda). That is weird, must be a mistake in the implementation somewhere.
3) Got it! I had an extra minus in the sigmoid argument. Redoing again.
1') simple logistic regression on the normalized features (X_train), and output (y_train {-1,1} -> {0,1}, trained on 70% of the given  data, and tested on the 30% gives 0.1067 misprediction.
2') playing with the lambda log diapason (starting from 0.0001): determined: nothing really changes until lambda reaches 0.01.
 - test and train errors are practically identical for the whole spectrum of lambda. meaning that there is no variance, only bias.
 - optimal lambda value is 0.2656, giving the cross-validation test error: 0.0964 ; and the untouched test error: 0.1311 :(
4) Trying to shrink the set of features. Ran cross-validation, trying to fit data without each single feature. The feature #18 (counting from the 'ones' column) promises 0.0954 cross-validation mean error. Unfortunately, applying linear regression wieht the feature #18 deleted to the untouched data gives 0.1178 mispredicaiton fraction.
5) Tried add features in the power of 2,3,4,5,6,7,8 (see generalized myPoly), The best rezult is achieved without this transformation.
6) Checked real polynoms of the second degree: cross-validation test error: 0.0210. But the fresh test misprediction ratio is again much worse: 0.1578. It seems that the bias is not gone yet :(