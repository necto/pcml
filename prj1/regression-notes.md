1) We have a typical learning curve for High bias: The training error is unacceptably high and
there is a small gap between training and test error.


1) Test leastSquare with gradient descent with normalized features.
rmseTe = 1.3683e+03 rmseTr = 1.3683e+03
2) Test Least squares using normal equations. The matrix X is ilconditioned 
so the result may be innacurate
rmseTe = 1.3683e+03 rmseTr = 1.3683e+03
3) Test Ridge regression

4) The data are grouped into 3 different clouds. In order to reduce 
the error, I use one model for each cloud. The rmse is better that way(694)

5) Dummy coding doesn't improve the predictions accuracy.

TODO, test other way to improve accuracy
test model with 3 cloud with ridge regression