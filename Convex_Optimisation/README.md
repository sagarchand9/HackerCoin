Here we are using returns as an indicator as it made more sense that to directly use the price.

Variable matrix x is weights of different currencies here. 

Condition 1 - 
Sum of weights should be positive.
Ax = b
Matrix A is 1x10 dim with all 1's. b is 1x1 dim with val 1. 
This gives w1 + w2 + ... w10 = 1


Condition 2 - 
Weights are all positive
Gx <= h
G is 10x10 diagonal matrix with values -1. h is 10x1 will all 0's
This gives each of the weights is >= 0



Our objective is to minimize volatility.
So in minimizing 0.5x'Px + q'x, to keep equations simple, we ignore the second term by making q as 0 matrix. In the first term, we make P as covariance matrix. So our equation is w1^2cov(cur1,cur1) + w1w2cov(cur1,cur2) + .. + w10w9cov(cur10,cur9) + w10^2cov(cur10,cur10)


