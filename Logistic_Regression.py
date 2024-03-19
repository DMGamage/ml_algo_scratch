import random
import numpy as np

class LogisticRegression:

    def logistic_regression(x,y,iterations=100,
                            learning_rate =0.01):
        m,n = len(x),len(x[0])
        beta_0 ,beta_other = initialize_params(n)

        for _ in range(iterations):
            gradient_beta_0,gradient_beta_other = (
                compute_gradients(x,y,beta_0,
                                  beta_others,m,n,50)
            )

            beta_0,beta_other = update_params(
                beta_0,beta_other,gradient_beta_0,
                gradient_beta_other,learning_rate

            )

            return beta_0,beta_other
        
    def initialize_params(n):
        beta_0 = 0
        beta_other = [random.random() for _ in range(n)]

        return beta_0,beta_other
    
    def compute_gradients(x,y,beta_0,beta_other,m,n):
        gradient_beta_0 = 0
        gradient_beta_other = [0]*n

        for i,point in enumerate(x):
            pred = logistic_function_np(point,beta_0,beta_other)

            for j , feature in enumerate(point):
                gradient_beta_other[j] +=(
                    pred -y[i] )*feature /m
                gradient_beta_0 +=(pred-y[i])/m

        return gradient_beta_0, gradient_beta_other
    
    def logistic_function_np(point,beta_0,beta_other):
        return 1/(
            1+ np.exp(-(beta_0 + point.dot(beta_other)))
        )
    
    def update_params(beta_0,beta_other,
                      gradient_beta_0,
                      gradient_beta_other,
                      learning_rate):
        beta_0 -=gradient_beta_0*learning_rate
        for i in range (len(beta_other)):
            beta_other[i] -=(gradient_beta_other[i]*learning_rate)
            return beta_0 , beta_other
