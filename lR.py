def initialize_params(dimentions):
    beta_0=0
    beta_other =[random.random() for _ in range(dimentions)]
    return beta_0,beta_other

def compute_gradient(x,y,beta_0,beta_other,dimension,m):
    gradient_beta_0 =0
    gradient_beta_other =[0]*dimension

    for i in range(m):
        y_i_hat = sum(x[i][j]*beta_other[j] 
                      for j in range(dimension)) + beta_0
        derror_dy = 2*(y[i]-y_i_hat)

        for j in range(dimension):
            gradient_beta_other[j] +=derror_dy*x[i][j]/n
            gradient_beta_0 +=derror_dy/n
        return gradient_beta_0,gradient_beta_other

def update_params(beta_0,beta_other,gradient_beta_0,
                  gradient_beta_other,learning_rate):
    beta_0 += gradient_beta_0*learning_rate

    for i in range(len(beta_1)):
        beta_other[i] += (gradient_beta_other[i]*
                          learning_rate)
        return beta_0,beta_other



def linear_regeression(x,y,iterations=100,learning_rate=0.01):
    n,m = len(x[0]),len(y[0])
    beta_0,beta_other = initialize_params(n)
    for _ in range(iterations):
        gradient_beta_0,gradient_beta_other = compute_gradient(x,y,beta_0,beta_other,n,m)
        beta_0,beta_other = update_params(
            beta_0,beta_other,gradient_beta_0,gradient_beta_other,learning_rate
        )
    return beta_0,beta_other