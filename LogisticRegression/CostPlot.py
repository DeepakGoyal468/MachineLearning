import numpy as np
import math
import matplotlib.pyplot as plt
mu = 0.0
sigma = 0.1
thetas = []
hThetaxs = []
x = [1, 34.6236596245, 78.0246928154]
logistic_costs = []
linear_costs = []



def linear_cost(hThetax):
    # if (hThetax >= 0.5):
    #     hThetax1 = 1
    # else:
    #     hThetax1 = 0

    linear_cost = np.square(hThetax) / (2)
    linear_costs.append(linear_cost)
    return  linear_costs


def logistic_cost(hThetax):

    logistic_cost = math.log(1-hThetax) * (-1)
    logistic_costs.append(logistic_cost)
    return logistic_costs


for i in range(100):
    theta = np.random.normal(mu, sigma, 3)
    thetas.append(theta)
    theta = np.array(theta).transpose()

    thetaX = np.matmul(theta, x)
    hThetax = 1/(1+(np.exp(-(thetaX))))

    linear_costs = linear_cost(hThetax)
    hThetaxs.append(hThetax)


    logistic_costs = logistic_cost(hThetax)



    #print(linear_cost)

plt.subplot(1,2,2)
plt.plot(thetas,logistic_costs,'ro')

plt.subplot(1,2,1)
plt.plot(thetas,linear_costs,'b*', )
plt.show()



