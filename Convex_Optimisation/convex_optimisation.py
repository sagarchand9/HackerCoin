import pandas as pd
import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")
Price = np.zeros((len(data),1))
wt = np.zeros((10, len(data)))
#wt = np.zeros((len(data), 10))
 
for i in range(2,len(data)+1):
    j = -i
    data1 = data[j:]
    cov = data1.cov()
    P = matrix(cov.as_matrix())
    q = matrix(np.zeros((10,1)))
    G = matrix(-np.diag(np.ones(10)))
    #G = matrix(-data.as_matrix())
    h = matrix(np.zeros((10,1)))
    A = matrix(np.ones((1,10)))
    b = matrix(np.ones((1,1)))

    sol = solvers.qp(P,q,G,h,A,b)

    weight = np.array(sol['x'])
    for k in range(10):
        wt[k, i-2] = weight[k]

    #vol = sol['primal objective']
    Ri = np.matmul(data.as_matrix()[j], weight)

    
    if(i==2):
        Price[i-1] = (1+Ri)
    else:
        print(i)
        Price[i-1] = (1+Ri)*Price[i-2]
   

curr = pd.read_csv("combined_final.csv")

inr = curr["INR"][::-1]
btc = curr["BTC"][::-1]
eur = curr["EUR"][::-1] 

plt.plot(Price, label = "hackoin") 
plt.plot(eur/eur.mean(), label = "EUR") 
plt.xlabel('Time')                         
plt.ylabel('Price') 
plt.legend()
plt.show()
