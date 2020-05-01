import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset=pd.read_csv("Salary_Data.csv")
X=dataset.iloc[:,0].values
Y=dataset.iloc[:,-1].values

xmean=np.mean(X)
ymean=np.mean(Y)

nume=0
denom=0
for i in range(30):
    nume+=((X[i]-xmean)*(Y[i]-ymean))
    denom+=(X[i]-xmean)**2
M=nume/denom
C=ymean-M*xmean
y=M*x+C




max_x=np.max(X)
min_x=np.min(X)

x=np.linspace(min_x,max_x,1000)
y=M*x+C



plt.plot(x,y,color="purple")
plt.scatter(X,Y,color="red")
plt.title("Experience vs Salary {Linear Regression}")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()













# Fitting Linear Regression to the dataset
