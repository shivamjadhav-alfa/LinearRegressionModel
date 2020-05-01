#Linear Regression model from scratch



#Importing essential libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing dataset to build model 

dataset=pd.read_csv("Salary_Data.csv")

#Taking Dependent and Independent variables

X=dataset.iloc[:,0].values
Y=dataset.iloc[:,-1].values

# following general form Y=Mx+C to find Correlation 
# Y is dependent variable M is Slope and C is intercept
# m=sum((y-y^)*(x-x^))/sum((x-x^)**2) """

x_mean=np.mean(X)
y_mean=np.mean(Y)

numerator=0
denominator=0
for i in range(30):
    numerator+=((X[i]-x_mean)*(Y[i]-y_mean))
    denominator+=(X[i]-x_mean)**2
M=numerator/denominator
C=y_mean-M*x_mean


# Building Model

max_x=np.max(X)
min_x=np.min(X)

x=np.linspace(min_x,max_x,1000)
y=M*x+C


# Visualising data

plt.plot(x,y,color="purple")
plt.scatter(X,Y,color="red")
plt.title("Experience vs Salary {Linear Regression}")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


