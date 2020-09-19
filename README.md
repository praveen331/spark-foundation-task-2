# spark-foundation-task-2
# PRAVEEN GUPTA



# LinearRegression with python Scikit Learn- Supervised ML
# TASK -2



# importing libraries
 import pandas as pd
 import numpy as np
 import matplotlib.pyplot as plt
 from sklearn.model_selection import train_test_split
 from sklearn.linear_model import LinearRegression
 
 #Reading the data from the pdf
dataset=pd.read_csv("http://bit.ly/w-data")
dataset.head()

#checking the size and details about dataset
dataset.describe()

#plotting  the data to find patterns
 dataset.plot(x='Hours',y='Scores',style='o')
 plt.tittle('Hours Studied against Score')
 ply.xlabel('Hours Studied')
 plt.ylabel9('Score')
 
 #creating variables to divide the independent and dependent variable
 x=dataset['Hours'].values.reshape(-1,1)
 y=dataset['Scores'].values.reshape(-1,)
  
 #now dividing the above data into 80-20 for training and testing respectively
 x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
 
#training our data
regressor =LinearRegression()
regressor.fit(x_train,y_train) 

#retrieving the intercept and slope
print("Intercept",regressor,intercept_)
print("Slope:",regressor.coef_)

# plotting the regression line to see the fit
line=regressor.coef_*x+regressor.intercept_
 
#plotting for the test data
plt.scatter(x,y)
plt.plot(x,line);

#predicting the values with the actual values
comp=pd.DataFrame({'Actual':y_test.flatten(), 'Predicted':y_pred.flatten()})
comp

#enter a specific value in the equation 
hrs=9.25
ans=regressor.predict(np.array([hrs]).reshape(-1,1))

print("The predicted score for {} hours is{}".format(hrs,ans[0]))
