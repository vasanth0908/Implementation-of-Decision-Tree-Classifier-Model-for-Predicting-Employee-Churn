# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2.  Upload and read the dataset.   
3. Check for any null values using the isnull() function.
4.  From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5.  Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: vasanth s
RegisterNumber:  212222110052

```
```
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evalution","number_project","average_montly_hours","time_spend_company","work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
*/
```

## Output:
![image](https://github.com/23004513/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138973069/74e1ebb6-ad77-496e-af13-3f67edb60b99)

![image](https://github.com/23004513/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138973069/17e368c6-56d9-4d89-a5e2-b9be702e4f91)

![image](https://github.com/23004513/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138973069/8c5ad82c-d925-4f9c-98b5-70b42c4f8b8f)

![image](https://github.com/23004513/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138973069/4bf731e6-0b87-4193-8858-c7c107ccf65e)

![image](https://github.com/23004513/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138973069/1dc35319-f036-4518-a1c7-a0fec802f2d9)

![image](https://github.com/23004513/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138973069/7f465a1f-5b68-41f2-b6b5-707675800086)

![image](https://github.com/23004513/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138973069/6bbcbd08-8d41-4005-9fcc-d9afb9007665)

![image](https://github.com/23004513/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138973069/387a443d-b738-4919-9ff3-758391a86026)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
