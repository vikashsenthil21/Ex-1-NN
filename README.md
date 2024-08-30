<H3>ENTER YOUR NAME  : Vikash s </H3> 
<H3>ENTER YOUR REGISTER NO: 212222240115</H3>
<H3>EX. NO.1</H3>
<H3>DATE: 27/08/2024</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```python
from google.colab import files
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('Churn_Modelling.csv')
print(df)

df.head()
df.tail()
df.columns

print(df.isnull().sum())
df.duplicated()

X = df.iloc[:, :-1].values
print(X)
y = df.iloc[:,-1].values
print(y)

df.fillna(df.mean().round(1), inplace=True)
print(df.isnull().sum())

df.describe()
df1 = df.drop(['Surname','Geography','Gender'],axis=1)
df1.head()

scaler = MinMaxScaler()
df2 = pd.DataFrame(scaler.fit_transform(df1))
print(df2)

X = df.iloc[:, :-1].values
print(X)
y = df.iloc[:,-1].values
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
print(X_train)
print("Length of X_train:",len(X_train))
print(X_test)
print("Length of X_test:",len(X_test))
```


## OUTPUT:
## DATA HEAD 
![image](https://github.com/user-attachments/assets/5148b8ca-68a7-4d3d-b47a-5bf802e4167f)

## DATA CHECKING
![image](https://github.com/user-attachments/assets/d8b0c3af-1167-4d37-a380-f424d49e6e14)

## NULL VALUES
![image](https://github.com/user-attachments/assets/9d0c5b8f-3098-426c-a975-57fcc2efef5f)

## X VALUE
![image](https://github.com/user-attachments/assets/e1e6543a-d911-4d54-b52a-cd55200aae16)

## Y VALUE
![image](https://github.com/user-attachments/assets/dc216c78-411b-48e4-ac36-5f6fb83dd876)

## OUTLIERS
![image](https://github.com/user-attachments/assets/1a2a1255-1472-40f1-aae5-4204a3f202f6)


## DROP
![image](https://github.com/user-attachments/assets/27676ab0-9c07-4e9a-b39f-e47a1cc793a2)

## NORMALIZATION
![image](https://github.com/user-attachments/assets/a675d1d4-2cda-4ab1-959e-d7a827bccea2)

## DATA SPLITING
![image](https://github.com/user-attachments/assets/532f5a5a-96f6-446a-88dc-689e9f6bf812)

## TRAINING & TEST DATA 
![image](https://github.com/user-attachments/assets/671c245e-d6d8-4be0-bada-f05b36500163)




## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


