'''
ALL the models were tested and accuracy scored were found of each model which were:
1.Logistic Regression 
Accuracy:0.8837209302325582
2.Decision tree 
Accuracy:0.9767441860465116
3.Random Forest 
Accuracy:0.9534332876755210
4.KNN
Accuracy:0.8139534883720932
5.SVM
Accuracy:0.8139534883720930

Best Model is Decision Tree with Accuracy of 0.9767441860465116

'''
import pandas as pd 
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,mean_absolute_error

df=pd.read_csv("Predicting Placement in Campus Recruitment.csv")                                        #Reading the Csv
print(df.isnull().sum())                                                                                #Checking the values which are null
df["salary"]=df["salary"].fillna(df["salary"].median())                                                 #median Values will be given to empty entries

df.info()                                                                                               #Checking the type of the Columns
le=LabelEncoder()
col=["gender","ssc_b","hsc_b","hsc_s","degree_t","workex","specialisation","status"]                    #columns which need conversion
for i in col:
    df[i]=le.fit_transform(df[i])

count=sns.countplot(x="status",data=df)
sns.boxplot(x="status",y="etest_p",data=df)

print(df.describe())

X=df.drop(["sl_no","status"],axis=1)                                                                    #defining the input Variable
y=df["status"]                                                                                          #defining Output Variable
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=30)                       #Splitting the Training and testing Data 

sc=StandardScaler()                                                                                     #Normalizing the data set
X_train=sc.fit_transform(X_train)                                                              
X_test=sc.transform(X_test)

my_model=DecisionTreeClassifier(random_state=0)                                                         #Choosing the best model on basis of accuracy which is (DecisionTreeClassifier)
results=my_model.fit(X_train,y_train)                                                                   #Training the Model
predictions=results.predict(X_test)                                                                     #Predicting the Values 


#For Decision Tree
print("Accuracy is :",accuracy_score(y_test,predictions))                                               #Highest Accuracy Model= 0.9767441860465116
print("Absolute Error:",mean_absolute_error(y_test,predictions))
print("confusion_matrix is")
conf_mat=confusion_matrix(y_test,predictions)                                                           
con_df = pd.DataFrame(conf_mat, index=["Predicted 0","Predicted 1"], columns=["Actual 0","Actual 1"])   #Converting the Matrix into tabular form
print(con_df)
print("Classification Reports is")
print(classification_report(y_test,predictions))                                                        #printing the classification report
new_pred=results.predict([[1,92.00,0,80.33,1,2,75.00,2,1,80.5,0,76.28,360000.0]])                       #checking the model for a set of inputs
print(new_pred)

#For Random forest
my_model2=RandomForestClassifier(n_estimators=50,criterion="entropy",random_state=50)
res1=my_model2.fit(X_train,y_train)
predictions2=res1.predict(X_test) 

print("Accuracy is :",accuracy_score(y_test,predictions2))                                               #Highest Accuracy Model= 0.9767441860465116
print("Absolute Error:",mean_absolute_error(y_test,predictions2))
print("confusion_matrix is")
conf_mat=confusion_matrix(y_test,predictions2)                                                           #printing the Confusion Matrix
con_df = pd.DataFrame(conf_mat, index=["Predicted 0","Predicted 1"], columns=["Actual 0","Actual 1"])   #Converting the Matrix into tabular form
print(con_df)
print("Classification Reports is")
print(classification_report(y_test,predictions2))                                                        #printing the classification report
new_pred=res1.predict([[1,92.00,0,80.33,1,2,75.00,2,1,80.5,0,76.28,360000.0]])                       #checking the model for a set of inputs
print(new_pred)



#For Logistic Regression
my_model3=LogisticRegression()
res2=my_model3.fit(X_train,y_train)
predictions3=res2.predict(X_test)  

print("Accuracy is :",accuracy_score(y_test,predictions3))                                               #Highest Accuracy Model= 0.9767441860465116
print("Absolute Error:",mean_absolute_error(y_test,predictions3))
print("confusion_matrix is")
conf_mat=confusion_matrix(y_test,predictions3)                                                           #printing the Confusion Matrix
con_df = pd.DataFrame(conf_mat, index=["Predicted 0","Predicted 1"], columns=["Actual 0","Actual 1"])   #Converting the Matrix into tabular form
print(con_df)
print("Classification Reports is")
print(classification_report(y_test,predictions3))                                                        #printing the classification report
new_pred=res2.predict([[1,92.00,0,80.33,1,2,75.00,2,1,80.5,0,76.28,360000.0]])                       #checking the model for a set of inputs
print(new_pred)