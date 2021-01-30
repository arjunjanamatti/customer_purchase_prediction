# importing required libraries
import pandas as pd
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix

# load the dataframe
try:
    raw_df = pd.read_csv('Data.csv',encoding='cp1252')
except:
    raw_df = pd.read_csv('Data.csv',encoding='utf-8')
print('Dataframe loaded successfully, and shape of dataframe is: {}'.format(raw_df.shape))

# converting the InvoiceDate to datetime
raw_df['InvoiceDate'] = pd.to_datetime(raw_df['InvoiceDate'])
# extracting only dates from datetime
raw_df['Date'] = raw_df['InvoiceDate'].dt.date
# converting the dates to datetime
raw_df['Date'] = pd.to_datetime(raw_df['Date'])
# sorting the pandas dataframe using dates
raw_df = raw_df.sort_values('Date')
# setting the index column as date column
raw_df.index = raw_df['Date']

# get the values of year-month format
per = raw_df.Date.dt.to_period("M")

# grouping the dataframe based on the customer Id and year-month
df_groupby = raw_df.groupby(['Customer ID', per]).agg('size')

# unstacking at year-month level such that the all the year-month will be in columns and customer id in rows index
df_final = df_groupby.unstack(level=-1)

# filling the null values with 0
df_final.fillna(0, inplace=True)

# here will convert this dataframe to a classification problem, hence all values of at least 1 time shopping by a customer is converted 1, else 0
cols_list = list(df_final.columns)
for cols in cols_list:
    df_final[cols] = df_final[cols].apply(lambda x: 1 if x>0 else 0)

#  Here for testing purposes, the final month in available dataset is taken as a target value
X = df_final.iloc[:,:-1]
y = df_final.iloc[:,-1]

#train & test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

def model_result(model, X_train, y_train, X_test, y_test):
    '''
    model is the name of model used for prediction
    This function will give the results of model for Roc_Auc_score and confusion matrix
    '''
    model_1 = model.fit(X_train, y_train)
    pred_value = model_1.predict(X_test)
    print('{} Roc Auc score: {}%'.format(model
                                         ,round(roc_auc_score(y_true = y_test,y_score = pred_value)*100, 2)))
    print('{} Confusion matrix: \n{}'.format(model, confusion_matrix(y_true = y_test,y_pred = pred_value)))

models = []
models.append(("LR",LogisticRegression()))
models.append(("NB",GaussianNB()))
models.append(("RF",RandomForestClassifier()))
models.append(("SVC",SVC()))
models.append(("Dtree",DecisionTreeClassifier()))
models.append(("XGB",xgb.XGBClassifier()))
models.append(("KNN",KNeighborsClassifier()))

for name,model in models:
    model_result(model, X_train, y_train, X_test, y_test)
    print()

# since Gaussian Naive Bayes gives the best result will predict using this model
nb_model = GaussianNB().fit(X,y)
prob_result_matrix = (nb_model.predict_proba(X))
a = (prob_result_matrix[:,1])
result_df = pd.DataFrame()
result_df['probablity of purchase'] = a
result_df.index = df_final.index
result_df.to_csv('result.csv')