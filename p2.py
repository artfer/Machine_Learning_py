import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)


votesFeatures = ["Class","handicapped-infants","water-project-cost-sharing",
"adoption-of-the-budget-resolution","physician-fee-freeze",
"el-salvador-aid","religious-groups-in-schools",
"anti-satellite-test-ban","aid-to-nicaraguan-contras",
"mx-missile","immigration","synfuels-corporation-cutback",
"education-spending","superfund-right-to-sue","crime",
"duty-free-exports","export-administration-act-south-africa"]

def import_data(file):
    if 'votes' in file:
        df = pd.read_csv(file,names=votesFeatures)

        #turn into numeric values
        df[df.columns[1:]] = df[df.columns[1:]].applymap(lambda x: '1' if x=='y' else('0' if x=='n' else '-1'))
        df[df.columns[:1]] = df[df.columns[:1]].applymap(lambda x: '1' if x=='republican' else '0')

        for col in df[df.columns]:
            df[col] = pd.to_numeric(df[col])
    elif 'soccer' in file:
        df = pd.read_csv(file)

        header = df.iloc[0]
        df = df[1:]
        df.rename(columns = header)
        df = df.dropna()
        df['B365H'] = df['B365H'].map(lambda x : round(x,0))
        df['B365D'] = df['B365D'].map(lambda x : round(x,0))
        df['B365A'] = df['B365A'].map(lambda x : round(x,0))
    elif 'heart' in file:
        df = pd.read_csv(file)

        header = df.iloc[0]
        df = df[1:]
        df.rename(columns = header)
        df = df.rename(columns={'target': 'Class'})
    return df


def split_data(df):
    train,test = train_test_split(df,test_size=0.3) # 70% training and 30% test
    train_c = train['Class']
    train = train.drop('Class',axis=1)
    test_c = test['Class']
    test = test.drop('Class',axis=1)
    return train,train_c,test,test_c


def train_and_test(data,type,train,train_c,test,test_c,debug=False):

    #Create a Classifier
    if type == 1:
        if debug: print('Naive Bayes:')
        model = GaussianNB()
    elif type == 2:
        if debug: print('Logistic Regression:')
        if data == 2:
            model = LogisticRegression(solver='saga',multi_class='auto')
        else:
            model = LogisticRegression(solver='liblinear')
    else:
        if debug: print('Decision Tree:')
        model = DecisionTreeClassifier()

    start = time()
    #Train the model using the training sets
    model.fit(train,train_c)
    end = time()
    train_time = end-start
    if debug: print('\tTime to train: %.3f seconds' % train_time)


    start = time()
    #Predict the response for test dataset
    pred_c = model.predict(test)
    end = time()
    test_time = end-start
    if debug: print('\tTime to predict: %.3f seconds' % test_time)

    # Model Accuracy, how often is the classifier correct?
    accuracy  = metrics.accuracy_score(test_c, pred_c)*100
    precision = 0
    recall = 0
    if data != 2:
        precision = metrics.precision_score(test_c, pred_c)*100
        recall    = metrics.recall_score(test_c, pred_c)*100
    if debug:
        print('\tAccuracy: %.3f%%' % accuracy)
        if data != 2:
            print('\tPrecision: %.3f%%' % precision)
            print('\tRecall: %.3f%%\n' % recall)

    if not debug:
        return train_time,test_time,accuracy,precision,recall
