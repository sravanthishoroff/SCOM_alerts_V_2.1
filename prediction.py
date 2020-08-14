import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re, pickle
le = LabelEncoder()
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import pickle
import random, json , requests

def document(file_name):
    results = []
    df = pd.read_excel(file_name)

    input_alarm_count = {"title":"Total count",
                            'count': df.shape[0]}
    results.append(input_alarm_count)

    df1 = df[['Summary of the alert','Ignorable or Non-Ignorable']]
    df2 = df[['Summary of the alert','Ignorable or Non-Ignorable']]

    print(df1['Ignorable or Non-Ignorable'].value_counts())

    df1['Ignorable or Non-Ignorable'] = df1['Ignorable or Non-Ignorable'].replace(to_replace='Non-ignorable',value='Non-Ignorable')
    print(df1['Ignorable or Non-Ignorable'].value_counts())

    df1['Ignorable or Non-Ignorable']=df1['Ignorable or Non-Ignorable'].apply(lambda x: 1 if (x)=='Non-Ignorable' else 0)
    print(df1['Ignorable or Non-Ignorable'].value_counts())
    
    df1['Summary of the alert'].dropna(how='all')
    df1['Ignorable or Non-Ignorable'].dropna(how='all')
    df1['Summary of the alert'] = df1['Summary of the alert'].apply(lambda x: x.lower())
    df1['Summary of the alert'] = df1['Summary of the alert'].apply(lambda s: re.sub(r"[^a-zA-Z0-9]"," ",s))

    lemmatizer=WordNetLemmatizer()
    def tokenize(text):
        tokens = nltk.word_tokenize(text)
        tokens =[w for w in tokens if not w in stop] 
        lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in tokens])
        return lemmatized_output

    df1['Summary of the alert'].apply(tokenize)
    tfidf_vector=TfidfVectorizer(use_idf=True,max_features=1000)
    tf = tfidf_vector.fit_transform(df1['Summary of the alert']).todense()

    vectorizer = pickle.load(open("vectorize.pkl", "rb"))

    #taking X and y
    X = vectorizer
    y = df1['Ignorable or Non-Ignorable']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # load model 
    pkl_file = 'C:\\Users\\SRAVANTHI SHOROFF\\Desktop\\sravanthi\\SCOM_ML_v_3\\votingMod1.pkl'
    with open(pkl_file,'rb') as file:
        votingclassifier = pickle.load(file)

    # predict class
    y_pred = votingclassifier.predict(X)
    # print(y_pred)
    # print(classification_report(y, y_pred))

    y_pred = y_pred.tolist()
    y_pred = pd.DataFrame(y_pred)
    print(y_pred)
    y_pred['predicted_value'] = y_pred
    y_pred = y_pred.drop([0],axis=1)

    outcount = pd.concat([y,y_pred],axis=1)
    
    outcount['result'] = outcount.apply(lambda x:'CorrectPrediction' if x['Ignorable or Non-Ignorable'] == x['predicted_value'] else 'IncorrectPrediction',axis=1).astype(str)

    outcount.rename(columns = {'Ignorable or Non-Ignorable':'actual_value'}, inplace = True)
    
    outcount['incident'] = outcount.apply(lambda x:'ConvertTicket' if x['actual_value'] and x['predicted_value'] == 1 else 'FlaseAlarm',axis=1).astype(str)
    # print(outcount['incident'].value_counts())

    dfnew_true = outcount.loc[outcount['incident'] == 'ConvertToTicket']
    t_df = df2[df2.index.isin(dfnew_true.index)]
    t_df.reset_index(drop=True, inplace=True)
    a_df = df2[df2.index.isin(outcount.index)]
    # print(a_df)

    final = pd.concat([a_df,outcount],axis = 1)
    print(final)

    output = pd.DataFrame(columns=['y_pred'],data = y_pred)
    # # print("output")
    output['pred_val'] = output['y_pred'].replace(to_replace=1,value='Yes')
    print(output['pred_val'])
    # #Generate false alarm excel
    false_output = output[output['pred_val']==0]

    # print(false_output.shape)
    false_alarm_count = {"title":"False Alarm",
                            'count' : false_output.shape[0]}

    results.append(false_alarm_count)

    output_df_false = df[df.index.isin(false_output.index)]
    output_df_false.to_excel('false_alarm_report.xls')

    #Generating true alarms from predicted output
    true_output = output[output['pred_val']=='Yes']

    # print(true_output.shape)
    true_alarm_count = {'title':"True Alarm ",
                            'count':true_output.shape[0]}
    results.append(true_alarm_count)
    output_df_true =df[df.index.isin(true_output.index)]

    # print(output_df_true)
    print(output_df_true.shape)

    data = {'result':results}
    # print(data)

    return final.to_html(header="true", table_id="table")
    # return data 