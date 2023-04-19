import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import json
from sklearn.svm            import SVC 
from sklearn.svm            import LinearSVC
from sklearn.linear_model   import LogisticRegression 
from sklearn.tree           import DecisionTreeClassifier
from sklearn.ensemble       import RandomForestClassifier 
import joblib 

#################################################################################################

def clean_data_processing(path):
    df = pd.read_csv(path)
    new_columns = [] 
    for i in df.columns:
        df[i.lower().strip()] = df[i] 
        new_columns.append(i.lower().strip())
    data_cleaned = df[new_columns]
    data_cleaned = data_cleaned.drop_duplicates().dropna()
    return data_cleaned

data_cleaned = clean_data_processing("data_uploaded/dummy_data.csv") 

#################################################################################################

def all_columns(data):
    return list(data.columns)

#################################################################################################

def drop_columns_check(list_1, list_2):
    for i in list_1:
        if i in list_2: 
            result = True 
        else:
            result = i 
            break 
    return result

#################################################################################################

def features_columns(data, target):
    data = data 
    target = target.lower().strip()
    feature_columns = data.drop(target, axis=1).columns 
    return list(feature_columns)

feature_columns_ = features_columns(data=data_cleaned, target='target')

#################################################################################################

def json_dictionary(feature_columns, data):
    data_dictionary = {}
    for i in feature_columns:
        if type(data[i][0]) == str:
            data_dictionary[i] = list((data[i].unique()))
    data_dictionary_encoded = {}
    for key, value in data_dictionary.items():
        value_dictionary = {}
        for idx, i in enumerate(value): 
            value_dictionary[i] = idx
        data_dictionary_encoded[key] = value_dictionary
    return data_dictionary, data_dictionary_encoded

json_data, json_data_encoded = json_dictionary(feature_columns=feature_columns_, data=data_cleaned)

#################################################################################################

def save_json_data(file_to_save):
    json_ = json.dumps(file_to_save)
    filename = open("data_uploaded_saved/json_data.json","w")
    filename.write(json_)
    filename.close()
    return 

save_json_data(file_to_save=json_data) 

#################################################################################################

def save_json_data_encoded(file_to_save):
    json_ = json.dumps(file_to_save)
    filename = open("data_uploaded_saved/json_data_encoded.json","w")
    filename.write(json_)
    filename.close()
    return 

save_json_data_encoded(file_to_save=json_data_encoded)

#################################################################################################

def load_json_data_encoded(path):
    filename = open(path)
    json_data = json.load(filename)
    filename.close()
    return json_data 

json_data_encoded = load_json_data_encoded(path="data_uploaded_saved/json_data_encoded.json")
# print(json_file)

#################################################################################################

def encode_data(features, process_data, json_file):
    data_encoded = process_data 
    for i in features:
        if type(process_data[i][0]) == str:
            process_data[i] =  process_data[i].apply(lambda x : json_file[i][x])
        else:
            process_data[i] = process_data[i]
    return data_encoded

data_encoded = encode_data(features=feature_columns_, process_data=data_cleaned, json_file=json_data_encoded)
# print(data_encoded)

#################################################################################################

def split_data_into_x_y(data, target):
    x = data.drop(target, axis=1)
    y = data[target]
    return x, y 

X, y = split_data_into_x_y(data=data_encoded, target="target")

#################################################################################################

def decode_input(input_dictionary, json_data_encoded): 
    input_features, input_features_decoded = [], []
    for i, j in input_dictionary.items():
        input_features.append(j)
        if i in json_data: 
            k = json_data_encoded[i][input_dictionary[i]]
        else:
            k = j 
        input_features_decoded.append(k)
    return input_features_decoded 

#################################################################################################

def data_to_np_array( x ):
    return np.array([x], dtype=float)

#################################################################################################

def instantiations():
    model_svc = SVC()
    model_lsv = LinearSVC()
    model_lgr = LogisticRegression()
    model_dtc = DecisionTreeClassifier()
    model_rfc = RandomForestClassifier()
    return model_svc, model_lsv, model_lgr, model_dtc, model_rfc

#################################################################################################

def model_fitting(X_train, y_train,  model_svc, model_lsv, model_lgr, model_dtc, model_rfc):
    model_svc_ = model_svc.fit(X_train, y_train)    
    model_lsv_ = model_lsv.fit(X_train, y_train)    
    model_lgr_ = model_lgr.fit(X_train, y_train)    
    model_dtc_ = model_dtc.fit(X_train, y_train)    
    model_rfc_ = model_rfc.fit(X_train, y_train) 
    return model_svc_, model_lsv_, model_lgr_, model_dtc_, model_rfc_ 

#################################################################################################

def save_models(model_svc, model_lsv, model_lgr, model_dtc, model_rfc):
    joblib.dump(model_svc, filename = "project_1_trained_models/trained_model_svc.joblib")   
    joblib.dump(model_lsv, filename = "project_1_trained_models/trained_model_lsv.joblib")   
    joblib.dump(model_lgr, filename = "project_1_trained_models/trained_model_lgr.joblib")   
    joblib.dump(model_dtc, filename = "project_1_trained_models/trained_model_dtc.joblib")   
    joblib.dump(model_rfc, filename = "project_1_trained_models/trained_model_rfc.joblib")  
    return 

#################################################################################################

def load_models():
    model_svc = joblib.load( "project_1_trained_models/trained_model_svc.joblib" )  
    model_lsv = joblib.load( "project_1_trained_models/trained_model_lsv.joblib" )    
    model_lgr = joblib.load( "project_1_trained_models/trained_model_lgr.joblib" )     
    model_dtc = joblib.load( "project_1_trained_models/trained_model_dtc.joblib" )     
    model_rfc = joblib.load( "project_1_trained_models/trained_model_rfc.joblib" ) 
    return model_svc, model_lsv, model_lgr, model_dtc, model_rfc 

#################################################################################################

def prediction(input_decoded_, model_svc, model_lsv, model_lgr, model_dtc, model_rfc):
    predict_svc = model_svc.predict(input_decoded_)[0]
    predict_lsv = model_lsv.predict(input_decoded_)[0] 
    predict_lgr = model_lgr.predict(input_decoded_)[0]  
    predict_dtc = model_dtc.predict(input_decoded_)[0]  
    predict_rfc = model_rfc.predict(input_decoded_)[0]
    return predict_svc, predict_lsv, predict_lgr, predict_dtc, predict_rfc

#################################################################################################

def check_accuracy(X_test, y_test, model_svc, model_lsv, model_lgr, model_dtc, model_rfc):
    accuracy_svc = model_svc.score(X_test, y_test)    
    accuracy_lsv = model_lsv.score(X_test, y_test)    
    accuracy_lgr = model_lgr.score(X_test, y_test)     
    accuracy_dtc = model_dtc.score(X_test, y_test)     
    accuracy_rfc = model_rfc.score(X_test, y_test)  
    return accuracy_svc, accuracy_lsv, accuracy_lgr, accuracy_dtc, accuracy_rfc   

#################################################################################################

def save_target_name(target):
    with open("data_uploaded_saved/target_name.txt", "w") as file:
        file.write( target ) 
        return 

#################################################################################################

def read_saved_target_name():
    with open("data_uploaded_saved/target_name.txt", "r") as file:
        for line in file:
            target_name = line.split("'")[0]
    return str(target_name)

#################################################################################################

def drop_columns_finally(columns, data_cleaned):
    return data_cleaned.drop(columns, axis=1)

#################################################################################################
