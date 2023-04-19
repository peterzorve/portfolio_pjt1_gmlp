
from flask import Flask, render_template, request, flash, redirect, url_for  
# from pjt2_tds__pjt1_helper_functions import * 
from helper_function_gmlp import * 
# from pjt3_tcc_helper_functions import * 
# from flask_sqlalchemy import SQLAlchemy
import os 
from werkzeug.utils import secure_filename 
# from time import time 
# from datetime import datetime 
# import pandas as pd 

# import pandas as pd 
# import joblib 

# project_1_trained_models trained_model

from sklearn.model_selection import train_test_split 

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = "data_uploaded" 
app.secret_key = 'super_secret_key'

@app.route('/', methods=['POST', 'GET'])
def pjt1_gmlp(): 

     model_types = ["Support Vector Classifier Model",  "Linear SVC Model", "Logistic Regression Model", "Decision Tree Classifier Model", "Random Forest Classifier Model"]

     data_path, data_target = "", "" 
     data_cleaned = ""

     input_dictionary, json_data, feature_columns_, input_decoded_ , predicted_results, accuracy_scores   = {}, {}, "", "", "", ""

     wrong_format = ""
     all_columns = ""

     if request.method == 'POST':
          if request.form['submit_button'] == 'start_training': 
               check_file_format  = request.files['csv_file']
               check_drop_columns = request.form.get('drop_') 
               filename_format = secure_filename(check_file_format.filename)

               if filename_format.endswith('.csv') or filename_format.endswith('.xlsx') or filename_format.endswith('.xls'):
                    uploaded_file = request.files['csv_file']
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
                    uploaded_file.save(file_path)
                    data_path   = str(file_path).replace("\\", "/")  
                    data_target = request.form.get( "target_" ) 
                    data_target = data_target.lower().strip()
                    save_target_name( str(data_target.lower().strip()) ) 
                    data_cleaned = clean_data_processing( data_path ) 
                    all_columns = [ (i.lower().strip()) for i in data_cleaned.columns]
                    data_target = str(data_target)

                    if data_target in all_columns: 
                         if check_drop_columns: 
                              columns_to_drop = [i.strip().lower() for i in check_drop_columns.split(",")]
                              checking = drop_columns_check(columns_to_drop, all_columns)

                              if checking == True:
                                   data_cleaned = drop_columns_finally(columns=columns_to_drop, data_cleaned=data_cleaned)

                                   data_cleaned.to_csv("data_uploaded_saved/cleaned_data.csv", index=False) 
                                   feature_columns_ = features_columns(data = data_cleaned, target = data_target )

                                   json_data, json_data_encoded = json_dictionary(feature_columns=feature_columns_, data=data_cleaned) 
                                   save_json_data(file_to_save=json_data)
                                   save_json_data_encoded(file_to_save=json_data_encoded)

                                   data_encoded = encode_data(features=feature_columns_, process_data=data_cleaned, json_file=json_data_encoded)
                                   X, y = split_data_into_x_y(data = data_encoded, target = data_target)
                                   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 

                                   model_svc, model_lsv, model_lgr, model_dtc, model_rfc = instantiations() 
                                   model_svc, model_lsv, model_lgr, model_dtc, model_rfc = model_fitting(X_train, y_train,  model_svc, model_lsv, model_lgr, model_dtc, model_rfc)                             
                                   save_models(model_svc, model_lsv, model_lgr, model_dtc, model_rfc)

                                   accuracy_svc, accuracy_lsv, accuracy_lgr, accuracy_dtc, accuracy_rfc = check_accuracy(X_test, y_test, model_svc, model_lsv, model_lgr, model_dtc, model_rfc)
                                   accuracy_scores = [round(accuracy_svc*100,1), round(accuracy_lsv*100,1), round(accuracy_lgr*100,1), round(accuracy_dtc*100,1), round(accuracy_rfc*100,1)]
                              else:
                                   wrong_format = checking  
                                   all_columns.remove(data_target)
                                   all_columns_sting = ""
                                   for i in all_columns:
                                        all_columns_sting = i.title() + ", " + all_columns_sting 
                                   all_columns_sting = all_columns_sting[:-2]
                                   wrong_format = f'You entered  "{wrong_format.title()}", but it is not in the column. The columns are : {all_columns_sting}'
                                   return render_template('pjt1_gmlp.html', wrong_format_=wrong_format)
                         else: 
                              data_cleaned.to_csv("data_uploaded_saved/cleaned_data.csv", index=False) 
                              feature_columns_ = features_columns(data = data_cleaned, target = data_target )

                              json_data, json_data_encoded = json_dictionary(feature_columns=feature_columns_, data=data_cleaned) 
                              save_json_data(file_to_save=json_data)
                              save_json_data_encoded(file_to_save=json_data_encoded)

                              data_encoded = encode_data(features=feature_columns_, process_data=data_cleaned, json_file=json_data_encoded)
                              X, y = split_data_into_x_y(data = data_encoded, target = data_target)
                              X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 

                              model_svc, model_lsv, model_lgr, model_dtc, model_rfc = instantiations() 
                              model_svc, model_lsv, model_lgr, model_dtc, model_rfc = model_fitting(X_train, y_train,  model_svc, model_lsv, model_lgr, model_dtc, model_rfc)                             
                              save_models(model_svc, model_lsv, model_lgr, model_dtc, model_rfc)

                              accuracy_svc, accuracy_lsv, accuracy_lgr, accuracy_dtc, accuracy_rfc = check_accuracy(X_test, y_test, model_svc, model_lsv, model_lgr, model_dtc, model_rfc)
                              accuracy_scores = [round(accuracy_svc*100,1), round(accuracy_lsv*100,1), round(accuracy_lgr*100,1), round(accuracy_dtc*100,1), round(accuracy_rfc*100,1)]
                    else: 
                         columns = [ (i.lower().strip().title()) for i in all_columns]
                         wrong_format = f"Target should one of the following columns - {columns}" 
                         return render_template('pjt1_gmlp.html', wrong_format_=wrong_format)
               else: 
                    wrong_format = "Wrong Format. Upload a .csv, .xlsx, or xls file"
                    return render_template('pjt1_gmlp.html', wrong_format_=wrong_format)             

          if request.form['submit_button'] == 'start_predicting':
               json_data_encoded = load_json_data_encoded(path="data_uploaded_saved/json_data_encoded.json") 
               json_data         = load_json_data_encoded(path="data_uploaded_saved/json_data.json") 
               data_cleaned      = clean_data_processing( "data_uploaded_saved/cleaned_data.csv" )
               data_target       = read_saved_target_name()

               feature_columns_  = features_columns(data = data_cleaned, target = data_target )
               json_data, json_data_encoded = json_dictionary(feature_columns=feature_columns_, data=data_cleaned) 

               for i in feature_columns_: 
                    input_dictionary[i] = request.form.get(i)  
               input_decoded_ = decode_input(input_dictionary=input_dictionary, json_data_encoded=json_data_encoded) 

               input_decoded_ = data_to_np_array(input_decoded_)
               model_svc, model_lsv, model_lgr, model_dtc, model_rfc = load_models() 
               predict_svc, predict_lsv, predict_lgr, predict_dtc, predict_rfc = prediction(input_decoded_, model_svc, model_lsv, model_lgr, model_dtc, model_rfc)
               predicted_results = [predict_svc, predict_lsv, predict_lgr, predict_dtc, predict_rfc]

     return render_template('pjt1_gmlp.html', json_data_ = json_data, feature_column_=feature_columns_, predicted_result_=predicted_results, accuracy_scores_=accuracy_scores, model_types_=model_types)      



################################################################################################################################################


if __name__ == '__main__':  
    app.run(debug=True)


