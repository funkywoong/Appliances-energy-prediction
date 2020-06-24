import numpy as np
import pandas as pd 
import time, warnings, os

from datetime import date
from sklearn import preprocessing, model_selection, metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import MinMaxScaler

# Wrap the features in list objects
col_temp = ['T_kitchen', 'T_living', 'T_Laundry', 'T_office', 'T_bath', 
            'T_outside', 'T_ironing', 'T_teenager', 'T_parent']
col_hum = ['RH_kitchen','RH_living','RH_laundry','RH_office','RH_bath',
           'RH_outside','RH_ironing','RH_teenager','RH_parent']
col_weather = ["T_out", "Tdewpoint","RH_out","Press_mm_hg",
                "Windspeed","Visibility"] 
col_light = ["lights"]
col_randoms = ["rv1", "rv2"]
col_target = ["Appliances"]
col_date = ['date']
col_add = ['time']

def rename_col(raw_data):
    """ Rename the columns' name for using easily
    """

    raw_data = raw_data.rename(
        {
            'T1':'T_kitchen', 'RH_1':'RH_kitchen',
            'T2':'T_living', 'RH_2':'RH_living',
            'T3':'T_Laundry', 'RH_3':'RH_laundry',
            'T4':'T_office', 'RH_4':'RH_office',
            'T5':'T_bath', 'RH_5':'RH_bath',
            'T6':'T_outside', 'RH_6':'RH_outside',
            'T7':'T_ironing', 'RH_7':'RH_ironing',
            'T8':'T_teenager', 'RH_8':'RH_teenager',
            'T9':'T_parent', 'RH_9':'RH_parent'
            
        }, axis='columns'
    )

    return raw_data

def gen_time_col(raw_data):
    """ Parsing the string data in 'date' column
    """
    temp_time = []

    date_df = raw_data[col_date]
    date_df.sort_values('date', inplace=True)

    item_split = [item[0].split(' ') for item in date_df.values]
    for _, each_time in item_split:
        # Generate each row's time
        hour, minute, second = each_time.split(':')
        hour_minute = '{}.{}'.format(hour, minute)
        temp_time.append(float(hour_minute)/24)
    
    raw_data['time'] = temp_time
    
    return raw_data

def eval_model(X_train, X_test_scale, y_test, tuned_nor_model):
    """ Evaluation the training model 
    """
    score = metrics.r2_score(y_test, tuned_nor_model.predict(X_test_scale))
    print('----------------------------------result----------------------------------')
    print(tuned_nor_model)
    print('\nr2 score : {}\n'.format(score))

    # Get which features affect on training 
    importances = tuned_nor_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_importances = [(X_train.columns[i], importances[i])
                            for i in indices]
    
    print('Top 5 features')
    count = 0
    for importance in top_importances:
        print('feature name : {}, importance ratio : {}'.format(importance[0], importance[1]))
        count = count + 1

        if count == 5: break
    print('---------------------------------------------------------------------------')

def main():
    """ Main function to execute the whole training process
    """

    # set the directory path
    data_file_name = os.listdir('./data')[0]
    dir_path = os.getcwd()
    data_path = dir_path + '/data/' + data_file_name

    # load csv data file in a script
    raw_data = pd.read_csv(data_path)

    # Rename all columns
    raw_data = rename_col(raw_data)

    # Generate time column
    raw_data = gen_time_col(raw_data)

    # Split data set to train and test data set
    train, test = train_test_split(raw_data, test_size=0.25, random_state=40)
    feature_vars = train[col_temp + col_hum + col_weather + col_light + col_randoms + col_add]
    target_vars = train[col_target]

    X_train = train[feature_vars.columns]
    y_train = train[target_vars.columns]
    X_test = test[feature_vars.columns]
    y_test = test[target_vars.columns]

    # Drop the unnecessary columns
    X_train.drop(['rv1', 'rv2', 'Visibility', 'lights'], axis=1, inplace=True)
    X_test.drop(['rv1', 'rv2', 'Visibility', 'lights'], axis=1, inplace=True)

    # Normalize the train, test dataset
    scaler = MinMaxScaler()
    X_train_scale = scaler.fit_transform(X_train)
    X_test_scale = scaler.transform(X_test)

    # Initiate the model object 
    tuned_nor_model = ExtraTreesRegressor(
        bootstrap=False, ccp_alpha=0.0, criterion='mse',
        max_depth=80, max_features='sqrt', max_leaf_nodes=None,
        max_samples=None, min_impurity_decrease=0.0,
        min_impurity_split=None, min_samples_leaf=1,
        min_samples_split=2, min_weight_fraction_leaf=0.0,
        n_estimators=150, n_jobs=None, oob_score=False,
        random_state=40, verbose=0, warm_start=False
    )

    # Training the model
    print('\ntraining . . .\n')
    tuned_nor_model.fit(X_train_scale, y_train)

    # Evaluation of the model
    eval_model(X_train, X_test_scale, y_test, tuned_nor_model)

if __name__ == '__main__':
    main()
    

    




    