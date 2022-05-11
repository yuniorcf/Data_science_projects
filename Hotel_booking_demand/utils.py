# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from keras.models import Sequential
from keras.layers import Dense, Dropout


# =========================================================================================================
# Function 'pairplot_check'
# =========================================================================================================

def pairplot_check(data_frame):
    '''
    This function creates a seaborn-based pairplot for visually inspect data integrity
    
    Vars:
        data_frame (pandas DataFrame): This is a pandas dataframe with only the two columns to be compared
        
    Outputs:
        plot
    '''
    
    with sns.plotting_context(rc={'axes.labelsize':14}):
        g=sns.pairplot(data_frame, diag_kind='auto')
        g.fig.set_size_inches(12,8)

    plt.tight_layout()


# =========================================================================================================
# Function 'parse_data'
# =========================================================================================================

def parse_data(data, skip_cols=False, col_list=None, normalize_data=False):
    '''
    This function prepares data for fitting. It creates the train, test dataset from the original data.
    In addition, can cormalize the numerical data and skip selected columns from the dataset
    
    Vars: 
        data (pandas DataFrame): Original data set to be processed
        
        skip_cols (bool): This is a boolean parameter to decide whether skip certain columns or not.
                          default = 'False'
        
        col_list (list): List of column names to skip from dataset
        
    Outputs:
        X_train, y_train, X_test (pandas DatFrame): Pandas data frames containing the training and test sets
    '''
    
    cat_cols = [col for col in data if np.isin(data[col].dropna().unique(), [0, 1]).all()]# Define categorical columns
    data_cat = data[cat_cols].drop('is_canceled', axis=1)# Categorical data
    data_num = data.drop(cat_cols, axis=1) # Numerical data
    
    # rename columns with large names on numerical dataset
    col_names={'arrival_date_week_number': 'ad_week_number',
               'arrival_date_day_of_month': 'ad_day_of_month',
               'arrival_date_month': 'ad_month',
               'stays_in_weekend_nights': 'nb_weekend_nights',
               'stays_in_week_nights': 'nb_week_nights',
               'previous_cancellations': 'prev_cancellations',
               'previous_bookings_not_canceled': 'prev_b_not_canceled',
               'required_car_parking_spaces': 'parking',
               'total_of_special_requests': 'special_requests'}
    
    data_num.rename(columns=col_names, inplace = True)
       
    
    if skip_cols:
        data_num = data_num.drop(col_list, axis=1) # Drop additional columns  
    
    
    if normalize_data:
        vars_to_normalize = ['lead_time', 'ad_week_number', 'ad_day_of_month', 'avg_daily_rate']
        #subset = data_num[vars_to_normalize]
        normalized_data=pd.DataFrame()  
        # Normalize numerical data
        for col_name in vars_to_normalize:
            normalized_data[col_name]=data_num[col_name].apply(np.log1p)

        # impute missing values on column 'avg_daily_rate'
        normalized_data['avg_daily_rate'] = normalized_data['avg_daily_rate'].fillna(value = normalized_data['avg_daily_rate'].mean())
        
        # put all numerical data together
        normalized_data = pd.concat([normalized_data, data_num.drop(['lead_time', 'ad_week_number', 'ad_day_of_month', 'avg_daily_rate'], axis=1)], axis=1)
    
        X = pd.concat([normalized_data, data_cat], axis = 1)
        
        
    else:
        X = pd.concat([data_num, data_cat], axis = 1)
        
    y = data.is_canceled

    # split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    return X_train, X_test, y_train, y_test

# =========================================================================================================
# Function 'model_fit'
# =========================================================================================================

def model_fit(model, X_train, y_train, X_test, y_test, cross_validate=False, cv_splits=None):
    '''
    This function performs the fitting of the predefined model. The function can perform
    cross validation in the training step for comparison purpose.
    
    Vars:
        model: Predefined model
        
        X_train, y_train, X_test, y_test (pandas DatFrame): Pandas data frames containing the training and test sets
        
        cross_validate (boolean): Optional argument to decide whether to perform cross validation or not
                                  default = 'False'
       
        cv_splits (int): number of validations in the cross validation algorithm
                                  default = 'None'
    
    Outputs:
        - If cross validation is used
        mean_score, min_score, max_score, std_dev (Float): Mean, minimum, maximum and standard deviation from cross validation score   
        
        - If cross validation is not used
        pred_score (Float): Score from prediction
        label_predict: predicted labels
    '''
    
    # instantiate the model
    if cross_validate:
        cv = StratifiedKFold(n_splits=cv_splits, random_state=42, shuffle=True)
        cv_results = cross_val_score(model, X_train, y_train, cv=cv, scoring = 'accuracy')
        min_score = round(np.min(cv_results), 4)
        max_score = round(np.max(cv_results), 4)
        std_dev = round(np.std(cv_results), 4)
        mean_score = round(np.mean(cv_results), 4)
        
        return mean_score, std_dev, min_score, max_score 
    else:
        model.fit(X_train, y_train)# perform model fitting 
        label_predict = model.predict(X_test)# Predict
        pred_score = accuracy_score(y_test,label_predict)# calculate the prediction score
        
        return pred_score, label_predict


# =========================================================================================================
# Function 'make_prediction'
# =========================================================================================================

def make_prediction(model, X_train, y_train, X_test, y_test, plot_confusion_matrix=True, plot_importances=True):
    '''
    This function call the predefined 'model_fit' function and uses its response variables to make a classification report.
    In addition, it can optionally plot the confusion matrix and/or feature importances
    Vars:
        model: Predefined model
        
        X_train, y_train, X_test (pandas DatFrame): Pandas data frames containing the training and test sets
        
        plot_confusion_matrix (boolean): whether to plot the confusion matrix from prediction or not
        
        plot_importances (boolean): whether to plot feature importances or not 
    '''
    
    pred_score, label_predict = model_fit(model, X_train, y_train, X_test, y_test)
    print('Accuracy Score rfc: ', pred_score)
    print()
    print(f'Classification report: \n {classification_report(y_test,label_predict)}')
    
    if plot_confusion_matrix:
        # plot confusion matrix
        cm = confusion_matrix(y_test, label_predict, labels=model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot()
        plt.title('Confusion Matrix', size=16)
        plt.grid(False)
        plt.show()
    
    if plot_importances:
        #plt.subplots(figsize=(8, 14))
        plt.subplots(figsize=(8, int(len(model.feature_importances_)/3)))
        feat_imp = pd.Series(model.feature_importances_, X_train.columns).sort_values(ascending=True)
        #feat_imp.plot(kind='barh', title='Feature Importances')
        feat_imp.plot(kind='barh', color='g')
        plt.title('Feature Importances', size=18)
        plt.xlabel('Importances', size=16)
        plt.ylabel('Features', size=16)
        plt.xticks(size=14)
        plt.yticks(size=14)
        plt.show()


def create_basemodel(shape):
    # create model
    
    model = Sequential()
    model.add(Dense(100, input_dim=shape, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='sigmoid'))
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
