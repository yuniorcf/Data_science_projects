
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTEN
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve

# import modules from keras librarie
from keras.models import Sequential
from keras.layers import Dense


# =========================================================================================================
# Function 'pie_plot'
# =========================================================================================================

def pie_plot(dataset, column, cutoff=18):
    '''
    Function to plot both posEntryMode and merchantCountry in a pie style
    
    Vars:
        column (str): name of column to analyze (e.g. 'posEntryMode' or 'merchantCountry')
        cutoff (int): integer to merge out lower incidence values. If lower than 10, the function will raise an error 
    '''
    if cutoff < 10:
        raise ValueError('Please try a higher cutoff value')
        
    else:
        
        subset = dataset[dataset['label']==1].groupby(column)[column].count().sort_values()

        ps=pd.Series()
        val=0
        i=0
        scode=[]

        for value in subset.values:
            if value <= cutoff:
                val += value
                i+=1
                scode.append(subset.index[i])
        ps.loc[0] = val
        ps = pd.concat([ps, subset.iloc[i:]], axis=0).rename(index={0: f'Others [{len(scode)}] '})

        # Plot
        plt.figure(figsize=(5, 5), dpi=80)
        explode = tuple([0.05]*len(ps))
        plt.pie(ps.values,
                autopct='%1.1f%%',
                textprops={'fontsize': 12},
                pctdistance=0.85,
                explode = explode
               )

        #draw circle
        centre_circle = plt.Circle((0,0),0.70,fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        labels=ps.index
        legend=plt.legend(labels, loc="center", fontsize=12,
                   bbox_transform=plt.gcf().transFigure)
        legend.get_title().set_fontsize('12')
        legend.set_title(column, prop = {'size':15})
        plt.axis('equal') 
        plt.tight_layout()
        plt.show()
        
# =========================================================================================================
# Function 'parse_data'
# =========================================================================================================

def parse_data(data, oversample=False, checkplot=False):
    '''
    This function encodes categorical variables either in dummy format or just numerical labels
    Vars:
        data (DataFrame): Data with both features and labels
        oversample (Boolean): define whether to perform imbalance correction on the dataset
        checkplot (Boolean): define whether to make a sanity bar plot in case over-sample is applied
    
    Outputs:
        X_train, X_test (DataFrame): train and test features datasets
        y_train, y_test (pandas Series): train and test label datasets
    '''
    
    # set to categorical
    cat_cols=['merchantCountry', 'posEntryMode', 'week_day', 'label']
    
    for col in cat_cols:
        data[col] = data[col].astype('category')
        
    cat_data = data[cat_cols]
    
            
    # Normalize numerical columns
    num_cols=['mcc', 'transactionAmount', 'availableCash']
    for col in num_cols:
        data[col] = data[col].apply(np.log1p)
    num_data=data[num_cols]
        
    X=pd.concat([num_data, cat_data], axis=1)
    X=X.drop('label', axis=1)
    y=data.label
    
    encoder=LabelEncoder()
          
    X['merchantCountry'] = encoder.fit_transform(X['merchantCountry'])
    X['posEntryMode'] = encoder.fit_transform(X['posEntryMode'])
    
    dummy_df=pd.DataFrame()
    dummy_df = pd.get_dummies(X['week_day'])
    
    # merge dummy data with X dataframe
    X = X.join(dummy_df)
    X = X.drop('week_day', axis=1)
    
    if oversample:
        
        over = SMOTEN(random_state=0, n_jobs=-1)
                        
        X_sm, y_sm = over.fit_resample(X, y)
        
        if checkplot:
            
            #plot
            lebel_count = data.label.value_counts()
            fig, axes = plt.subplots(1,2, figsize=(15, 5))
            lebel_count.plot(ax=axes[0], kind='bar', color=['b', 'r'])
            axes[0].set_title('Label Count on Original Data', fontsize=20)
            axes[0].set_xticklabels(['safe', 'fraud'], rotation=45, size=16)
            axes[0].set_ylabel('Counts', size=18)
            count=pd.DataFrame(y_sm).value_counts()
            count.plot(ax=axes[1], kind="bar", color=["b", "r"])
            axes[1].set_title('Label Count Upon Oversampling', fontsize=20)
            axes[1].set_xticklabels(['safe', 'fraud'], rotation=45, size=16)
            axes[1].set_xlabel('')
            plt.show()
        
        # split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.3, stratify=y_sm, random_state=42)
              
    
    else:
        
        # split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return X_train, X_test, y_train, y_test
    
    
# =========================================================================================================
# Function 'ANN'
# =========================================================================================================       
 
 
def ANN(shape):
    '''
    Function to define a Keras-based model
    Vars:
        shape (int): input dimension
    
    Output:
        model: defined model
    '''
    
    model = Sequential()
    model.add(Dense(100, input_dim=shape, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model 
    
    
    
 
# =========================================================================================================
# Function 'model_fit'
# =========================================================================================================

def model_fit(model, X_train, X_test, y_train, y_test, ANN=False, plot_ROC=True):
    '''
    This function perform model fitting.
    
    Vars:
        model: model to be fitted. It could be either a scikit-learn model or an Artificial Neuronal Network (ANN)
        X_train, X_test (DataFrame): train and test features datasets
        y_train, y_test (pandas Series): train and test label datasets
        ANN (Boolean): Set to True in case an ANN is going to be fitted (Default = False)
        plot_ROC (Boolean): Set to True if a ROC curve is to be displayed. (Default = False). This option is only available if model is not an ANN
    Outputs:
        history (keras.callbacks.History): History from both training and testing steps where accuracy and loss are stored
        acc_score (float): accuracy score if the fitted model is not an ANN 
    '''
    
    # Model fit
    if ANN:
        y_train=pd.get_dummies(y_train)
        y_test=pd.get_dummies(y_test)
        history = model.fit(X_train, y_train, epochs=180, validation_data=(X_test,y_test), verbose=1)
                                
        return history

    else:
        model.fit(X_train, y_train)
        label_predict_proba = model.predict_proba(X_test)[:,1]
        label_predict = model.predict(X_test)
        roc_auc = roc_auc_score(y_test, label_predict_proba)
        acc_score = accuracy_score(y_test, label_predict)
            
        print(f'roc_auc_score [{model}]: {roc_auc}')
        print(f'acc_score [{model}]: {acc_score}')
        print(confusion_matrix(y_test,label_predict))
        print(classification_report(y_test,label_predict))
            
        if plot_ROC:
            # plot no skill roc curve
            plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
            
            # calculate roc curve for model
            fpr, tpr, _ = roc_curve(y_test, label_predict_proba)
            
            # plot model roc curve
            plt.plot(fpr, tpr, marker='.', label='ROC AUC score: {:.3f}'.format(roc_auc))
            
            # axis labels
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            
            # show the legend
            plt.legend()
            
            # show the plot
            plt.show()
        return acc_score  
        
   
 
