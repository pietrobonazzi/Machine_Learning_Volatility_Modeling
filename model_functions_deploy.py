# %% [markdown]
# # Machine Learning Volatility Modeling
# ## Master's Thesis - Empirical Study 
# ### Università della Svizzera italiana
# 
# Pietro Bonazzi - pietro.bonazzi@usi.ch
# 
# Volatility Models - v.4 - DEPLOY VERSION v.3

# %%
# Import packages 
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend 
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from scipy.stats import norm
from dieboldmariano import dm_test
from xgboost import XGBRegressor

# %% [markdown]
# ### Data Preparation

# %%
def prepare_data(data_stock, mkt_covariates, lags_RV = [1,5,22], lags_RQ = [1,5,22], RQ_add = True):
    '''
    This function prepares the data for the model.

    Parameters
    ----------
    data_stock : DataFrame
        DataFrame containing the stock data.
    mkt_covariates : DataFrame
        DataFrame containing the market covariates.
    lags_RV : list, optional
        List of lags for the realized variance. The default is [1,5,22].
    lags_RQ : list, optional    
        List of lags for the realized quarticity. The default is [1,5,22].
    RQ_add : bool, optional
        Boolean to decide whether to add the realized quarticity. The default is True.
    '''

    data_stock.index = pd.to_datetime(data_stock.index)
    data_stock['RV_'+  str(lags_RV[0])] = data_stock['RV'].rolling(window=lags_RV[0]).mean()
    data_stock['RV_'+  str(lags_RV[1])] = data_stock['RV_'+  str(lags_RV[0])].rolling(window=lags_RV[1]).mean()
    data_stock['RV_'+  str(lags_RV[2])] = data_stock['RV_'+  str(lags_RV[0])].rolling(window=lags_RV[2]).mean()

    if RQ_add == True: 
        data_stock['RQ_'+  str(lags_RQ[0])] = data_stock['RQ'].rolling(window=lags_RQ[0]).mean()
        data_stock['RQ_'+  str(lags_RQ[1])] = data_stock['RQ_'+  str(lags_RQ[0])].rolling(window=lags_RQ[1]).mean()
        data_stock['RQ_'+  str(lags_RQ[2])] = data_stock['RQ_'+  str(lags_RQ[0])].rolling(window=lags_RQ[2]).mean()

    mkt_covariates.index = pd.to_datetime(mkt_covariates.index)
    mkt_covariates['CHFUSD'] = mkt_covariates['CHFUSD'].pct_change()
    mkt_covariates['CHFEUR'] = mkt_covariates['CHFEUR'].pct_change()
    mkt_covariates['GSWISS10'] = mkt_covariates['GSWISS10'].diff()
    mkt_covariates['CCFASZE'] = mkt_covariates['CCFASZE'].diff()
    mkt_covariates['SFSNTC'] = mkt_covariates['SFSNTC'].diff()
    mkt_covariates['SSARON'] = mkt_covariates['SSARON'].diff()
    mkt_covariates = mkt_covariates[30:-1]

    data = pd.concat([data_stock, mkt_covariates], axis=1, join='inner')
    data['Returns'] = data['Returns'].shift(-1)
    data['RV'] = data['RV'].shift(-1)
    data['RQ'] = data['RQ'].shift(-1)
    data.dropna(inplace=True)
    data = pd.DataFrame(data)
    
    return data

# %%
def create_df_selected_features(data, features):

    # Select the features
    select_features = ['Returns', 'RV'] + features
    data = data[select_features]

    # Define the train, validation and test dates
    train_date = data.index[int(len(data)*0.6)] # 60% of the data
    test_date = data.index[-1] # 30% of the data

    # Split the data into train and test
    df_train = data.loc[:train_date]
    df_test = data.loc[train_date:]
    df_test = df_test.drop(df_test.index[0])

    return df_train, df_test

# %% [markdown]
# ## HAR 

# %% [markdown]
# ### HAR Benchmark

# %%
def har_bmk (df_train, df_test, plot = False):

    '''
    HAR benchmark model. Default RV lags are 1, 5 and 22.

    Parameters
    ----------
    df_train : pandas.DataFrame
        Train data.
    df_test : pandas.DataFrame
        Test data.
    plot : bool, optional

    '''

    model = LinearRegression()

    # Fit the model
    model.fit(df_train.drop(['Returns', 'RV'], axis=1), df_train['RV'])
    prediction_bmk = model.predict(df_test.drop(['Returns', 'RV'], axis=1))

    # Calculate the RMSE
    rmse = np.sqrt(mse(df_test['RV'], prediction_bmk))
    
    results_bmk = pd.DataFrame({'pred_bmk': prediction_bmk, 'RV': df_test['RV']})
    results_bmk.index = df_test.index


    # Calculate the violation ratio
    alpha = 0.05
    var = pd.DataFrame({'VAR': results_bmk['pred_bmk']* norm.ppf(alpha), 'Returns': df_test['Returns']})
    violation_ratio = (var['Returns'] < var['VAR']).sum()/len(df_test)

    #DMW test (not necessary for benchmark model)
    dmw_test = (0,0)

    if plot is True:
        plt.rcParams["font.family"] = "Times New Roman"
        fig, ax = plt.subplots(figsize=(11, 3))
        ax.plot(results_bmk['pred_bmk'], label='prediction', color='black', linewidth=0.9)
        ax.plot(results_bmk['RV'], label='RV', color='grey', linewidth=0.7)
        ax.set_title('HAR_benchmark')   
        ax.legend(loc='upper left')
        ax.set_xlabel('Date')
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        plt.show()
    

    return rmse, violation_ratio, dmw_test, results_bmk

# %% [markdown]
# ### HARQ Regularized

# %%
def har_regularized(df_train, df_test, results_bmk, regularization = 'Lasso', plot = False):
    '''
    HAR regularized model. Default RV lags are 1, 5 and 22.
    
    Parameters
    ----------
    df_train : pandas.DataFrame
        Train data.
    df_test : pandas.DataFrame
        Test data.
    results_bmk : pandas.DataFrame
        Results of the HAR benchmark model.
    regularization : str, optional
        Regularization method. The default is 'Lasso'.
    plot : bool, optional
    
    '''

    # Define type of model
    if regularization == 'Lasso':
        model = LassoCV() # Lasso linear regression with built-in cross-validation of the alpha parameter
    elif regularization == 'Ridge':
        model = RidgeCV()
    elif regularization == 'ElasticNet':
        model = ElasticNetCV()
    elif regularization == 'Linear':
        model = LinearRegression()    


    # Fit the model
    model.fit(df_train.drop(['Returns', 'RV'], axis=1), df_train['RV'])
    prediction = model.predict(df_test.drop(['Returns', 'RV'], axis=1))

    # Calculate the RMSE
    rmse = np.sqrt(mse(df_test['RV'], prediction))
    
    results = pd.DataFrame({'pred': prediction, 'RV': df_test['RV']})
    results.index = df_test.index

    # Calculate the violation ratio
    alpha = 0.05
    var = pd.DataFrame({'VAR': results['pred']* norm.ppf(alpha), 'Returns': df_test['Returns']})
    violation_ratio = (var['Returns'] < var['VAR']).sum()/len(df_test)


    # Calculate the Diebold-Mariano test
    df_dmw_test = pd.concat([results_bmk['pred_bmk'], results['pred'], df_test['RV']], axis=1)
    dmw_test = dm_test(df_dmw_test['RV'], df_dmw_test['pred_bmk'], df_dmw_test['pred'], one_sided=True)

    if plot is True:
        plt.rcParams["font.family"] = "Times New Roman"
        fig, ax = plt.subplots(figsize=(11, 3))
        ax.plot(results['pred'], label='prediction', color='black', linewidth=0.9)
        ax.plot(results['RV'], label='RV', color='grey', linewidth=0.7)
        ax.set_title('HAR_' + regularization)   
        ax.legend(loc='upper left')
        ax.set_xlabel('Date')
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        plt.show()
    

    return rmse, violation_ratio, dmw_test

# %% [markdown]
# ## LSTM
# ### LSTM Single Layer

# %%
# Function to prepare the data for the LSTM model 
def windowed_dataset(x_series, y_series, n_past):
    dataX, dataY = [], []
    for i in range((n_past-1), len(x_series)):
        start_idx = x_series.index[i-n_past+1]
        end_idx = x_series.index[i]
        a = x_series[start_idx:end_idx].values
        dataX.append(a)
        dataY.append(y_series[end_idx])
    return np.array(dataX), np.array(dataY)

# %%
def lstm_single_layer(df_train, df_test, results_bmk, neurons = 20, n_past = 22, batch_size = 32, plot = False): 

    '''
    LSTM single layer model.

    Parameters
    ----------
    df_train : pandas.DataFrame
        Train data.
    df_test : pandas.DataFrame
        Test data.
    results_bmk : pandas.DataFrame  
        Results of the HAR benchmark model.
    neurons : str, optional
        Number of neurons in the LSTM layer. The default is '20'.
    n_past : str, optional
        Number of past observations to use as input. The default is '22'.
    batch_size : str, optional
        Batch size. The default is '32'.
    plot : bool, optional   

    Inspired by: https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
    https://github.com/chibui191/bitcoin_volatility_forecasting

    '''


    np.random.seed(2024)
    n_dims = df_train.shape[1]-2
    x_train, y_train = windowed_dataset(df_train.drop(['Returns', 'RV'], axis=1), df_train['RV'], n_past)

    # Define the model
    lstm = tf.keras.models.Sequential([ 
        tf.keras.layers.InputLayer(input_shape=[n_past, n_dims]),

        tf.keras.layers.LSTM(neurons),
        
        tf.keras.layers.Dense(1)
    ])
    
    def rmse(y_true, y_pred):
        loss = backend.sqrt(backend.mean(backend.square((y_true - y_pred))))
        return loss

    lstm.compile(loss='mse', 
               optimizer="adam", 
               metrics=[rmse])
    
    #checkpoint = ModelCheckpoint('NN_models/lstm_single_layer.h5',
    #                            save_best_only=True,
    #                            monitor='val_rmse')

    early_stopping = EarlyStopping(patience=100,
                                  restore_best_weights=True,
                                  monitor='val_rmse')
    
    lstm_res = lstm.fit(x_train, y_train, 
                        callbacks=early_stopping,
                        validation_split=0.3, shuffle=True,
                        verbose=0, batch_size=batch_size, epochs=500)
    
    x_test, y_test = windowed_dataset(df_test.drop(['Returns', 'RV'], axis=1), df_test['RV'], n_past)
    x, returns = windowed_dataset(df_test.drop(['Returns', 'RV'], axis=1), df_test['Returns'], n_past)

    # Predictions
    pred = lstm.predict(x_test)
    df_results = pd.DataFrame({'RV': df_test['RV'].iloc[n_past-1:], 'pred': pred.flatten(), 'returns': returns})

    # Calculate the RMSE
    rmse = np.sqrt(mse(df_results['RV'], df_results['pred']))

    # Calculate the violation ratio
    alpha = 0.05
    var = pd.DataFrame({'VAR': df_results['pred']* norm.ppf(alpha), 'Returns': df_results['returns']})
    violation_ratio = (var['Returns'] < var['VAR']).sum()/len(df_results)

    # Calculate the Diebold-Mariano test
    df_dmw_test = pd.concat([results_bmk['pred_bmk'], df_results['pred'], df_results['RV']], axis=1).dropna()
    dmw_test = dm_test(df_dmw_test['RV'], df_dmw_test['pred_bmk'], df_dmw_test['pred'], one_sided=True)

    if plot == True:
        plt.rcParams["font.family"] = "Times New Roman"
        fig, ax = plt.subplots(figsize=(11, 3))
        ax.plot(df_results['pred'], label='prediction', color='black', linewidth=0.9)
        ax.plot(df_results['RV'], label='RV', color='grey', linewidth=0.7)
        ax.set_title('LSTM_single_layer')   
        ax.legend(loc='upper left')
        ax.set_xlabel('Date')
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        plt.show()

    return rmse, violation_ratio, dmw_test

# %% [markdown]
# ### LSTM 2-layers with dropout

# %%
def lstm_2_layers_with_dropout(df_train, df_test, results_bmk, neurons = [32,16], n_past = 22, batch_size = 32, dropout = 0.1, plot = False): 

    '''
    LSTM 2 layers with dropout model.

    Parameters
    ----------
    df_train : pandas.DataFrame
        Train data.
    df_test : pandas.DataFrame
        Test data.
    results_bmk : pandas.DataFrame
        Results of the HAR benchmark model.
    neurons : list, optional
        Number of neurons in the LSTM layers. The default is [32,16].
    n_past : int, optional
        Number of past observations to use as input. The default is 22.
    batch_size : int, optional
        Batch size. The default is 32.
    dropout : float, optional
        Dropout rate. The default is 0.1.
    plot : bool, optional
    '''

    np.random.seed(2024)
    n_dims = df_train.shape[1]-2
    x_train, y_train = windowed_dataset(df_train.drop(['Returns', 'RV'], axis=1), df_train['RV'], n_past)

    lstm = tf.keras.models.Sequential([ 
        tf.keras.layers.InputLayer(input_shape=[n_past, n_dims]),
        tf.keras.layers.BatchNormalization(), 

        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(neurons[0], return_sequences=True)),
        tf.keras.layers.Dropout(dropout),

        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(neurons[1])),
        tf.keras.layers.Dropout(dropout),

        tf.keras.layers.Dense(1)
    ])
    
    def rmse(y_true, y_pred):
        loss = backend.sqrt(backend.mean(backend.square((y_true - y_pred))))
        return loss

    lstm.compile(loss='mse', 
               optimizer="adam", 
               metrics=[rmse])
    
    #checkpoint = ModelCheckpoint('NN_models/lstm_2_layers_with_dropout.h5',
    #                            save_best_only=True,
    #                            monitor='val_rmse')

    early_stopping = EarlyStopping(patience=100,
                                  restore_best_weights=True,
                                  monitor='val_rmse')
    
    lstm_res = lstm.fit(x_train, y_train, 
                        callbacks= early_stopping,
                        validation_split=0.3, shuffle=True,
                        verbose=0, batch_size=batch_size, epochs=500)
    
    x_test, y_test = windowed_dataset(df_test.drop(['Returns', 'RV'], axis=1), df_test['RV'], n_past)
    x, returns = windowed_dataset(df_test.drop(['Returns', 'RV'], axis=1), df_test['Returns'], n_past)

    # Predictions
    pred = lstm.predict(x_test)
    df_results = pd.DataFrame({'RV': df_test['RV'].iloc[n_past-1:], 'pred': pred.flatten(), 'returns': returns})


    # Calculate the RMSE
    rmse = np.sqrt(mse(df_results['RV'], df_results['pred']))

    # Calculate the violation ratio
    alpha = 0.05
    var = pd.DataFrame({'VAR': df_results['pred']* norm.ppf(alpha), 'Returns': df_results['returns']})
    violation_ratio = (var['Returns'] < var['VAR']).sum()/len(df_results)

    # Calculate the Diebold-Mariano test
    df_dmw_test = pd.concat([results_bmk['pred_bmk'], df_results['pred'], df_results['RV']], axis=1).dropna()
    dmw_test = dm_test(df_dmw_test['RV'], df_dmw_test['pred_bmk'], df_dmw_test['pred'], one_sided=True)

    if plot == True:
        plt.rcParams["font.family"] = "Times New Roman"
        fig, ax = plt.subplots(figsize=(11, 3))
        ax.plot(df_results['pred'], label='prediction', color='black', linewidth=0.9)
        ax.plot(df_results['RV'], label='RV', color='grey', linewidth=0.7)
        ax.set_title('LSTM_2_layers_with_dropout') 
        ax.legend(loc='upper left')
        ax.set_xlabel('Date')
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        plt.show()

    return rmse, violation_ratio, dmw_test

# %%
def lstm_3_layers_with_dropout(df_train, df_test, results_bmk, neurons = [32,16,8], n_past = 22, batch_size = 32, dropout = 0.1, plot = False): 

    '''
    LSTM 3 layers with dropout model.

    Parameters
    ----------
    df_train : pandas.DataFrame
        Train data.
    df_test : pandas.DataFrame
        Test data.
    results_bmk : pandas.DataFrame
        Results of the HAR benchmark model.
    neurons : list, optional    
        Number of neurons in the LSTM layers. The default is [32,16,8].
    n_past : int, optional  
        Number of past observations to use as input. The default is 22. 
    batch_size : int, optional  
        Batch size. The default is 32.  
    dropout : float, optional
        Dropout rate. The default is 0.1.
    plot : bool, optional
    '''

    np.random.seed(2024)
    n_dims = df_train.shape[1]-2
    x_train, y_train = windowed_dataset(df_train.drop(['Returns', 'RV'], axis=1), df_train['RV'], n_past)

    lstm = tf.keras.models.Sequential([ 
        tf.keras.layers.InputLayer(input_shape=[n_past, n_dims]),
        tf.keras.layers.BatchNormalization(), 

        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(neurons[0], return_sequences=True)),
        tf.keras.layers.Dropout(dropout),

        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(neurons[1], return_sequences=True)),
        tf.keras.layers.Dropout(dropout),

        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(neurons[2])),
        tf.keras.layers.Dropout(dropout),

        tf.keras.layers.Dense(1)
    ])
    
    def rmse(y_true, y_pred):
        loss = backend.sqrt(backend.mean(backend.square((y_true - y_pred))))
        return loss

    lstm.compile(loss='mse', 
               optimizer="adam", 
               metrics=[rmse])
    
    #checkpoint = ModelCheckpoint('NN_models/lstm_2_layers_with_dropout.h5',
    #                            save_best_only=True,
    #                            monitor='val_rmse')

    early_stopping = EarlyStopping(patience=100,
                                  restore_best_weights=True,
                                  monitor='val_rmse')
    
    lstm_res = lstm.fit(x_train, y_train, 
                        callbacks= early_stopping,
                        validation_split=0.3, shuffle=True,
                        verbose=0, batch_size=batch_size, epochs=500)
    
    x_test, y_test = windowed_dataset(df_test.drop(['Returns', 'RV'], axis=1), df_test['RV'], n_past)
    x, returns = windowed_dataset(df_test.drop(['Returns', 'RV'], axis=1), df_test['Returns'], n_past)

    # Predictions
    pred = lstm.predict(x_test)
    df_results = pd.DataFrame({'RV': df_test['RV'].iloc[n_past-1:], 'pred': pred.flatten(), 'returns': returns})

    # Calculate the RMSE
    rmse = np.sqrt(mse(df_results['RV'], df_results['pred']))

    # Calculate the violation ratio
    alpha = 0.05
    var = pd.DataFrame({'VAR': df_results['pred']* norm.ppf(alpha), 'Returns': df_results['returns']})
    violation_ratio = (var['Returns'] < var['VAR']).sum()/len(df_results)

    # Calculate the Diebold-Mariano test
    df_dmw_test = pd.concat([results_bmk['pred_bmk'], df_results['pred'], df_results['RV']], axis=1).dropna()
    dmw_test = dm_test(df_dmw_test['RV'], df_dmw_test['pred_bmk'], df_dmw_test['pred'], one_sided=True)

    if plot == True:
        plt.rcParams["font.family"] = "Times New Roman"
        fig, ax = plt.subplots(figsize=(11, 3))
        ax.plot(df_results['pred'], label='prediction', color='black', linewidth=0.9)
        ax.plot(df_results['RV'], label='RV', color='grey', linewidth=0.7)
        ax.set_title('LSTM_3_layers_with_dropout') 
        ax.legend(loc='upper left')
        ax.set_xlabel('Date')
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        plt.show()

    return rmse, violation_ratio, dmw_test

# %%
def lstm_3_layers_with_dropout_nobidirectional(df_train, df_test, results_bmk, neurons = [32,16,8], n_past = 22, batch_size = 32, dropout = 0.1, plot = False): 

    '''
    LSTM 3 layers with dropout model without bidirectional layers.

    Parameters
    ----------
    df_train : pandas.DataFrame
        Train data.
    df_test : pandas.DataFrame
        Test data.
    results_bmk : pandas.DataFrame
        Results of the HAR benchmark model.
    neurons : list, optional    
        Number of neurons in the LSTM layers. The default is [32,16,8].
    n_past : int, optional  
        Number of past observations to use as input. The default is 22. 
    batch_size : int, optional  
        Batch size. The default is 32.  
    dropout : float, optional
        Dropout rate. The default is 0.1.
    plot : bool, optional
    '''

    np.random.seed(2024)
    n_dims = df_train.shape[1]-2
    x_train, y_train = windowed_dataset(df_train.drop(['Returns', 'RV'], axis=1), df_train['RV'], n_past)

    lstm = tf.keras.models.Sequential([ 
        tf.keras.layers.InputLayer(input_shape=[n_past, n_dims]),
        tf.keras.layers.BatchNormalization(), 

        tf.keras.layers.LSTM(neurons[0], return_sequences=True),
        tf.keras.layers.Dropout(dropout),

        tf.keras.layers.LSTM(neurons[1], return_sequences=True),
        tf.keras.layers.Dropout(dropout),

        tf.keras.layers.LSTM(neurons[2]),
        tf.keras.layers.Dropout(dropout),

        tf.keras.layers.Dense(1)
    ])
    
    def rmse(y_true, y_pred):
        loss = backend.sqrt(backend.mean(backend.square((y_true - y_pred))))
        return loss

    lstm.compile(loss='mse', 
               optimizer="adam", 
               metrics=[rmse])
    
    #checkpoint = ModelCheckpoint('NN_models/lstm_2_layers_with_dropout_nobidirectional.h5',
    #                            save_best_only=True,
    #                            monitor='val_rmse')

    early_stopping = EarlyStopping(patience=100,
                                  restore_best_weights=True,
                                  monitor='val_rmse')
    
    lstm_res = lstm.fit(x_train, y_train, 
                        callbacks= early_stopping,
                        validation_split=0.3, shuffle=True,
                        verbose=0, batch_size=batch_size, epochs=500)
    
    x_test, y_test = windowed_dataset(df_test.drop(['Returns', 'RV'], axis=1), df_test['RV'], n_past)
    x, returns = windowed_dataset(df_test.drop(['Returns', 'RV'], axis=1), df_test['Returns'], n_past)

    # Predictions
    pred = lstm.predict(x_test)
    df_results = pd.DataFrame({'RV': df_test['RV'].iloc[n_past-1:], 'pred': pred.flatten(), 'returns': returns})

    # Calculate the RMSE
    rmse = np.sqrt(mse(df_results['RV'], df_results['pred']))

    # Calculate the violation ratio
    alpha = 0.05
    var = pd.DataFrame({'VAR': df_results['pred']* norm.ppf(alpha), 'Returns': df_results['returns']})
    violation_ratio = (var['Returns'] < var['VAR']).sum()/len(df_results)

    # Calculate the Diebold-Mariano test
    df_dmw_test = pd.concat([results_bmk['pred_bmk'], df_results['pred'], df_results['RV']], axis=1).dropna()
    dmw_test = dm_test(df_dmw_test['RV'], df_dmw_test['pred_bmk'], df_dmw_test['pred'], one_sided=True)

    if plot == True:
        plt.rcParams["font.family"] = "Times New Roman"
        fig, ax = plt.subplots(figsize=(11, 3))
        ax.plot(df_results['pred'], label='prediction', color='black', linewidth=0.9)
        ax.plot(df_results['RV'], label='RV', color='grey', linewidth=0.7)
        ax.set_title('LSTM_3_layers_with_dropout_nobidirectional') 
        ax.legend(loc='upper left')
        ax.set_xlabel('Date')
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        plt.show()

    return rmse, violation_ratio, dmw_test

# %% [markdown]
# ### MLP

# %%
def mlp_with_dropout(df_train, df_test, results_bmk, neurons = [32,16,8], n_past = 22, batch_size = 32, dropout = 0.1, plot = False): 

    '''
    MLP (multi layer perceptron) with dropout. 

    Parameters
    ----------
    df_train : pandas.DataFrame
        Train data.
    df_test : pandas.DataFrame
        Test data.
    results_bmk : pandas.DataFrame
        Results of the HAR benchmark model.
    neurons : list, optional
        Number of neurons in the MLP layers. The default is [32,16,8].
    n_past : int, optional
        Number of past observations to use as input. The default is 22.
    batch_size : int, optional
        Batch size. The default is 32.
    dropout : float, optional
        Dropout rate. The default is 0.1.
    plot : bool, optional
    ''' 

    np.random.seed(2024)
    n_dims = df_train.shape[1]-2
    x_train, y_train = windowed_dataset(df_train.drop(['Returns', 'RV'], axis=1), df_train['RV'], n_past)

    model = tf.keras.models.Sequential([

        tf.keras.layers.InputLayer(input_shape=[n_past, n_dims]),
        tf.keras.layers.BatchNormalization(), 

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(neurons[0]),
        tf.keras.layers.Dropout(dropout),

        tf.keras.layers.Dense(neurons[1]),
        tf.keras.layers.Dropout(dropout),

        tf.keras.layers.Dense(neurons[2]),
        tf.keras.layers.Dropout(dropout),

        tf.keras.layers.Dense(1)
    ])


    def rmse(y_true, y_pred):
        loss = backend.sqrt(backend.mean(backend.square((y_true - y_pred))))
        return loss

    model.compile(loss='mse', 
               optimizer="adam", 
               metrics=[rmse])
    
    #checkpoint = ModelCheckpoint('NN_models/mlp_with_dropout.h5',
    #                            save_best_only=True,
    #                            monitor='val_rmse')

    early_stopping = EarlyStopping(patience=100,
                                  restore_best_weights=True,
                                  monitor='val_rmse')
    
    lstm_res = model.fit(x_train, y_train, 
                        callbacks= early_stopping,
                        validation_split=0.3, shuffle=True,
                        verbose=0, batch_size=batch_size, epochs=500)
    
    x_test, y_test = windowed_dataset(df_test.drop(['Returns', 'RV'], axis=1), df_test['RV'], n_past)
    _, returns = windowed_dataset(df_test.drop(['Returns', 'RV'], axis=1), df_test['Returns'], n_past)

    # Predictions
    pred = model.predict(x_test)
    df_results = pd.DataFrame({'RV': df_test['RV'].iloc[n_past-1:], 'pred': pred.flatten(), 'returns': returns})

    # Calculate the RMSE
    rmse = np.sqrt(mse(df_results['RV'], df_results['pred']))

    # Calculate the violation ratio
    alpha = 0.05
    var = pd.DataFrame({'VAR': df_results['pred']* norm.ppf(alpha), 'Returns': df_results['returns']})
    violation_ratio = (var['Returns'] < var['VAR']).sum()/len(df_results)

    # Calculate the Diebold-Mariano test
    df_dmw_test = pd.concat([results_bmk['pred_bmk'], df_results['pred'], df_results['RV']], axis=1).dropna()
    dmw_test = dm_test(df_dmw_test['RV'], df_dmw_test['pred_bmk'], df_dmw_test['pred'], one_sided=True)

    if plot == True:
        plt.rcParams["font.family"] = "Times New Roman"
        fig, ax = plt.subplots(figsize=(11, 3))
        ax.plot(df_results['pred'], label='prediction', color='black', linewidth=0.9)
        ax.plot(df_results['RV'], label='RV', color='grey', linewidth=0.7)
        ax.set_title('MLP_with_dropout')
        ax.legend(loc='upper left')
        ax.set_xlabel('Date')
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        plt.show()

    return rmse, violation_ratio, dmw_test

# %% [markdown]
# ### XGBoost

# %%
def xgboost(df_train, df_test, results_bmk, n_estimators = 1000, learning_rate=0.01, plot = False):

    # Define the model
    model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, n_jobs=10, random_state=2024)

    # Fit the model
    model.fit(df_train.drop(['Returns', 'RV'], axis=1), df_train['RV'])
    prediction = model.predict(df_test.drop(['Returns', 'RV'], axis=1))

    # Calculate the RMSE
    rmse = np.sqrt(mse(df_test['RV'], prediction))
    
    results = pd.DataFrame({'pred': prediction, 'RV': df_test['RV']})
    results.index = df_test.index


    # Calculate the violation ratio
    alpha = 0.05
    var = pd.DataFrame({'VAR': results['pred']* norm.ppf(alpha), 'Returns': df_test['Returns']})
    violation_ratio = (var['Returns'] < var['VAR']).sum()/len(df_test)


    # Calculate the Diebold-Mariano test
    df_dmw_test = pd.concat([results_bmk['pred_bmk'], results['pred'], df_test['RV']], axis=1)
    dmw_test = dm_test(df_dmw_test['RV'], df_dmw_test['pred_bmk'], df_dmw_test['pred'], one_sided=True)

    if plot is True:
        plt.rcParams["font.family"] = "Times New Roman"
        fig, ax = plt.subplots(figsize=(11, 3))
        ax.plot(results['pred'], label='prediction', color='black', linewidth=0.9)
        ax.plot(results['RV'], label='RV', color='grey', linewidth=0.7)
        ax.set_title('XGBoost')   
        ax.legend(loc='upper left')
        ax.set_xlabel('Date')
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        plt.show()
    

    return rmse, violation_ratio, dmw_test


