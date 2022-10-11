import lightgbm
import pandas
import yaml
import optuna
import joblib
import numpy as np
from pickle import dump
import lightgbm_label_creation as llc
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.impute import KNNImputer
import data_invest
import urllib
import warnings

import keras_tuner as kt
import optuna

from keras.backend import clear_session
from keras.datasets import mnist
from keras.layers import Conv2D
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Bidirectional
from keras.layers import Flatten
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from keras.metrics import MeanAbsoluteError
from keras.losses import BinaryCrossentropy
from keras.losses import MeanAbsoluteError
from keras.optimizers import Adam
from keras.models import load_model
import nasdaq
import random
class predict():
    def __init__(self, days):
        with open('configs.yml', 'r') as stream:
            self.config = yaml.safe_load(stream)
        self.database = data_invest.database()

        self.data = pandas.DataFrame()
        rand_list = []
        n = 500
        nas = nasdaq.return_nadaq()
        for i in range(n):
            rand_list.append(random.randint(0, len(nas)))
        nas_new = [nas[i] for i in rand_list]
        print(nas_new)

        years = [2016,2017,2018,2019,2020,2021,2022]
        for year in years:
            d = [self.data, self.database.training_data(year, nas_new)]
            self.data = pandas.concat(d)
            print(year)
            print(len(self.data))

        self.data = self.data.sort_values(by=['symbol', 'Date'], ascending=[True, True])
        self.data = self.data.reset_index(drop=True)
        # self.data = pandas.read_csv('data/train_data.csv', dtype=self.config['dtypes'])
        # self.data.dropna(thresh=35, inplace=True)
        print('finished_loading_data')
        # dtypr = self.data.dtypes
        # for each in dtypr:
        #     print(each)
        # for each in self.data.columns:
        #     print(each)
        self.data.replace('None', 0, inplace=True)
        # print(self.data.memory_usage(deep=True))
        # self.data[self.config['numeric_columns']] = self.data[self.config['numeric_columns']].apply(pandas.to_numeric, downcast="float",errors='coerce')
        # print(self.data.dtypes)
        # print(self.data.memory_usage(deep=True))
        self.columns = self.data.columns.tolist()
        self.scaled_data = None
        self.scaler = MinMaxScaler()
        self.X_train = None
        self.algorithm = 'binary'
        self.llc = llc.create_label()
        self.days = days
        self.X_pred = None
        self.N_TRAIN_EXAMPLES = 3000
        self.N_VALID_EXAMPLES = 1000
        self.BATCHSIZE = 128
        self.CLASSES = 10
        self.EPOCHS = 10
        self.label_columns = None
        self.label_columns_indices = None
        self.column_indices = None
        # Work out the window parameters.
        self.input_width = None
        self.label_width = None
        self.shift = None
        self.total_window_size = None
        self.input_slice = None
        self.input_indices = None
        self.label_start = None
        self.labels_slice = None
        self.label_indices = None
    #def EDA(self):
        #profile = ProfileReport(self.data, title="Pandas Profiling Report", explorative=True)
        #profile.to_file("alpha_eda.html")

    def data_prep(self, data):
        for each in data.columns:
            data[each] = pandas.to_numeric(data[each], downcast="float", errors='coerce')
        data.drop(['Date', 'reportedCurrency', 'reportedCurrencyINCOME_STATEMENT',
                   'symbol', 'remove'], axis=1, inplace=True)
        print('start imputer')
        # imputer = KNNImputer(n_neighbors=2, weights="uniform", copy=False)
        # imputer.fit_transform(data)
        print('stop imputer')
        data = data.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
        return data

    def normalization(self, data, scale):
        if scale == 'MinMax':
            self.scaler = MinMaxScaler()
        elif scale == 'Standard':
            self.scaler = StandardScaler()
        self.scaler.fit(data)
        dump(self.scaler, open('model/scaler.pkl', 'wb'))

    # def label_normalization(self, data):
    #     y = np.array(data)
    #     y = y.reshape(-1, 1)
    #     y = self.scaler.transform(y)
    #     y = y.reshape(1, -1)[0]
    #     return y

    def label_normalization(self, y):
        self.label_scaler = StandardScaler()
        y = np.array(y)
        print(y)
        y = y.reshape(-1, 1)
        print(y)
        self.label_scaler.fit(y)
        dump(self.label_scaler, open('model/label_scaler.pkl', 'wb'))


    def pca_transform(self):
        self.X_train = np.nan_to_num(self.X_train, copy=True, nan=0.0, posinf=None, neginf=None)
        self.X_valid = np.nan_to_num(self.X_valid, copy=True, nan=0.0, posinf=None, neginf=None)
        #self.pca = PCA(.90, svd_solver='full')
        self.pca = PCA(n_components = 35, svd_solver='full')
        self.pca.fit(self.X_train)
        #
        # print(self.pca.n_features_)
        # print(self.pca.n_samples_)
        # print(self.pca.components_)
        # print(self.pca.explained_variance_)
        # print(self.pca.explained_variance_ratio_)

        self.X_train_pca = self.pca.transform(self.X_train)
        self.X_valid_pca = self.pca.transform(self.X_valid)


    def feature_engineering(self):
        data = self.data.copy()
        data['diff_from_SMA'] = data['5adjustedclose'] - data['SMA']
        data = data.reindex(sorted(data.columns), axis=1)
        self.data = data
        # Work out the label column indices.

    def trained_data(self):
        label = 'percentage_change'
        data = self.data.copy()
        y = self.llc.t2_close(data, self.days, label)
        print(data.index)
        print(y.index)
        self.y_pred = y[y[label] == 0]
        y = y[y[label] != 0]

        # self.y_pred = self.y_pred['tomorrows_close']
        self.X_pred = data.iloc[self.y_pred.index, :]

        print(len(self.X_pred))
        print(len(self.y_pred))

        data = data.iloc[y.index, :]
        #y = y.iloc[data.index, :]

        print(len(y))
        print(len(data))

        # self.X_test = data[data['Date'] >= '2022-08-15']
        # self.y_test = y[y['Date'] >= '2022-08-15']
        # self.y_test = self.y_test['tomorrows_close']

        # data = data[data['Date'] < '2022-08-15']
        # y = y[y['Date'] < '2022-08-15']
        y = y[label]

        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(data, y, test_size=0.1)
        # self.X_valid, self.X_test, self.y_valid, self.y_test = train_test_split(self.X_valid, self.y_valid, test_size=0.40)
        self.non_normal_X_test = self.X_valid.copy()
        self.non_normal_X_pred = self.X_pred.copy()

        self.X_train = self.data_prep(self.X_train)
        self.X_valid = self.data_prep(self.X_valid)
        # self.X_test = self.data_prep(self.X_test)
        self.X_pred = self.data_prep(self.X_pred)

        self.pcaColumns = self.X_train.columns

        self.normalization(self.X_train, 'Standard')
        self.X_train = self.scaler.transform(self.X_train)
        self.X_valid = self.scaler.transform(self.X_valid)
        # self.X_test = self.scaler.transform(self.X_test)
        self.X_pred = self.scaler.transform(self.X_pred)

        #self.pca_data = self.X_train.copy()
        self.label_normalization(self.y_train)
        self.y_train = self.label_scaler.transform(np.array(self.y_train).reshape(-1, 1))
        print(self.y_train)
        # self.test_data = lgb.Dataset(self.X_test, label=self.y_test)
    def build_model(self, hp):
        model = Sequential()
        model.add(Embedding(input_dim=len(self.X_train_pca[0]), output_dim=hp.Int('output_dm',min_value=5,max_value=50,step=5))),
       # model.add(Embedding(input_dim=len(self.X_train_pca[0]), output_dim=5))
        model.add(GRU(hp.Int('input_unit',min_value=10,max_value=200,step=10),return_sequences=True))
       #model.add(GRU(30,return_sequences=True))
        for i in range(hp.Int('n_layers', 1, 4)):
            model.add(LSTM(hp.Int(f'lstm_{i}_units',min_value=32,max_value=512,step=32),return_sequences=True))
        model.add(LSTM(hp.Int('layer_2_neurons',min_value=32,max_value=512,step=32)))
        model.add(Dropout(hp.Float('Dropout_rate',min_value=0,max_value=0.3,step=0.1)))
        model.add(Dense(1, activation=hp.Choice('dense_activation',values=['relu', 'sigmoid'],default='relu')))
        model.compile(loss=MeanAbsoluteError(),
                      optimizer=Adam(1e-4),
                      metrics=MeanAbsoluteError())
        model.summary()
        return model
    def build_simple_model(self):

        model = Sequential()
        model.add(Embedding(input_dim=len(self.X_train_pca[0]), output_dim=20))
        model.add(GRU(20),return_sequences=True)
        # for i in range(hp.Int('n_layers', 1, 4)):
        #     model.add(LSTM(hp.Int(f'lstm_{i}_units',min_value=32,max_value=512,step=32),return_sequences=True))
        # model.add(LSTM(hp.Int('layer_2_neurons',min_value=32,max_value=512,step=32)))
        model.add(Dropout(.1))
        model.add(Dense(1, activation='relu'))
        model.compile(loss=MeanAbsoluteError(),
                      optimizer=Adam(1e-4),
                      metrics=MeanAbsoluteError())
        model.summary()
        return model

    def train_simple_algorithm(self):
        epocheroos = 20
        model = self.build_simple_model()
        history = model.fit(self.X_train_pca, self.y_train, epochs=epocheroos, validation_split=0.2)
    def train_algorithm(self):
        epocheroos = 100
        tuner = kt.Hyperband(
            hypermodel=self.build_model,
            objective= kt.Objective(name="val_mean_absolute_error",direction="min"),
            max_epochs=epocheroos,
            factor=3,
            hyperband_iterations=1,
            directory='neural_optimizer',
            project_name='hyperband',
            overwrite= False
        )
        stop_early = EarlyStopping(monitor='val_mean_absolute_error', patience=5)
        print("Starting Tuner Search")
        tuner.search(self.X_train_pca, self.y_train, epochs=epocheroos, validation_split=0.2, callbacks=[stop_early])

        # # Get the optimal hyperparameters
        print("Starting Get Best Hyperparameters")
        best_hps = tuner.get_best_hyperparameters(num_trials=3)[0]

        # Build the model with the optimal hyperparameters and train it on the data for x epochs
        print("Building model with Best Hyperparameters")
        model = tuner.hypermodel.build(best_hps)
        history = model.fit(self.X_train_pca, self.y_train, epochs=epocheroos, validation_split=0.2)

        print("Getting Best Epoch from the history")
        val_acc_per_epoch = history.history['val_mean_absolute_error']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
        print('Best epoch: %d' % (best_epoch,))

        hypermodel = tuner.hypermodel.build(best_hps)

        # Retrain the model
        print("Retraining with best Epoch")
        hypermodel.fit(self.X_train_pca, self.y_train, epochs=best_epoch, validation_split=0.2)

        print("Starting Eval of best Model")
        eval_result = hypermodel.evaluate(self.X_valid_pca, self.y_valid)
        print("[test loss, test accuracy]:", eval_result)
        hypermodel.save(f'model/neural_net_model_01_{days}_days')
        self.model = hypermodel

    def predict_algorithm(self,days):
        #model = load_model(f'model/neural_net_model_01_{days}_days')
        pred_01 = self.model.predict(self.X_valid_pca)
        predictions= []
        for each in pred_01:
            j = sum(each) / len(each)
            print(j)
            predictions.append(j)
        print(predictions)
        transformed_pred =self.label_scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

        self.non_normal_X_test['pred_01'] = transformed_pred
        self.non_normal_X_test['truth'] = self.y_valid

        self.non_normal_X_test =self.non_normal_X_test.loc[:,
              ['Date', 'symbol', '5adjustedclose','pred_01','truth']]

        self.non_normal_X_test = self.non_normal_X_test.rename_axis('idx').sort_values(by=['symbol', 'Date'], ascending=[True, True])
        print(self.non_normal_X_test)
        print(len(self.non_normal_X_test))
        self.non_normal_X_test.to_csv(f'data/neural_net_predict_test_{days}_days.csv')

    def go(self):
        self.feature_engineering()
        self.trained_data()
        self.pca_transform()
        self.train_algorithm()
        #self.train_simple_algorithm()
        #self.build_simple_model()
        self.predict_algorithm(days)

if __name__ == "__main__":
    days = 30
    p = predict(days)
    p.go()


