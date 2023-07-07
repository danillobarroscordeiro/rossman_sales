import pandas as pd
pd.set_option('display.float_format', lambda x: '%.2f' % x)
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pickle
import os
import inflection
import math3
from datetime import datetime as dt
import datetime
from sklearn.preprocessing import RobustScaler, MinMaxScaler, LabelEncoder
import xgboost as xgb
import requests
import s3fs
import joblib

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
s3 = s3fs.S3FileSystem(
    anon=False, key=AWS_ACCESS_KEY_ID, secret=AWS_SECRET_ACCESS_KEY)

class Rossmann(object):
    def __init__(self):
        self_home_path = 's3://rossmann-sales/'
        self.competition_distance_scaler = joblib.load(
            open(self_home_path + 'parameter/competition_distance_scaler.pkl', 'rb')
        )
        self.competition_time_month_scaler = joblib.load(
            open(self_home_path + 'parameter/competition_time_month_scaler.pkl', 'rb')
            )
        self.promo2_time_week_scaler = joblib.load(
            open(self_home_path + 'parameter/promo2_time_week_scaler.pkl', 'rb')
            )
        self.year_scaler = joblib.load(
            open(self_home_path + 'parameter/year_scaler.pkl', 'rb')
            )
        self.store_type_scaler = joblib.load(
            open(self_home_path + 'parameter/store_type_scaler.pkl', 'rb')
            )
      
        
    def data_cleaning(self, df1):


## Renaming columns and defining columns types

        cols_old = [
            'Store', 'DayOfWeek', 
            'Date','Open', 'Promo',
            'StateHoliday', 'SchoolHoliday', 
            'StoreType', 'Assortment',
            'CompetitionDistance', 'CompetitionOpenSinceMonth',
            'CompetitionOpenSinceYear', 'Promo2', 
            'Promo2SinceWeek', 'Promo2SinceYear', 
            'PromoInterval'
        ]

        snakecase = lambda x: inflection.underscore(x)

        cols_new = list(map(snakecase, cols_old))

        df1.columns = cols_new

        df1['date'] = pd.to_datetime(df1['date'])


        # CompetitionDistance
        df1['competition_distance'] = df1['competition_distance'].apply(
            lambda x: x if pd.notnull(x) else 200000
        )

        #competition_open_since_month
        df1['competition_open_since_month'] = df1.apply(
            lambda x: x['competition_open_since_month']  
            if pd.notnull(x['competition_open_since_month'])
            else x['date'].month, axis=1
        ) 

        #competition_open_since_year
        df1['competition_open_since_year'] = df1.apply(
            lambda x: x['competition_open_since_year'] 
            if pd.notnull(x['competition_open_since_year'])
            else x['date'].year, axis=1
        ) 


        # Promo2SinceWeek 
        df1['promo2_since_week'] = df1.apply(
            lambda x: x['promo2_since_week'] 
            if pd.notnull(x['promo2_since_week'])
            else x['date'].week, axis=1
        ) 

        # Promo2SinceYear
        df1['promo2_since_year'] = df1.apply(
            lambda x: x['promo2_since_year'] 
            if pd.notnull(x['promo2_since_year'])
            else x['date'].year, axis=1
            ) 


        # PromoInterval

        month_map ={
            1: 'Jan',2: 'Fev', 3: 'Mar', 
            4: 'Apr', 5: 'May', 6: 'Jun', 
            7: 'Jul', 8: 'Aug', 9: 'Set', 
            10: 'Oct', 11: 'Nov',12: 'Dec'
        }
        df1['promo_interval'].fillna(0,inplace=True)
        df1['month_map'] = df1['date'].dt.month.map(month_map)

        df1['is_promo2'] = df1[['promo_interval','month_map']].apply(
            lambda x: 0 if x['promo_interval'] == 0
            else 1 if x['month_map'] in x['promo_interval'].split(',') 
            else 0, axis=1
        )

        ## Change column types

        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype(int)
        df1['competition_open_since_year'] = df1['competition_open_since_year'].astype(int)
        df1['promo2_since_week'] = df1['promo2_since_week'].astype(int)
        df1['promo2_since_year'] = df1['promo2_since_year'].astype(int)
        
        return df1

    def feature_engineering(self,df2):

        ## Feature Engineering

        # year
        df2['year']= df2['date'].dt.year

        #month
        df2['month']= df2['date'].dt.month

        #day
        df2['day']= df2['date'].dt.day

        #week of year
        df2['week_of_year'] = df2['date'].dt.isocalendar().week.astype(int)

        # year week
        df2['year_week'] = df2['date'].dt.strftime('%Y-%W')

        # competition_since

        df2['competition_since'] = df2.apply(
            lambda x: dt(year=x['competition_open_since_year'], 
            month=x['competition_open_since_month'], 
            day=1),axis=1
        ) 

        df2['competition_time_month'] = (
            (df2['date'] - df2['competition_since'])/30
        ).apply(lambda x: x.days).astype(int)


        #promo2 since

        df2['promo2_since'] = df2['promo2_since_year'].astype(str) 
        + '-' + df2['promo2_since_week'].astype(str)
        df2['promo2_since'] = df2['promo2_since'].apply(
            lambda x: dt.strptime(x + '-1','%Y-%W-%w') -  datetime.timedelta(days=7)
        )
        df2['promo2_time_week'] = (
            ((df2['date'] - df2['promo2_since']) / 7)
            .apply(lambda x:x.days)
            .astype(int)
        )

        #assortment
        df2['assortment'] = df2['assortment'].apply(
            lambda x: 'basic' if x == 'a' 
            else 'extra' if x == 'b' 
            else 'extended'
        )

        #state holiday

        df2['state_holiday'] = df2['state_holiday'].apply(
            lambda x: 'public_holiday' 
            if x == 'a' 
            else 'easter_holiday' if x == 'b' 
            else 'christmas' if x=='c' 
            else 'regular_day'
        )

# Feature filtering

## Line filtering

        df2 = df2[(df2['open']  != 0)]

        ## Column filtering

        cols_drop = ['open','promo_interval', 'month_map']
        df2 = df2.drop(cols_drop, axis=1)
        
        return df2
    
    def data_preparation(self, df5):


        ## Rescaling

        rescaling_df  = df5.select_dtypes(include=['int64','float64'])
        rs = RobustScaler()
        mms = MinMaxScaler()
        df5['competition_distance'] = self.competition_distance_scaler.transform(
            df5[['competition_distance']].values
        )
        df5['competition_time_month'] = self.competition_time_month_scaler.transform(
            df5[['competition_time_month']].values
        )
        df5['promo2_time_week'] = self.promo2_time_week_scaler.transform(
            df5[['promo2_time_week']].values
        )
        df5['year'] = self.year_scaler.transform(
            df5[['year']].values
        )


        ## Transformation

        ### Encoding

        #state holiday - One hot encoding
        df5 = pd.get_dummies(df5,prefix=['state_holiday'], columns=['state_holiday'])

        #store_type - target encoding
        df5['store_type'] = self.store_type_scaler.fit_transform(df5['store_type'])

        #assortment - ordinal encoding
        assortment_dict = {'basic' :1, 'extra':2, 'extended':3}
        df5['assortment'] = df5['assortment'].map(assortment_dict)

        ### Nature transformation

        #month
        df5['month_sin'] = df5['month'].apply(
            lambda x: np.sin(x * (2.*np.pi/12))
        )
        df5['month_cos'] = df5['month'].apply(
            lambda x: np.cos(x * (2.*np.pi/12))
        )

        #day
        df5['day_sin'] = df5['day'].apply(
            lambda x: np.sin(x * (2.*np.pi/30))
        )
        df5['day_cos'] = df5['day'].apply(
            lambda x: np.cos(x * (2.*np.pi/30))
        )

        #day of week
        df5['day_of_week_sin'] = df5['day_of_week'].apply(
            lambda x: np.sin(x * (2.*np.pi/7))
        )
        df5['day_of_week_cos'] = df5['day_of_week'].apply(
            lambda x: np.cos(x * (2.*np.pi/7))
        )

        #week of year
        df5['week_of_year_sin'] = df5['day_of_week'].apply(
            lambda x: np.sin(x*(2.*np.pi/52))
        )
        df5['week_of_year_cos'] = df5['day_of_week'].apply(
            lambda x: np.cos(x*(2.*np.pi/52))
        )
        
        cols_selected = [
            'store',
            'promo',
            'store_type',
            'assortment',
            'competition_distance',
            'competition_open_since_month',
            'competition_open_since_year',
            'promo2',
            'promo2_since_week',
            'promo2_since_year',
            'competition_time_month',
            'promo2_time_week',
            'month_cos',
            'month_sin',
            'day_sin',
            'day_cos',
            'day_of_week_sin',
            'day_of_week_cos',
            'week_of_year_sin',
            'week_of_year_cos'
        ]
        
        return df5[cols_selected]
    
    def get_prediction(self, model, original_data,test_data):
        prediction_values = model.predict(test_data)
        
        #join prediction into original data
        original_data['prediction'] = np.expm1(prediction_values)
        
        return original_data.to_json(orient='records')
