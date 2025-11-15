import numpy as np
import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    import logging
    logging.basicConfig(level=logging.INFO)

    # Pisahkan fitur dan target
    X = data.drop(columns=['Potability'])
    y = data['Potability']

    # Pipeline Preprocessing
    preprocessing_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Cleaning + Scaling
    X_preprocessed = preprocessing_pipeline.fit_transform(X)

    # Simpan Pipeline
    dump(preprocessing_pipeline, 'preprocessing_pipeline.joblib')

    # Gabungkan lagi data
    X_preprocessed = pd.DataFrame(X_preprocessed, columns=X.columns)

    df_scaled = pd.concat([X_preprocessed, y], axis=1)

    # Simpan Preprocessed Data
    df_scaled.to_csv('water_potability_preprocessing.csv', index=False)

    return df_scaled