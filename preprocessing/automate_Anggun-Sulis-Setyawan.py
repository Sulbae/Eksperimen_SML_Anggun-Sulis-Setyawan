import numpy as np
import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import os
import logging

def preprocess_data() -> None:

    logging.basicConfig(level=logging.INFO)

    # Dataset Source
    data_folder = "./dataset_raw"
    file_name = "water_potability_raw.csv"

    data_path = os.path.join(data_folder, file_name)
    
    # Load dataset
    data = pd.read_csv(data_path)
    logging.info(f"Dataset berhasil di-load dari {data_path}")

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
    pipeline_path = os.path.join("preprocessing", "preprocessing_pipeline.joblib")
    dump(preprocessing_pipeline, pipeline_path)

    # Gabungkan lagi data
    X_preprocessed = pd.DataFrame(X_preprocessed, columns=X.columns)

    df_scaled = pd.concat([X_preprocessed, y], axis=1)

    # Simpan Preprocessed Data
    df_scaled.to_csv("preprocessing/water_potability_preprocessing.csv", index=False)

if __name__ == "__main__":
    preprocess_data()