import mlflow
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import argparse
import os
import logging

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Konfigurasi MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Water Potability Preprocessing")

# Fungsi preprocess data
def preprocess_data(data, impute_method, save_path, output_path):
    logger.info("Mulai preprocessing...")

    # Cek kondisi data
    if not data.isnull().any().any():
        logging.warning("Dataset tidak mengandung nilai NaN atau missing values.")

        # Langsung simpan data ke output_path
        data.to_csv(output_path, index=False)

        return data, output_path

    # Pipeline imputasi (penanganan missing values --jika ada)
    cleaning = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=impute_method))
    ])

    # Fit pipeline untuk memproses data
    data_clean = cleaning.fit_transform(data)

    # Simpan pipeline
    dump(cleaning, save_path) # save path untuk menyimpan preprocessor

    # Simpan data hasil preprocessing
    cleaned_data = pd.DataFrame(data_clean, columns=data.columns)
    cleaned_data.to_csv(output_path, index=False)

    return cleaned_data, output_path

# Script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="water_potability_raw.csv")
    parser.add_argument("--impute_method", type=str, default="median")
    parser.add_argument("--save_path", type=str, default="preprocessing/preprocessor.joblib")
    parser.add_argument("--output_path", type=str, default="preprocessing/water_potability_preprocessing.csv")
    args = parser.parse_args()

    # Path absolut file .py ini
    py_dir = os.path.dirname(os.path.abspath(__file__))

    # Pastikan path ke absolut lokasi file python
    dataset_path = os.path.join(py_dir, args.dataset)
    save_path = os.path.join(py_dir, args.save_path)
    output_path = os.path.join(py_dir, args.output_path)

    # Buat folder tujuan jika belum tersedia
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Validasi dataset
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {dataset_path}")

    logger.info(f"Dataset ditemukan di: {dataset_path}")

    # Memuat data
    data = pd.read_csv(dataset_path)
    dataset_version = "v1.0"

    # Proses data
    cleaned_data, cleaned_data_path = preprocess_data(
        data,
        args.impute_method,
        args.save_path,
        args.output_path
    )

    def log_to_mlflow():
        # Log parameter
            mlflow.log_param("dataset_version", dataset_version)
            mlflow.log_param("dataset_path", dataset_path)
            mlflow.log_param("impute_method", args.impute_method)

            # Log artifacts
            mlflow.log_artifact(dataset_path, artifact_path="raw_data")
            if os.path.exists(save_path):
                mlflow.log_artifact(save_path, artifact_path="preprocessor")
            mlflow.log_artifact(cleaned_data_path, artifact_path="cleaned_data")

    if mlflow.active_run() is None:
        with mlflow.start_run():
             log_to_mlflow()
    else:
         log_to_mlflow()
    
    logger.info("Preprocessing Selesai!")