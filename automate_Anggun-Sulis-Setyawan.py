import numpy as np
import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def preprocess_data(data):
    import logging
    logging.basicConfig(level=logging.INFO)

    # Cek kondisi
