"""Shared, leakage-safe feature processing for training and the app."""
import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor

NUMERIC = ['bedrooms', 'bathrooms', 'parking', 'advertised_area_m2',
           'bathrooms_per_bedroom', 'description_words']
CATEGORICAL = ['suburb', 'property_type']
INPUT_COLUMNS = ['suburb', 'property_type', 'bedrooms', 'bathrooms', 'parking',
                 'advertised_area_m2', 'description_excerpt']

# Text cleaning and feature engineering for housing data.
def clean_text(value):
    text = str(value) if pd.notna(value) else ''
    text = re.sub(r'\$\s*[\d,.]+\s*(?:million|m|k)?', ' ', text, flags=re.I)
    text = re.sub(r'[^.!?\n]*(?:sold for|sale price|price guide|auction result)[^.!?\n]*', ' ', text, flags=re.I)
    text = re.sub(r'\bsold\b[!*\s]*|\bprior to auction\b', ' ', text, flags=re.I)
    return re.sub(r'\s+', ' ', text).strip()

# Feature builder for numeric, categorical, and text features.
class FeatureBuilder(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = X.copy()
        for c in ['bedrooms', 'bathrooms', 'parking', 'advertised_area_m2']:
            out[c] = pd.to_numeric(out[c], errors='coerce')
        out['bathrooms_per_bedroom'] = out.bathrooms / out.bedrooms.replace(0, np.nan)
        out['description_excerpt'] = out.description_excerpt.map(clean_text)
        out['description_words'] = out.description_excerpt.str.split().str.len()
        out[CATEGORICAL] = out[CATEGORICAL].fillna('Unknown').astype(str)
        return out

# Model creation and pipeline assembly for housing price prediction.
def make_model(name, include_text=True):
    numerical = Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
                          ('scale', StandardScaler())])
    transforms = [('num', numerical, NUMERIC),
                  ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL)]
    if include_text:
        transforms.append(('text', TfidfVectorizer(max_features=100, min_df=2, stop_words='english'), 'description_excerpt'))
    prep = ColumnTransformer(transforms, sparse_threshold=0)
    models = {
        'Ridge': Ridge(alpha=10),
        'Decision tree': DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=42),
        'Random forest': RandomForestRegressor(n_estimators=250, max_depth=6, min_samples_leaf=3, max_features=0.8, random_state=42, n_jobs=1),
        'Median baseline': DummyRegressor(strategy='median'),
    }
    reg = TransformedTargetRegressor(regressor=models[name], func=np.log, inverse_func=np.exp)
    return Pipeline([('features', FeatureBuilder()), ('preprocess', prep), ('regressor', reg)])

# Function to validate user inputs before making predictions.
def validate_inputs(frame):
    missing = set(INPUT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(sorted(missing)))
    return frame[INPUT_COLUMNS].copy()

# Prediction function for housing properties.
def predict_properties(bundle, frame):
    valid = validate_inputs(frame)
    values = bundle['model'].predict(valid)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('The model did not return a valid price.')
    return values
