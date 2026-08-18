import mlflow
import mlflow.xgboost
import mlflow.pyfunc
import shap
import streamlit as st

MLFLOW_TRACKING_URI = 'sqlite:///mlflow.db'

XGB_MODELS = ['podium', 'win', 'points_scored', 'strategy_shift']
REGISTERED_NAMES = {
    'podium': 'f1-podium',
    'win': 'f1-win',
    'points_scored': 'f1-points-scored',
    'strategy_shift': 'f1-strategy-shift',
    'overtaking': 'f1-overtaking',
}


@st.cache_resource
def load_models():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    models = {}
    for name in XGB_MODELS:
        models[name] = mlflow.xgboost.load_model(f'models:/{REGISTERED_NAMES[name]}/latest')
    models['overtaking'] = mlflow.pyfunc.load_model(f'models:/{REGISTERED_NAMES["overtaking"]}/latest')
    return models


@st.cache_resource
def load_explainers():
    models = load_models()
    return {name: shap.TreeExplainer(models[name]) for name in XGB_MODELS}
