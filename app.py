"""
Customer Churn Prediction Application

A professional Streamlit app for predicting customer churn risk using a trained
neural network model. Follows best practices for code organization, error handling,
and user experience.
"""

import logging
import os
from typing import Dict, Tuple
from pathlib import Path
# import warnings
# warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

# Page configuration
PAGE_CONFIG = {
    "page_title": "Customer Churn Predictor",
    "page_icon": "📊",
    "layout": "centered",
    "initial_sidebar_state": "collapsed",
}

# Model file paths
MODEL_DIR = Path(__file__).parent
MODEL_FILES = {
    "model": MODEL_DIR / "churn_model.h5",
    "scaler": MODEL_DIR / "scaler.joblib",
    "features": MODEL_DIR / "features.joblib",
}

# Prediction thresholds and risk levels
CHURN_THRESHOLD = 0.68
RISK_LEVELS = {
    "high": (CHURN_THRESHOLD, 1.0),
    "low": (0.0, CHURN_THRESHOLD),
}

# Input constraints
INPUT_CONSTRAINTS = {
    "credit_score": {"min": 300, "max": 850, "default": 600},
    "age": {"min": 18, "max": 100, "default": 40},
    "tenure": {"min": 0, "max": 10, "default": 5},
    "balance": {"min": 0, "default": 50000},
    "income": {"min": 0, "default": 50000},
    "num_products": {"min": 1, "max": 4, "default": 2},
}

GEOGRAPHY_OPTIONS = ["Delhi", "Mumbai", "Bengaluru"]
GENDER_OPTIONS = ["Male", "Female"]
UPI_OPTIONS = [(1, "Yes"), (0, "No")]

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================================
# MODEL LOADING & CACHING
# ============================================================================


@st.cache_resource
def load_model_components() -> Tuple[tf.keras.Model, object, list]:
    """
    Load trained model, scaler, and feature list from disk.
    
    Uses Streamlit's cache to avoid reloading on every run.
    
    Returns:
        Tuple containing: (model, scaler, features)
        
    Raises:
        FileNotFoundError: If any required model file is missing
        ValueError: If model files cannot be loaded
    """
    try:
        # Validate all files exist
        for name, path in MODEL_FILES.items():
            if not path.exists():
                raise FileNotFoundError(f"{name.capitalize()} file not found: {path}")
        
        logger.info("Loading model components...")
        model = load_model(str(MODEL_FILES["model"]))
        scaler = joblib.load(str(MODEL_FILES["scaler"]))
        features = joblib.load(str(MODEL_FILES["features"]))
        
        logger.info("Model components loaded successfully")
        return model, scaler, features
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


# ============================================================================
# DATA PREPROCESSING
# ============================================================================


def create_input_dataframe(
    credit_score: int,
    age: int,
    tenure: int,
    balance: float,
    upi: int,
    income: float,
    gender: str,
    geography: str,
    num_products: int,
    features: list,
) -> pd.DataFrame:
    """
    Create and preprocess input data for model prediction.
    
    Applies one-hot encoding for categorical variables and returns data
    in the correct feature order expected by the model.
    
    Args:
        credit_score: Customer credit score (300-850)
        age: Customer age (18-100)
        tenure: Years as customer
        balance: Current account balance
        upi: Whether UPI is enabled (0 or 1)
        income: Estimated yearly income
        gender: Customer gender ("Male" or "Female")
        geography: Customer location
        num_products: Number of products held
        features: List of expected features in correct order
        
    Returns:
        DataFrame formatted for model input
    """
    input_data = {
        "Credit Score": credit_score,
        "Age": age,
        "Customer Since": tenure,
        "Current Account": balance,
        "Num of products": num_products,
        "UPI Enabled": upi,
        "Estimated Yearly Income": income,
        "Geography_Delhi": 1 if geography == "Delhi" else 0,
        "Geography_Mumbai": 1 if geography == "Mumbai" else 0,
        "Gender_Male": 1 if gender == "Male" else 0,
    }
    
    # Create DataFrame and reorder columns to match training data
    df_input = pd.DataFrame([input_data])[features]
    logger.info(f"Input data created with shape: {df_input.shape}")
    
    return df_input


# ============================================================================
# PREDICTION & RISK ASSESSMENT
# ============================================================================


def predict_churn_probability(
    model: tf.keras.Model,
    scaler: object,
    input_df: pd.DataFrame,
) -> float:
    """
    Generate churn probability prediction.
    
    Args:
        model: Trained neural network model
        scaler: Fitted StandardScaler for feature normalization
        input_df: Preprocessed input features
        
    Returns:
        Churn probability as float between 0 and 1
    """
    try:
        scaled_input = scaler.transform(input_df)
        prediction_proba = model.predict(scaled_input, verbose=0)[0][0]
        logger.info(f"Prediction made: {prediction_proba:.4f}")
        
        return float(prediction_proba)
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise


def assess_risk_level(probability: float) -> Dict[str, str]:
    """
    Assess risk level based on churn probability.
    
    Args:
        probability: Predicted churn probability
        
    Returns:
        Dict with keys: level (high/low), icon, message
    """
    if probability >= CHURN_THRESHOLD:
        return {
            "level": "high",
            "icon": "⚠️",
            "title": "High Risk",
            "message": "This customer is likely to churn. Consider retention strategies.",
            "metric_type": "error",
        }
    else:
        return {
            "level": "low",
            "icon": "✅",
            "title": "Low Risk",
            "message": "This customer is likely to stay. Continue engagement.",
            "metric_type": "success",
        }


# ============================================================================
# UI COMPONENTS
# ============================================================================


def render_header():
    """Render application header and description."""
    st.title("📊 Customer Churn Prediction")
    st.markdown(
        """
        Predict the likelihood of customer account closure using AI.
        Enter customer details below to assess churn risk.
        """
    )


def render_input_form() -> Dict:
    """
    Render customer input form.
    
    Returns:
        Dictionary containing user inputs
    """
    with st.form("prediction_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        
        with col1:
            credit_score = st.number_input(
                "Credit Score",
                min_value=INPUT_CONSTRAINTS["credit_score"]["min"],
                max_value=INPUT_CONSTRAINTS["credit_score"]["max"],
                value=INPUT_CONSTRAINTS["credit_score"]["default"],
                help="Customer's credit score (300-850)",
            )
            age = st.number_input(
                "Age",
                min_value=INPUT_CONSTRAINTS["age"]["min"],
                max_value=INPUT_CONSTRAINTS["age"]["max"],
                value=INPUT_CONSTRAINTS["age"]["default"],
                help="Customer's age in years",
            )
            tenure = st.number_input(
                "Customer Since (Years)",
                min_value=INPUT_CONSTRAINTS["tenure"]["min"],
                max_value=INPUT_CONSTRAINTS["tenure"]["max"],
                value=INPUT_CONSTRAINTS["tenure"]["default"],
                help="How long the customer has been with the company",
            )
            balance = st.number_input(
                "Current Account Balance",
                min_value=INPUT_CONSTRAINTS["balance"]["min"],
                value=INPUT_CONSTRAINTS["balance"]["default"],
                help="Current balance in customer's account",
            )
        
        with col2:
            upi = st.selectbox(
                "UPI Enabled?",
                options=UPI_OPTIONS,
                format_func=lambda x: x[1],
                help="Does the customer use UPI services?",
            )[0]
            income = st.number_input(
                "Estimated Yearly Income",
                min_value=INPUT_CONSTRAINTS["income"]["min"],
                value=INPUT_CONSTRAINTS["income"]["default"],
                help="Customer's estimated annual income",
            )
            gender = st.selectbox(
                "Gender",
                GENDER_OPTIONS,
                help="Customer's gender",
            )
            geography = st.selectbox(
                "Geography",
                GEOGRAPHY_OPTIONS,
                help="Customer's location",
            )
            num_products = st.slider(
                "Number of Products",
                min_value=INPUT_CONSTRAINTS["num_products"]["min"],
                max_value=INPUT_CONSTRAINTS["num_products"]["max"],
                value=INPUT_CONSTRAINTS["num_products"]["default"],
                help="Number of products/services the customer uses",
            )
        
        submit_button = st.form_submit_button(
            "🔮 Predict Churn Probability",
            use_container_width=True,
            type="primary",
        )
    
    return {
        "submitted": submit_button,
        "credit_score": credit_score,
        "age": age,
        "tenure": tenure,
        "balance": balance,
        "upi": upi,
        "income": income,
        "gender": gender,
        "geography": geography,
        "num_products": num_products,
    }


def render_results(probability: float, risk_assessment: Dict):
    """
    Render prediction results with risk assessment.
    
    Args:
        probability: Predicted churn probability
        risk_assessment: Dictionary containing risk level details
    """
    st.divider()
    
    # Main result display
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"{risk_assessment['icon']} {risk_assessment['title']}")
        st.write(risk_assessment["message"])
    
    with col2:
        if risk_assessment["metric_type"] == "error":
            st.metric(
                "Churn Probability",
                f"{probability:.1%}",
                delta=f"{(probability - CHURN_THRESHOLD) * 100:.1f}%",
                delta_color="inverse",
            )
        else:
            st.metric(
                "Churn Probability",
                f"{probability:.1%}",
                delta=f"{(CHURN_THRESHOLD - probability) * 100:.1f}%",
            )
    
    # Additional insights
    st.divider()
    with st.expander("📈 Detailed Analysis"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Risk Level",
                risk_assessment["level"].upper(),
                help="Relative to churn threshold",
            )
        with col2:
            st.metric(
                "Threshold",
                f"{CHURN_THRESHOLD:.0%}",
                help="Classification threshold",
            )
        with col3:
            margin = abs(probability - CHURN_THRESHOLD)
            st.metric(
                "Margin",
                f"{margin:.2%}",
                help="Distance from decision boundary",
            )


# ============================================================================
# MAIN APPLICATION
# ============================================================================


def main():
    """Main application entry point."""
    st.set_page_config(**PAGE_CONFIG)
    
    try:
        # Load model components
        model, scaler, features = load_model_components()
        
        # Render header
        render_header()
        
        # Get user inputs
        form_data = render_input_form()
        
        if form_data["submitted"]:
            with st.spinner("🔄 Analyzing customer profile..."):
                # Preprocess input
                input_df = create_input_dataframe(
                    credit_score=form_data["credit_score"],
                    age=form_data["age"],
                    tenure=form_data["tenure"],
                    balance=form_data["balance"],
                    upi=form_data["upi"],
                    income=form_data["income"],
                    gender=form_data["gender"],
                    geography=form_data["geography"],
                    num_products=form_data["num_products"],
                    features=features,
                )
                
                # Make prediction
                probability = predict_churn_probability(model, scaler, input_df)
                
                # Assess risk
                risk_assessment = assess_risk_level(probability)
                
                # Display results
                render_results(probability, risk_assessment)
    
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {str(e)}")
        logger.error(f"FileNotFoundError: {str(e)}")
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {str(e)}")
        logger.error(f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()