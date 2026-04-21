# 📊 Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![License](https://img.shields.io/badge/License-MIT-green)

A professional Deep Learning model to predict customer churn using an Artificial Neural Network (ANN), built with Python, TensorFlow/Keras, and Scikit-Learn.

---

## 🎯 Project Overview

This project develops a predictive model to identify customers at risk of account closure (churn). Using a dataset of ~10,000 customer records from American Express, we built a deep learning classifier that achieves **82% accuracy** with a **73.87% ROC-AUC score**.

The model is deployed as an interactive web application using Streamlit, allowing stakeholders to assess individual customer churn risk in real-time.

---

## ✨ Key Achievements

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 82% |
| **ROC-AUC Score** | 73.87% |
| **Recall (Churn Class)** | 60% |
| **Precision** | ~73% |
| **Classification Threshold** | 0.68 |
| **Training Dataset** | 11,060 samples (balanced dataset) |

### Data Handling
- **Original Dataset**: ~10,000 records
- **Class Imbalance**: 5,530 (No Churn) vs 1,418 (Churn) 
- **Solution**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Result**: Perfectly balanced training set of 11,060 samples

### Model Architecture
- **Type**: Artificial Neural Network (ANN)
- **Hidden Layers**: 3 layers with ReLU activation
- **Optimizer**: RMSprop
- **Loss Function**: Binary Crossentropy
- **Epochs**: 137
- **Batch Size**: 64

---

## 🛠️ Tech Stack

- **Deep Learning**: TensorFlow 2.x, Keras
- **Data Processing**: Pandas, NumPy
- **Model Persistence**: Joblib, HDF5
- **Web Framework**: Streamlit
- **Machine Learning**: Scikit-Learn
- **Preprocessing**: StandardScaler, SMOTE

---

## 📁 Project Structure

```
American Express Data Analysis/
├── README.md                              # Project documentation
├── app.py                                 # Streamlit web application
├── churn_model.h5                        # Trained neural network model (HDF5)
├── scaler.joblib                         # Fitted StandardScaler
├── features.joblib                       # Feature list for model input
└── Actual Files/
    ├── American Express User Exit Prediction.csv  # Raw dataset
    └── Customer_Churn_Model.ipynb            # Jupyter notebook with full analysis

```

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install tensorflow keras pandas numpy scikit-learn streamlit joblib
```

### Installation
1. Clone or download the project
2. Navigate to the project directory
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 💡 How to Use

1. **Enter Customer Details**:
   - Credit Score (300-850)
   - Age (18-100 years)
   - Tenure (0-10 years)
   - Account Balance
   - UPI Status
   - Annual Income
   - Gender
   - Geography
   - Number of Products

2. **Click "Predict Churn Probability"** to get the risk assessment

3. **View Results**:
   - Risk Level: High or Low (based on 0.68 threshold)
   - Churn Probability: Percentage likelihood
   - Detailed Analysis: Additional metrics and insights

---

## 📸 Screenshots

### Application Interface
The Streamlit app provides an intuitive interface for entering customer details and getting instant churn predictions.

![Customer Churn Prediction App Interface](Screenshots/screenshot.png)

*Streamlit web application showing the input form with customer details and prediction button*

---


## 📊 Model Performance

### Classification Metrics
- **Accuracy**: 82% - Overall correctness of predictions
- **Precision**: ~73% - Of predicted churners, 73% actually churned
- **Recall**: 60% - Identifies 60% of actual churners
- **ROC-AUC**: 73.87% - Strong discriminative ability across thresholds

### Threshold Optimization
The model was optimized at a **0.68 probability threshold** to maximize recall for the churn class. This ensures better identification of at-risk customers, even if it means slightly higher false positives.

---

## 🔧 Configuration

Key parameters in `app.py`:
```python
CHURN_THRESHOLD = 0.68        # Decision boundary
INPUT_CONSTRAINTS = {          # Input validation ranges
    "credit_score": {"min": 300, "max": 850, ...},
    ...
}
```

Modify these values in the CONFIGURATION section to adjust model behavior.

---

## 📈 Model Development

### Data Preprocessing
1. **Feature Scaling**: StandardScaler normalization
2. **Categorical Encoding**: One-hot encoding for Gender and Geography
3. **Class Balancing**: SMOTE to handle imbalanced dataset
4. **Train-Test Split**: 80-20 split with stratification

### Training Details
- **Optimizer**: RMSprop (adaptive learning rate)
- **Loss**: Binary Crossentropy (suitable for binary classification)
- **Batch Size**: 64 (balance between speed and stability)
- **Epochs**: 137 (early stopping applied)
- **Regularization**: Dropout layers to prevent overfitting

### Validation Strategy
- Cross-validation on training data
- Held-out test set evaluation
- ROC-AUC curve analysis for threshold selection

---

## 🎓 Model Insights

### Most Important Features
The model primarily uses:
- **Credit Score**: Financial reliability indicator
- **Age**: Demographic factor
- **Tenure**: Customer loyalty indicator
- **Balance**: Account engagement level
- **Number of Products**: Cross-selling indicator

### Interpretation
Customers with:
- ✗ Lower credit scores
- ✗ Younger age profile
- ✗ Short tenure
- ✗ Low account balance
- ✗ Few products

...are at higher risk of churning.

---

## 🔒 Error Handling

The application includes robust error handling for:
- Missing model files
- Invalid input values
- Model prediction failures
- Data preprocessing errors

All errors are logged and displayed clearly to users.

---

## 📝 Logging

The application includes logging at INFO and ERROR levels. Logs are printed to console and can be configured in `app.py`:

```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
```

---

## 🔄 Model Caching

The Streamlit app uses `@st.cache_resource` to cache the model, scaler, and features. This significantly improves performance after the first load.

To clear cache: `Ctrl+Shift+C` in the Streamlit app or restart the terminal.

---

## 🚀 Future Enhancements

- [ ] API endpoint for batch predictions
- [ ] Model retraining pipeline
- [ ] Feature importance visualization (SHAP values)
- [ ] Customer segmentation analysis
- [ ] A/B testing framework for retention strategies
- [ ] Database integration for prediction history
- [ ] Real-time monitoring dashboard
- [ ] Model versioning and experiment tracking

---

## 📚 References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [SMOTE Algorithm](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
- [Class Imbalance Handling](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html)

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

Tushar Gupta

---

## 💬 Support

For issues, questions, or suggestions, please refer to the code documentation or contact the project author.

---
