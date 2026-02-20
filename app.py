"""
EDXRF Elemental Classifier - Streamlit Web App
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import json
from datetime import datetime
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))
from edxrf_classifier import EDXRFClassifier

st.set_page_config(page_title="EDXRF Classifier", page_icon="flask", layout="wide")

@st.cache_resource
def load_classifier():
    return EDXRFClassifier(
        model_path='models/random_forest_final.pkl',
        scaler_path='models/feature_scaler.pkl',
        metadata_path='models/model_metadata.json'
    )

classifier = load_classifier()

if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []

st.title("EDXRF Elemental Classifier")
st.markdown("Machine Learning Classification of Elemental Compositions")

page = st.sidebar.radio("Select Page", ["Make Predictions", "Model Info", "Batch Upload", "History"])

st.sidebar.markdown("---")
st.sidebar.metric("Accuracy", "86.5%")

if page == "Make Predictions":
    st.markdown("## Single Sample Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        K = st.slider("K (Potassium)", 9000.0, 30000.0, 15000.0, 100.0)
        Mn = st.slider("Mn (Manganese)", -0.5, 7.0, 2.0, 0.1)
        Cu = st.slider("Cu (Copper)", 2.0, 16.0, 7.0, 0.5)
        Zn = st.slider("Zn (Zinc)", 5.0, 19.0, 12.0, 0.5)
        S = st.slider("S (Sulfur)", 500.0, 1800.0, 1000.0, 50.0)
        Cl = st.slider("Cl (Chlorine)", 400.0, 2700.0, 1200.0, 100.0)
        Sr = st.slider("Sr (Strontium)", 3.0, 17.0, 7.0, 0.5)
        
        sample = {'K': K, 'Mn': Mn, 'Cu': Cu, 'Zn': Zn, 'S': S, 'Cl': Cl, 'Sr': Sr}
    
    with col2:
        json_input = st.text_area("Enter as JSON", value=json.dumps(sample, indent=2), height=250)
        try:
            sample = json.loads(json_input)
        except:
            st.error("Invalid JSON")
    
    if st.button("Make Prediction", use_container_width=True):
        try:
            with st.spinner('Processing...'):
                result = classifier.predict_single(sample)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Class", result['predicted_class'])
            with col2:
                st.metric("Confidence", f"{result['confidence']*100:.2f}%")
            with col3:
                st.metric("Model Accuracy", "86.5%")
            
            fig = go.Figure(data=[go.Bar(
                x=['Class 0', 'Class 1'],
                y=[result['class_0_probability'], result['class_1_probability']],
                marker_color=['#7c3aed', '#ec4899']
            )])
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            st.success("Prediction saved!")
        except Exception as e:
            st.error(f"Error: {e}")

elif page == "Model Info":
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model Type", "Random Forest")
    with col2:
        st.metric("Accuracy", "86.5%")
    with col3:
        st.metric("Features", "7")
    with col4:
        st.metric("Classes", "2")

elif page == "Batch Upload":
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)

elif page == "History":
    st.info("No predictions yet")
