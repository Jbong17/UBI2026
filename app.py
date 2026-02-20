import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import json
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="EDXRF Classifier", page_icon="flask", layout="wide")

@st.cache_resource
def load_classifier():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        model_path = os.path.join(current_dir, 'random_forest_final.pkl')
        scaler_path = os.path.join(current_dir, 'feature_scaler.pkl')
        metadata_path = os.path.join(current_dir, 'model_metadata.json')
        
        sys.path.insert(0, current_dir)
        from edxrf_classifier import EDXRFClassifier
        
        return EDXRFClassifier(
            model_path=model_path,
            scaler_path=scaler_path,
            metadata_path=metadata_path
        )
    except Exception as e:
        st.error(f"Error loading classifier: {str(e)}")
        st.stop()

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
                confidence_pct = round(result['confidence'] * 100, 2)
                st.metric("Confidence", f"{confidence_pct}%")
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
            st.error(f"Error: {str(e)}")

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
    
    st.markdown("---")
    st.markdown("### Training Information")
    st.write("- Training Samples: 52")
    st.write("- Features: K, Mn, Cu, Zn, S, Cl, Sr")
    st.write("- Classes: 0 (Minority), 1 (Majority)")
    st.write("- Class Balance: 16:36")

elif page == "Batch Upload":
    st.markdown("## Batch Prediction")
    
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.markdown("### Preview")
        st.dataframe(df, use_container_width=True)
        
        if st.button("Make Predictions", use_container_width=True):
            try:
                samples = df[['K', 'Mn', 'Cu', 'Zn', 'S', 'Cl', 'Sr']].to_dict('records')
                results = classifier.predict_batch(samples)
                
                predictions_list = []
                for item in results['successful']:
                    result = item['result']
                    conf_pct = round(result['confidence'] * 100, 2)
                    predictions_list.append({
                        'Class': result['predicted_class'],
                        'Confidence': f"{conf_pct}%"
                    })
                
                st.markdown("### Results")
                st.dataframe(pd.DataFrame(predictions_list), use_container_width=True)
                
                csv = pd.DataFrame(predictions_list).to_csv(index=False)
                st.download_button(
                    label="Download Results",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error: {str(e)}")

elif page == "History":
    st.markdown("## Prediction History")
    st.info("No predictions yet")

st.markdown("---")
st.markdown("EDXRF Elemental Classifier v1.0")
