import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import json
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(
    page_title="Ubi Kinampay Classification",
    page_icon="🍠",
    layout="wide",
    initial_sidebar_state="expanded"
)

custom_css = """
<style>
    :root {
        --primary-purple: #6B3FA0;
        --dark-purple: #4A2C6B;
        --light-purple: #D4A5D9;
        --accent-gold: #D4AF37;
        --ubi-dark: #2D1B3D;
        --kinampay-positive: #9B59B6;
        --not-kinampay: #95A5A6;
    }
    
    .main {
        padding: 20px;
        background-color: #f8f7f5;
    }
    
    .stTabs [data-baseweb="tab-list"] button {
        font-weight: 600;
        font-size: 16px;
        color: #4A2C6B;
        background-color: #E8D5E8;
        border-radius: 8px;
        margin: 5px;
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #6B3FA0;
        color: white;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #6B3FA0 0%, #9B59B6 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(107, 63, 160, 0.3);
    }
    
    .kinampay-positive {
        background: linear-gradient(135deg, #9B59B6 0%, #AF7AC5 100%);
        color: white;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #D4AF37;
    }
    
    .not-kinampay {
        background: linear-gradient(135deg, #95A5A6 0%, #B0BEC5 100%);
        color: white;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #7F8C8D;
    }
    
    .header-title {
        color: #4A2C6B;
        font-weight: 700;
        text-align: center;
        margin-bottom: 10px;
    }
    
    .model-info-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #D4AF37;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .element-card {
        background: linear-gradient(135deg, #E8D5E8 0%, #F0E5F0 100%);
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #6B3FA0;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .element-title {
        color: #4A2C6B;
        font-weight: 700;
        font-size: 16px;
        margin-bottom: 8px;
    }
    
    .element-description {
        color: #333;
        font-size: 14px;
        line-height: 1.6;
    }
    
    .confidence-high {
        color: #27AE60;
        font-weight: bold;
    }
    
    .confidence-low {
        color: #E74C3C;
        font-weight: bold;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

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

st.markdown("<h1 class='header-title'>🍠 AI-assisted Predictive Variety Classification and Geographic Provenance of Purple Yam (Bohol Ubi 'Kinampay')</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6B3FA0; font-size: 16px;'>Advanced Forensic Elemental Analysis for Variety Authentication and Origin Verification</p>", unsafe_allow_html=True)

st.sidebar.markdown("## 🔬 Model Information")
st.sidebar.metric("Algorithm", "Random Forest")
st.sidebar.metric("Accuracy (LOOCV)", "86.5%")
st.sidebar.metric("Training Samples", "52")
st.sidebar.markdown("---")
st.sidebar.markdown("**Elemental Features Analyzed:**")
st.sidebar.write("K • Mn • Cu • Zn • S • Cl • Sr")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Information", "🔍 Single Sample Classification", "📁 Batch Classification", "📜 History"])

with tab1:
    st.markdown("## Model Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### AI-Assisted Classification Framework
        
        This advanced machine learning model employs **Random Forest algorithms** trained on forensic elemental analysis data 
        to authenticate Purple Yam ('Kinampay') samples and determine their geographic provenance.
        
        #### Key Features:
        - **Forensic Elemental Analysis**: Uses energy-dispersive X-ray fluorescence (EDXRF) to quantify elemental composition
        - **Geographic Fingerprinting**: Elemental profiles serve as unique geographic signatures
        - **Variety Authentication**: Distinguishes authentic Bohol Ubi 'Kinampay' from other purple yam varieties
        - **Provenance Verification**: Maps elemental patterns to geographic origin and soil composition
        """)
    
    with col2:
        st.markdown("""
        #### Model Performance
        """)
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Sensitivity", "68.8%", help="True positive rate for Ubi 'Kinampay'")
            st.metric("Training Data", "52 samples")
        with col_b:
            st.metric("Specificity", "94.4%", help="True negative rate")
            st.metric("Features", "7 elements")
    
    st.markdown("---")
    
    st.markdown("## Elemental Signature Analysis")
    st.markdown("The model analyzes **7 key elemental compounds** that form a unique 'fingerprint' for geographic origin:")
    
    elements_data = [
        {
            'element': 'K (Potassium)',
            'description': 'Indicator of soil potassium availability and fertilization practices. Reflects nutrient management and soil characteristics.'
        },
        {
            'element': 'Mn (Manganese)',
            'description': 'Reflects soil oxidation state and mineral content. Indicates weathering patterns and geological substrate composition.'
        },
        {
            'element': 'Cu (Copper)',
            'description': 'Bioaccumulation from soil and environmental exposure. Traces pesticide use, mining proximity, and agricultural practices.'
        },
        {
            'element': 'Zn (Zinc)',
            'description': 'Essential micronutrient with geographic variation. Linked to soil pH, organic matter, and regional geochemistry.'
        },
        {
            'element': 'S (Sulfur)',
            'description': 'Related to soil mineralogy and volcanic activity. Indicates atmospheric deposition and industrial influence patterns.'
        },
        {
            'element': 'Cl (Chlorine)',
            'description': 'Indicator of coastal influence and atmospheric deposition. Strong marker for proximity to marine environments.'
        },
        {
            'element': 'Sr (Strontium)',
            'description': 'Geological marker with strong geographic correlation. Strontium isotope ratios are region-specific geological fingerprints.'
        }
    ]
    
    for elem in elements_data:
        st.markdown(f"""
        <div class='element-card'>
            <div class='element-title'>{elem['element']}</div>
            <div class='element-description'>{elem['description']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    ### Classification Categories
    
    **🟣 Bohol Ubi 'Kinampay'** (Positive Class)
    - Authentic purple yam from Bohol region
    - Distinctive elemental profile from local soil
    - Premium variety with protected geographic indication
    
    **⚪ Not Ubi 'Kinampay'** (Negative Class)
    - Other purple yam varieties
    - Different geographic origins
    - Distinct elemental signatures
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### Scientific Methodology
    
    1. **Sample Preparation**: Standardized elemental extraction from yam tissue
    2. **EDXRF Analysis**: Non-destructive X-ray fluorescence spectroscopy
    3. **Feature Normalization**: Standardization of elemental concentrations
    4. **ML Classification**: Random Forest ensemble learning with 500 trees
    5. **Confidence Scoring**: Probabilistic prediction with uncertainty quantification
    
    ### Applications
    - 🏪 **Supply Chain Verification**: Authenticate products at market entry
    - 🔬 **Research & Development**: Geographic origin traceability
    - 📦 **Quality Control**: Batch testing and provenance certification
    - 🌍 **Geographic Indication Protection**: Enforce PDO/PGI standards
    """)

with tab2:
    st.markdown("## Single Sample Classification")
    st.markdown("Enter elemental composition values for a sample to predict if it is **Bohol Ubi 'Kinampay'**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Enter Elemental Values")
        
        K = st.number_input("K (Potassium) - ppm", min_value=5000.0, max_value=35000.0, value=15000.0, step=100.0)
        Mn = st.number_input("Mn (Manganese) - ppm", min_value=-1.0, max_value=10.0, value=2.0, step=0.1)
        Cu = st.number_input("Cu (Copper) - ppm", min_value=0.0, max_value=20.0, value=7.0, step=0.5)
        Zn = st.number_input("Zn (Zinc) - ppm", min_value=0.0, max_value=25.0, value=12.0, step=0.5)
        S = st.number_input("S (Sulfur) - ppm", min_value=0.0, max_value=2500.0, value=1000.0, step=50.0)
        Cl = st.number_input("Cl (Chlorine) - ppm", min_value=0.0, max_value=3500.0, value=1200.0, step=100.0)
        Sr = st.number_input("Sr (Strontium) - ppm", min_value=0.0, max_value=25.0, value=7.0, step=0.5)
        
        sample = {'K': K, 'Mn': Mn, 'Cu': Cu, 'Zn': Zn, 'S': S, 'Cl': Cl, 'Sr': Sr}
    
    with col2:
        st.markdown("### JSON Input (Optional)")
        json_input = st.text_area("Paste JSON here", value=json.dumps(sample, indent=2), height=300)
        try:
            sample = json.loads(json_input)
        except:
            st.error("Invalid JSON format")
    
    st.markdown("---")
    
    if st.button("🔬 Classify Sample", use_container_width=True, key="classify_single"):
        try:
            with st.spinner('Analyzing elemental signature...'):
                result = classifier.predict_single(sample)
            
            st.markdown("---")
            st.markdown("## 📊 Classification Result")
            
            predicted_class = result['predicted_class']
            confidence = result['confidence']
            class_0_prob = result['class_0_probability']
            class_1_prob = result['class_1_probability']
            
            if predicted_class == 1:
                class_name = "Bohol Ubi 'Kinampay'"
                color_class = "kinampay-positive"
                emoji = "🟣"
            else:
                class_name = "Not Ubi 'Kinampay'"
                color_class = "not-kinampay"
                emoji = "⚪"
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class='{color_class}'>
                <strong>{emoji} Predicted Class</strong><br>
                {class_name}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                confidence_class = "confidence-high" if confidence > 0.75 else "confidence-low"
                st.markdown(f"""
                <div class='metric-card'>
                <strong>Confidence Score</strong><br>
                <span class='{confidence_class}'>{confidence*100:.2f}%</span>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.metric("Model Accuracy (LOOCV)", "86.5%")
            
            st.markdown("---")
            
            st.markdown("### Probability Distribution")
            
            fig = go.Figure(data=[go.Bar(
                x=["Not Ubi 'Kinampay'", "Bohol Ubi 'Kinampay'"],
                y=[class_0_prob, class_1_prob],
                marker_color=['#95A5A6', '#9B59B6'],
                text=[f"{class_0_prob*100:.1f}%", f"{class_1_prob*100:.1f}%"],
                textposition='auto',
            )])
            
            fig.update_layout(
                height=400,
                showlegend=False,
                xaxis_title="Classification Class",
                yaxis_title="Probability",
                template="plotly_white",
                font=dict(size=12)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            if confidence < 0.70:
                st.warning("⚠️ Low confidence score - Result should be verified with additional testing")
            elif confidence > 0.85:
                st.success("✅ High confidence classification - Result is reliable")
            
            st.markdown("---")
            
            st.markdown("### Elemental Profile Summary")
            
            profile_df = pd.DataFrame({
                'Element': ['K', 'Mn', 'Cu', 'Zn', 'S', 'Cl', 'Sr'],
                'Value (ppm)': [K, Mn, Cu, Zn, S, Cl, Sr],
                'Expected Range': [
                    '9,630-29,335',
                    '-0.34-6.46',
                    '2.18-15.67',
                    '5.76-18.22',
                    '507-1,738',
                    '488-2,611',
                    '3.6-16.6'
                ]
            })
            
            st.dataframe(profile_df, use_container_width=True, hide_index=True)
            
            st.success("Sample classification completed and saved to history")
            
            st.session_state.predictions_history.append({
                'timestamp': datetime.now(),
                'sample': sample,
                'prediction': class_name,
                'confidence': confidence
            })
            
        except Exception as e:
            st.error(f"Error during classification: {str(e)}")

with tab3:
    st.markdown("## Batch Classification")
    st.markdown("Upload a CSV file with multiple samples for high-throughput classification")
    
    st.markdown("**CSV Format Required:**")
    st.code("K,Mn,Cu,Zn,S,Cl,Sr\n15000,2.0,7.0,12.0,1000,1200,7.0\n20000,1.5,10.0,15.0,1200,1600,10.0", language="csv")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'], key="batch_upload")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        st.markdown("### Preview of Uploaded Data")
        st.dataframe(df, use_container_width=True)
        
        if st.button("🔬 Classify All Samples", use_container_width=True, key="classify_batch"):
            try:
                with st.spinner(f'Classifying {len(df)} samples...'):
                    samples = df[['K', 'Mn', 'Cu', 'Zn', 'S', 'Cl', 'Sr']].to_dict('records')
                    results = classifier.predict_batch(samples)
                
                st.markdown("---")
                st.markdown("## Classification Results")
                
                predictions_list = []
                for item in results['successful']:
                    result = item['result']
                    pred_class = result['predicted_class']
                    conf = result['confidence']
                    
                    if pred_class == 1:
                        class_name = "Bohol Ubi 'Kinampay'"
                    else:
                        class_name = "Not Ubi 'Kinampay'"
                    
                    predictions_list.append({
                        'Classification': class_name,
                        'Confidence': f"{conf*100:.2f}%",
                        'Class 0 Prob': f"{result['class_0_probability']*100:.2f}%",
                        'Class 1 Prob': f"{result['class_1_probability']*100:.2f}%"
                    })
                
                results_df = pd.DataFrame(predictions_list)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Samples", len(results_df))
                with col2:
                    kinampay_count = sum(1 for x in predictions_list if "Bohol" in x['Classification'])
                    st.metric("Bohol Ubi 'Kinampay'", kinampay_count)
                with col3:
                    not_kinampay_count = len(results_df) - kinampay_count
                    st.metric("Not Ubi 'Kinampay'", not_kinampay_count)
                
                st.markdown("---")
                
                st.markdown("### Detailed Results")
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"ubi_kinampay_classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"Error during batch classification: {str(e)}")

with tab4:
    st.markdown("## Prediction History")
    
    if st.session_state.predictions_history:
        st.metric("Total Predictions", len(st.session_state.predictions_history))
        
        history_data = []
        for h in reversed(st.session_state.predictions_history):
            history_data.append({
                'Timestamp': h['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'Prediction': h['prediction'],
                'Confidence': f"{h['confidence']*100:.2f}%"
            })
        
        st.dataframe(pd.DataFrame(history_data), use_container_width=True, hide_index=True)
        
        if st.button("Clear History", use_container_width=True):
            st.session_state.predictions_history = []
            st.rerun()
    else:
        st.info("No predictions yet. Start by classifying samples in the 'Single Sample Classification' or 'Batch Classification' tabs.")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B3FA0; padding: 20px;'>
<strong>🍠 Purple Yam 'Kinampay' Forensic Classification System v1.0</strong><br>
<small>AI-assisted Predictive Variety Classification & Geographic Provenance Verification<br>
Model Accuracy: 86.5% | Forensic Elemental Analysis | EDXRF-Based Classification</small>
</div>
""", unsafe_allow_html=True)
