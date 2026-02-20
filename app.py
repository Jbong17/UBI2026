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
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        padding: 2rem;
        background: linear-gradient(135deg, #f8f7f5 0%, #f0ebe5 100%);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab-list"] button {
        font-weight: 600;
        font-size: 15px;
        color: #5a4a6b;
        background: linear-gradient(135deg, #f0e5f0 0%, #e8d5e8 100%);
        border: 2px solid transparent;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab-list"] button:hover {
        border-color: #9B59B6;
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background: linear-gradient(135deg, #7B4FB8 0%, #6B3FA0 100%);
        color: white;
        border-color: #D4AF37;
    }
    
    .header-title {
        color: #3d2852;
        font-weight: 700;
        text-align: center;
        font-size: 2.2rem;
        line-height: 1.3;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        color: #7a6b8f;
        text-align: center;
        font-size: 1.05rem;
        font-weight: 500;
        letter-spacing: 0.3px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #7B4FB8 0%, #6B3FA0 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(107, 63, 160, 0.25);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(107, 63, 160, 0.35);
    }
    
    .kinampay-positive {
        background: linear-gradient(135deg, #9B59B6 0%, #AF7AC5 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #D4AF37;
        box-shadow: 0 4px 15px rgba(155, 89, 182, 0.25);
    }
    
    .not-kinampay {
        background: linear-gradient(135deg, #95A5A6 0%, #B0BEC5 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #34495E;
        box-shadow: 0 4px 15px rgba(149, 165, 166, 0.25);
    }
    
    .model-info-box {
        background: white;
        padding: 1.75rem;
        border-radius: 12px;
        border-left: 5px solid #D4AF37;
        margin: 1.25rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border-top: 1px solid #f0e5f0;
    }
    
    .element-card {
        background: linear-gradient(135deg, #fefdfb 0%, #f8f4f0 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #6B3FA0;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
        border: 1px solid #e8d5e8;
    }
    
    .element-card:hover {
        box-shadow: 0 4px 12px rgba(107, 63, 160, 0.15);
        transform: translateX(4px);
    }
    
    .element-title {
        color: #3d2852;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 0.75rem;
        font-family: 'Poppins', sans-serif;
        letter-spacing: -0.3px;
    }
    
    .element-description {
        color: #5a5a5a;
        font-size: 0.95rem;
        line-height: 1.7;
        font-weight: 400;
    }
    
    .confidence-high {
        color: #27AE60;
        font-weight: 700;
        font-size: 1.15rem;
    }
    
    .confidence-low {
        color: #E74C3C;
        font-weight: 700;
        font-size: 1.15rem;
    }
    
    .section-title {
        color: #3d2852;
        font-weight: 700;
        font-size: 1.6rem;
        margin-bottom: 1rem;
        margin-top: 2rem;
        font-family: 'Poppins', sans-serif;
        letter-spacing: -0.5px;
    }
    
    .stNumberInput input {
        font-size: 1rem;
        border-radius: 8px;
        border: 2px solid #e8d5e8;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }
    
    .stNumberInput input:focus {
        border-color: #9B59B6;
        box-shadow: 0 0 0 3px rgba(155, 89, 182, 0.1);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #7B4FB8 0%, #6B3FA0 100%);
        color: white;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.05rem;
        border: none;
        box-shadow: 0 4px 15px rgba(107, 63, 160, 0.3);
        transition: all 0.3s ease;
        letter-spacing: 0.3px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(107, 63, 160, 0.4);
    }
    
    .input-section {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        border: 1px solid #f0e5f0;
        margin-bottom: 1.5rem;
    }
    
    .input-label {
        color: #3d2852;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
        font-family: 'Poppins', sans-serif;
    }
    
    .result-header {
        color: #3d2852;
        font-weight: 700;
        font-size: 1.5rem;
        margin-bottom: 1.5rem;
        font-family: 'Poppins', sans-serif;
    }
    
    .footer-text {
        color: #6B3FA0;
        text-align: center;
        padding: 2rem 0;
        font-weight: 600;
        font-size: 1.05rem;
        font-family: 'Poppins', sans-serif;
    }
    
    .sidebar-metric {
        background: linear-gradient(135deg, #f0e5f0 0%, #e8d5e8 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.75rem 0;
        border-left: 4px solid #6B3FA0;
    }
    
    .stDataFrame {
        font-size: 0.95rem;
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
st.markdown("<p class='header-subtitle'>Advanced Forensic Elemental Analysis for Variety Authentication and Origin Verification</p>", unsafe_allow_html=True)
st.markdown("<hr style='margin: 2rem 0; border: none; border-top: 2px solid #e8d5e8;'>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🔬 Model Information")
    st.markdown("<div class='sidebar-metric'><strong>Algorithm</strong><br>Random Forest Classifier</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-metric'><strong>Accuracy (LOOCV)</strong><br>86.5%</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-metric'><strong>Training Samples</strong><br>52 samples</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-metric'><strong>Classification Method</strong><br>Ensemble Learning</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("#### Elemental Features:")
    st.markdown("**K** • **Mn** • **Cu** • **Zn** • **S** • **Cl** • **Sr**")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Information", "🔍 Single Sample Classification", "📁 Batch Classification", "📜 History"])

with tab1:
    st.markdown("<div class='section-title'>Model Overview</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### AI-Assisted Classification Framework
        
        This advanced machine learning model employs **Random Forest algorithms** trained on forensic elemental analysis data 
        to authenticate Purple Yam ('Kinampay') samples and determine their geographic provenance.
        
        #### Key Features:
        - **Forensic Elemental Analysis**: Energy-dispersive X-ray fluorescence (EDXRF) spectroscopy
        - **Geographic Fingerprinting**: Elemental profiles as unique geographic signatures
        - **Variety Authentication**: Distinguishes Bohol Ubi 'Kinampay' from other varieties
        - **Provenance Verification**: Maps elemental patterns to geographic origin
        """)
    
    with col2:
        st.markdown("""
        #### Performance Metrics
        """)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("<div class='metric-card' style='margin: 0.5rem 0;'><strong>Sensitivity</strong><br><span style='font-size: 1.4rem;'>68.8%</span></div>", unsafe_allow_html=True)
            st.markdown("<div class='metric-card' style='margin: 0.5rem 0;'><strong>Training Data</strong><br><span style='font-size: 1.4rem;'>52</span></div>", unsafe_allow_html=True)
        with col_b:
            st.markdown("<div class='metric-card' style='margin: 0.5rem 0;'><strong>Specificity</strong><br><span style='font-size: 1.4rem;'>94.4%</span></div>", unsafe_allow_html=True)
            st.markdown("<div class='metric-card' style='margin: 0.5rem 0;'><strong>Features</strong><br><span style='font-size: 1.4rem;'>7</span></div>", unsafe_allow_html=True)
    
    st.markdown("<hr style='margin: 2rem 0; border: none; border-top: 1px solid #e8d5e8;'>", unsafe_allow_html=True)
    
    st.markdown("<div class='section-title'>Elemental Signature Analysis</div>", unsafe_allow_html=True)
    st.markdown("<p style='color: #5a5a5a; font-size: 1.05rem; margin-bottom: 1.5rem;'>The model analyzes 7 key elemental compounds that form a unique 'fingerprint' for geographic origin:</p>", unsafe_allow_html=True)
    
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
    
    st.markdown("<hr style='margin: 2rem 0; border: none; border-top: 1px solid #e8d5e8;'>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Classification Categories
        
        **🟣 Bohol Ubi 'Kinampay'** (Positive Class)
        - Authentic purple yam from Bohol region
        - Distinctive elemental profile from local soil
        - Premium variety with protected geographic indication
        """)
    
    with col2:
        st.markdown("""
        **⚪ Not Ubi 'Kinampay'** (Negative Class)
        - Other purple yam varieties
        - Different geographic origins
        - Distinct elemental signatures
        """)
    
    st.markdown("<hr style='margin: 2rem 0; border: none; border-top: 1px solid #e8d5e8;'>", unsafe_allow_html=True)
    
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
    st.markdown("<div class='section-title'>Single Sample Classification</div>", unsafe_allow_html=True)
    st.markdown("<p style='color: #5a5a5a; font-size: 1.05rem; margin-bottom: 1.5rem;'>Enter elemental composition values for a sample to predict if it is <strong>Bohol Ubi 'Kinampay'</strong></p>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("<div class='input-section'>", unsafe_allow_html=True)
        st.markdown("<div class='input-label'>📊 Enter Elemental Values</div>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(2)
        
        with col1:
            K = st.number_input("Potassium (K) - ppm", min_value=5000.0, max_value=35000.0, value=15000.0, step=100.0)
            Cu = st.number_input("Copper (Cu) - ppm", min_value=0.0, max_value=20.0, value=7.0, step=0.5)
            S = st.number_input("Sulfur (S) - ppm", min_value=0.0, max_value=2500.0, value=1000.0, step=50.0)
            Sr = st.number_input("Strontium (Sr) - ppm", min_value=0.0, max_value=25.0, value=7.0, step=0.5)
        
        with col2:
            Mn = st.number_input("Manganese (Mn) - ppm", min_value=-1.0, max_value=10.0, value=2.0, step=0.1)
            Zn = st.number_input("Zinc (Zn) - ppm", min_value=0.0, max_value=25.0, value=12.0, step=0.5)
            Cl = st.number_input("Chlorine (Cl) - ppm", min_value=0.0, max_value=3500.0, value=1200.0, step=100.0)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    sample = {'K': K, 'Mn': Mn, 'Cu': Cu, 'Zn': Zn, 'S': S, 'Cl': Cl, 'Sr': Sr}
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        pass
    
    with col2:
        pass
    
    with col3:
        classify_button = st.button("🔬 Classify Sample", use_container_width=True, key="classify_single")
    
    if classify_button:
        try:
            with st.spinner('Analyzing elemental signature...'):
                result = classifier.predict_single(sample)
            
            st.markdown("<hr style='margin: 2rem 0; border: none; border-top: 2px solid #e8d5e8;'>", unsafe_allow_html=True)
            st.markdown("<div class='result-header'>📊 Classification Result</div>", unsafe_allow_html=True)
            
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
                <strong style='font-size: 1.1rem;'>{emoji} Predicted Class</strong><br>
                <span style='font-size: 1.3rem; font-weight: 700; margin-top: 0.5rem;'>{class_name}</span>
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
                st.markdown("""
                <div class='metric-card'>
                <strong>Model Accuracy</strong><br>
                <span style='font-size: 1.3rem; font-weight: 700;'>86.5%</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<hr style='margin: 2rem 0; border: none; border-top: 1px solid #e8d5e8;'>", unsafe_allow_html=True)
            
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
                font=dict(size=12, family="Inter"),
                plot_bgcolor='rgba(240, 229, 240, 0.3)',
                paper_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            if confidence < 0.70:
                st.warning("⚠️ **Low Confidence Score** — Result should be verified with additional testing")
            elif confidence > 0.85:
                st.success("✅ **High Confidence Classification** — Result is reliable")
            
            st.markdown("<hr style='margin: 2rem 0; border: none; border-top: 1px solid #e8d5e8;'>", unsafe_allow_html=True)
            
            st.markdown("### Elemental Profile Summary")
            
            profile_df = pd.DataFrame({
                'Element': ['K', 'Mn', 'Cu', 'Zn', 'S', 'Cl', 'Sr'],
                'Value (ppm)': [f"{x:.2f}" for x in [K, Mn, Cu, Zn, S, Cl, Sr]],
                'Expected Range': [
                    '9,630 - 29,335',
                    '-0.34 - 6.46',
                    '2.18 - 15.67',
                    '5.76 - 18.22',
                    '507 - 1,738',
                    '488 - 2,611',
                    '3.6 - 16.6'
                ]
            })
            
            st.dataframe(profile_df, use_container_width=True, hide_index=True)
            
            st.success("✓ Sample classification completed and saved to history")
            
            st.session_state.predictions_history.append({
                'timestamp': datetime.now(),
                'sample': sample,
                'prediction': class_name,
                'confidence': confidence
            })
            
        except Exception as e:
            st.error(f"Error during classification: {str(e)}")

with tab3:
    st.markdown("<div class='section-title'>Batch Classification</div>", unsafe_allow_html=True)
    st.markdown("<p style='color: #5a5a5a; font-size: 1.05rem; margin-bottom: 1.5rem;'>Upload a CSV file with multiple samples for high-throughput classification</p>", unsafe_allow_html=True)
    
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
                
                st.markdown("<hr style='margin: 2rem 0; border: none; border-top: 2px solid #e8d5e8;'>", unsafe_allow_html=True)
                st.markdown("<div class='result-header'>📊 Classification Results</div>", unsafe_allow_html=True)
                
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
                    st.markdown(f"<div class='metric-card'><strong>Total Samples</strong><br><span style='font-size: 1.4rem;'>{len(results_df)}</span></div>", unsafe_allow_html=True)
                with col2:
                    kinampay_count = sum(1 for x in predictions_list if "Bohol" in x['Classification'])
                    st.markdown(f"<div class='metric-card'><strong>Bohol Ubi 'Kinampay'</strong><br><span style='font-size: 1.4rem;'>{kinampay_count}</span></div>", unsafe_allow_html=True)
                with col3:
                    not_kinampay_count = len(results_df) - kinampay_count
                    st.markdown(f"<div class='metric-card'><strong>Not Ubi 'Kinampay'</strong><br><span style='font-size: 1.4rem;'>{not_kinampay_count}</span></div>", unsafe_allow_html=True)
                
                st.markdown("<hr style='margin: 2rem 0; border: none; border-top: 1px solid #e8d5e8;'>", unsafe_allow_html=True)
                
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
    st.markdown("<div class='section-title'>Prediction History</div>", unsafe_allow_html=True)
    
    if st.session_state.predictions_history:
        st.markdown(f"<div class='metric-card' style='text-align: left; padding: 1.5rem;'><strong>Total Predictions</strong><br><span style='font-size: 1.3rem;'>{len(st.session_state.predictions_history)} samples classified</span></div>", unsafe_allow_html=True)
        
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

st.markdown("<hr style='margin: 2rem 0; border: none; border-top: 2px solid #e8d5e8;'>", unsafe_allow_html=True)
st.markdown("""
<div class='footer-text'>
🍠 Purple Yam 'Kinampay' Forensic Classification System v1.0<br>
<span style='font-size: 0.9rem; color: #8a7aa0;'>AI-assisted Authentication & Geographic Provenance Verification | Model Accuracy: 86.5%</span>
</div>
""", unsafe_allow_html=True)
