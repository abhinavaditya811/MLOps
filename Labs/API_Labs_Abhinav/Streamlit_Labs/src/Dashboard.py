import json
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
from streamlit.logger import get_logger

# FastAPI backend endpoint
FASTAPI_BACKEND_ENDPOINT = "http://localhost:8000"

# Model location
IRIS_MODEL_LOCATION = Path(__file__).resolve().parents[2] / 'FastAPI_Labs' / 'model' / 'iris_model.pkl'

# Test JSON file location
TEST_JSON_PATH = Path(__file__).resolve().parents[1] / 'data' / 'test.json'

# streamlit logger
LOGGER = get_logger(__name__)

def get_feature_info():
    """
    Get feature information from the test.json file or return defaults.
    Returns:
        dict: Feature names with their descriptions and typical ranges.
    """
    try:
        if TEST_JSON_PATH.exists():
            with open(TEST_JSON_PATH, 'r') as f:
                test_data = json.load(f)
                # If test.json contains feature info, use it
                if 'feature_info' in test_data:
                    return test_data['feature_info']
                # Otherwise extract from input_test data to infer ranges
                elif 'input_test' in test_data:
                    sample_data = test_data['input_test']
                    # Use sample values as defaults and create reasonable ranges
                    feature_info = {}
                    for key, value in sample_data.items():
                        if key == 'sepal_length':
                            feature_info[key] = {'min': 4.3, 'max': 7.9, 'default': value, 'help': 'Sepal length in cm'}
                        elif key == 'sepal_width':
                            feature_info[key] = {'min': 2.0, 'max': 4.4, 'default': value, 'help': 'Sepal width in cm'}
                        elif key == 'petal_length':
                            feature_info[key] = {'min': 1.0, 'max': 6.9, 'default': value, 'help': 'Petal length in cm'}
                        elif key == 'petal_width':
                            feature_info[key] = {'min': 0.1, 'max': 2.5, 'default': value, 'help': 'Petal width in cm'}
                    return feature_info
    except Exception as e:
        LOGGER.warning(f"Could not read test.json: {e}")
    
    # Fallback to default feature info
    return {
        'sepal_length': {'min': 4.3, 'max': 7.9, 'default': 5.8, 'help': 'Sepal length in cm'},
        'sepal_width': {'min': 2.0, 'max': 4.4, 'default': 3.1, 'help': 'Sepal width in cm'},
        'petal_length': {'min': 1.0, 'max': 6.9, 'default': 3.8, 'help': 'Petal length in cm'},
        'petal_width': {'min': 0.1, 'max': 2.5, 'default': 1.2, 'help': 'Petal width in cm'}
    }

def create_iris_visualization(features_dict):
    """
    Create a radar chart visualization of iris features.
    """
    feature_info = get_feature_info()
    
    # Normalize features to 0-1 scale for visualization
    normalized_features = []
    feature_names = []
    
    for key, value in features_dict.items():
        if key in feature_info:
            info = feature_info[key]
            normalized_value = (value - info['min']) / (info['max'] - info['min'])
            normalized_features.append(normalized_value)
            feature_names.append(key.replace('_', ' ').title())
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=normalized_features,
        theta=feature_names,
        fill='toself',
        name='Iris Measurements',
        line=dict(color='rgb(106, 81, 163)')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Iris Flower Measurements Profile",
        height=400
    )
    
    return fig

def create_probability_chart(probabilities):
    """
    Create a bar chart for prediction probabilities.
    """
    df = pd.DataFrame(list(probabilities.items()), columns=['Species', 'Probability'])
    
    # Color mapping for species
    color_map = {
        'Setosa': '#FF6B6B',
        'Versicolor': '#4ECDC4', 
        'Virginica': '#45B7D1'
    }
    
    colors = [color_map.get(species, '#999999') for species in df['Species']]
    
    fig = px.bar(df, x='Species', y='Probability', 
                 title='Species Prediction Probabilities',
                 color='Species',
                 color_discrete_map=color_map)
    
    fig.update_layout(
        showlegend=False, 
        height=400,
        yaxis=dict(range=[0, 1])  # Fix: Use yaxis dict instead of update_yaxis
    )
    
    return fig

def create_feature_comparison_chart(features_dict):
    """
    Create a comparison chart showing how current measurements compare to typical ranges.
    """
    feature_info = get_feature_info()
    
    data = []
    for key, value in features_dict.items():
        if key in feature_info:
            info = feature_info[key]
            data.append({
                'Feature': key.replace('_', ' ').title(),
                'Current': value,
                'Min': info['min'],
                'Max': info['max'],
                'Default': info['default']
            })
    
    df = pd.DataFrame(data)
    
    fig = go.Figure()
    
    # Add ranges as bars
    fig.add_trace(go.Bar(
        name='Range',
        x=df['Feature'],
        y=df['Max'] - df['Min'],
        base=df['Min'],
        marker_color='lightblue',
        opacity=0.3
    ))
    
    # Add current values as markers
    fig.add_trace(go.Scatter(
        name='Current Value',
        x=df['Feature'],
        y=df['Current'],
        mode='markers',
        marker=dict(size=12, color='red')
    ))
    
    fig.update_layout(
        title='Current Measurements vs Typical Ranges',
        yaxis_title='Measurement (cm)',
        height=400
    )
    
    return fig

def run():
    # Set page config
    st.set_page_config(
        page_title="Enhanced Iris Classification Demo",
        page_icon="🌸",
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #663399;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Build the sidebar
    with st.sidebar:
        st.markdown("### 🌸 Enhanced Iris Classifier")
        
        # Check backend status
        try:
            backend_request = requests.get(FASTAPI_BACKEND_ENDPOINT)
            if backend_request.status_code == 200:
                st.success("Backend online ✅")
                
                # Get model info
                model_info_response = requests.get(f"{FASTAPI_BACKEND_ENDPOINT}/model_info")
                if model_info_response.status_code == 200:
                    model_info = model_info_response.json()
                    st.markdown(f"**Model:** {model_info.get('model_name', 'Unknown')}")
                
            else:
                st.warning("Backend connection issues 😭")
        except requests.ConnectionError as ce:
            LOGGER.error(ce)
            st.error("Backend offline 😱")

        st.markdown("### 🔧 Configure Flower Measurements")
        
        # Model Selection
        try:
            # Get available models from backend
            model_info_response = requests.get(f"{FASTAPI_BACKEND_ENDPOINT}/model_info")
            if model_info_response.status_code == 200:
                model_info = model_info_response.json()
                available_models = model_info.get('available_models', ['Decision Tree'])
                best_model = model_info.get('best_model_name', 'Decision Tree')
                
                # Model selector
                selected_model = st.selectbox(
                    "Choose Model:",
                    options=available_models,
                    index=available_models.index(best_model) if best_model in available_models else 0,
                    help="Select which trained model to use for prediction"
                )
                
                # Show if this is the best model
                if selected_model == best_model:
                    st.success(f"✨ Best performing model")
                else:
                    st.info(f"Best model: {best_model}")
                
                st.session_state["selected_model"] = selected_model
            else:
                st.warning("Could not load model info")
                st.session_state["selected_model"] = None
        except:
            st.error("Backend connection error")
            st.session_state["selected_model"] = None
        
        # Get feature information for sliders
        feature_info = get_feature_info()
        
        # Input method selection
        input_method = st.radio("Choose input method:", ["Manual Input", "JSON File Upload"])
        
        iris_features = {}
        
        if input_method == "Manual Input":
            # Create sliders for each iris feature
            for feature, info in feature_info.items():
                iris_features[feature] = st.slider(
                    info['help'],
                    min_value=info['min'],
                    max_value=info['max'],
                    value=info['default'],
                    step=0.1,
                    format="%.1f cm"
                )
            
            st.session_state["IS_JSON_FILE_AVAILABLE"] = False
            st.session_state["iris_features"] = iris_features
            
        else:
            # JSON file upload
            test_input_file = st.file_uploader('Upload iris test data (JSON)', type=['json'])
            
            if test_input_file:
                st.write('📋 Preview file')
                test_input_data = json.load(test_input_file)
                st.json(test_input_data)
                st.session_state["IS_JSON_FILE_AVAILABLE"] = True
                st.session_state["test_input_data"] = test_input_data
            else:
                st.session_state["IS_JSON_FILE_AVAILABLE"] = False
            
        # Predict button
        predict_button = st.button('🔍 Classify Iris Species', type="primary")

    # Main dashboard body
    st.markdown('<h1 class="main-header">🌸 Enhanced Iris Classification Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Create three columns for layout
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("### 📊 Current Measurements")
        if input_method == "Manual Input":
            # Show radar chart
            fig_radar = create_iris_visualization(iris_features)
            st.plotly_chart(fig_radar, use_container_width=True)
    
    with col2:
        st.markdown("### 📏 Measurement Ranges")
        if input_method == "Manual Input":
            # Show feature comparison
            fig_comparison = create_feature_comparison_chart(iris_features)
            st.plotly_chart(fig_comparison, use_container_width=True)
    
    with col3:
        st.markdown("### 🎯 Prediction Results")
        
        # Prediction logic
        if predict_button:
            # Determine input source
            if (input_method == "JSON File Upload" and 
                "IS_JSON_FILE_AVAILABLE" in st.session_state and 
                st.session_state["IS_JSON_FILE_AVAILABLE"]):
                
                client_input = json.dumps(st.session_state["test_input_data"]['input_test'])
                
            elif input_method == "Manual Input":
                # Add selected model to the input
                iris_features_with_model = iris_features.copy()
                if "selected_model" in st.session_state and st.session_state["selected_model"]:
                    iris_features_with_model["model_name"] = st.session_state["selected_model"]
                
                client_input = json.dumps(iris_features_with_model)
                
            else:
                st.error("Please provide input data")
                st.stop()
            
            # Check if model exists
            if IRIS_MODEL_LOCATION.is_file():
                try:
                    with st.spinner('🔮 Classifying iris species...'):
                        # Get detailed prediction
                        predict_response = requests.post(
                            f'{FASTAPI_BACKEND_ENDPOINT}/predict_detailed', 
                            client_input,
                            headers={'Content-Type': 'application/json'}
                        )
                    
                    if predict_response.status_code == 200:
                        prediction_result = json.loads(predict_response.content)
                        
                        # Display main prediction
                        species = prediction_result["species"]
                        confidence = prediction_result["confidence"]
                        model_used = prediction_result["model_used"]
                        
                        st.success(f"**Species:** {species}")
                        st.info(f"**Confidence:** {confidence:.1%}")
                        st.text(f"**Model:** {model_used}")
                        
                        # Show probability chart
                        if "probabilities" in prediction_result:
                            prob_fig = create_probability_chart(prediction_result["probabilities"])
                            st.plotly_chart(prob_fig, use_container_width=True)
                        
                    else:
                        st.error(f"Prediction failed: {predict_response.status_code}")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    LOGGER.error(e)
                    
            else:
                st.error("Model not found. Please run train.py first.")

    # Information sections
    with st.expander("ℹ️ About Enhanced Iris Classification"):
        st.markdown("""
        This enhanced system automatically compares multiple machine learning algorithms and uses the best performer:
        
        **Available Models:**
        - 🌳 **Decision Tree**: Simple, interpretable rules
        - 🌲 **Random Forest**: Ensemble of decision trees
        - 📈 **Gradient Boosting**: Sequential learning algorithm
        - 🎯 **Support Vector Machine**: Optimal boundary classifier
        
        **Iris Species:**
        - 🌺 **Setosa**: Smaller flowers, distinctive petal patterns
        - 🌸 **Versicolor**: Medium-sized, balanced proportions
        - 🌷 **Virginica**: Larger flowers, longer petals
        """)
    
    # Sample JSON format
    with st.expander("📝 Sample JSON Input Format"):
        sample_json = {
            "input_test": {
                "sepal_length": 5.8,
                "sepal_width": 3.1,
                "petal_length": 3.8,
                "petal_width": 1.2
            }
        }
        st.json(sample_json)

if __name__ == "__main__":
    run()