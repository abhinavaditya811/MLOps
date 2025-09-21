import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json

# Flask backend endpoint
FLASK_BACKEND_ENDPOINT = "http://127.0.0.1:8080"

def create_radar_chart(measurements):
    """
    Create a radar chart of iris measurements.
    """
    features = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    values = [measurements['sepal_length'], measurements['sepal_width'], 
              measurements['petal_length'], measurements['petal_width']]
    
    # Normalize values for visualization (0-1 scale)
    ranges = {'sepal_length': (4.3, 7.9), 'sepal_width': (2.0, 4.4), 
              'petal_length': (1.0, 6.9), 'petal_width': (0.1, 2.5)}
    
    normalized_values = []
    for feature, value in zip(['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], values):
        min_val, max_val = ranges[feature]
        normalized = (value - min_val) / (max_val - min_val)
        normalized_values.append(max(0, min(1, normalized)))
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=normalized_values,
        theta=features,
        fill='toself',
        name='Current Measurements',
        line=dict(color='#1f77b4')
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Iris Measurements Profile",
        height=400
    )
    
    return fig

def create_probability_chart(probabilities):
    """
    Create a bar chart for prediction probabilities.
    """
    if not probabilities:
        return None
    
    df = pd.DataFrame(list(probabilities.items()), columns=['Species', 'Probability'])
    
    fig = px.bar(df, x='Species', y='Probability', 
                 title='Species Prediction Probabilities',
                 color='Species',
                 color_discrete_map={
                     'setosa': '#ff7f7f',
                     'versicolor': '#7fbf7f', 
                     'virginica': '#7f7fff'
                 })
    
    fig.update_layout(
        showlegend=False, 
        height=400,
        yaxis=dict(range=[0, 1])
    )
    
    return fig

def main():
    st.set_page_config(
        page_title="Enhanced Flask Iris Classification",
        page_icon="🌺",
        layout="wide"
    )

    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">🌺 Enhanced Flask Iris Classification</h1>', 
                unsafe_allow_html=True)

    # Sidebar controls
    with st.sidebar:
        st.markdown("### 🔧 Configuration")
        
        # Check backend status
        try:
            response = requests.get(f"{FLASK_BACKEND_ENDPOINT}/", timeout=5)
            if response.status_code == 200:
                backend_info = response.json()
                st.success("Flask Backend Online ✅")
                st.info(f"Best Model: {backend_info.get('best_model', 'Unknown')}")
                
                # Get available models
                models_response = requests.get(f"{FLASK_BACKEND_ENDPOINT}/models")
                if models_response.status_code == 200:
                    models_info = models_response.json()
                    available_models = models_info.get('available_models', [])
                    
                    # Model selector
                    selected_model = st.selectbox(
                        "Choose Model:",
                        options=available_models,
                        help="Select which model to use for prediction"
                    )
                    
                    if selected_model == backend_info.get('best_model'):
                        st.success("✨ Best performing model")
                else:
                    selected_model = None
                    st.warning("Could not load model options")
                
            else:
                st.error("Backend connection failed 😕")
                selected_model = None
        except requests.exceptions.RequestException:
            st.error("Backend offline 😱")
            selected_model = None

        st.markdown("### 📏 Iris Measurements")
        
        # Input controls
        sepal_length = st.slider('Sepal Length (cm)', 4.3, 7.9, 5.8, 0.1)
        sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.4, 3.1, 0.1)
        petal_length = st.slider('Petal Length (cm)', 1.0, 6.9, 3.8, 0.1)
        petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 1.2, 0.1)
        
        # Predict button
        predict_button = st.button('🔍 Classify Species', type="primary")

    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📊 Current Measurements")
        
        # Create measurement dictionary
        measurements = {
            'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width
        }
        
        # Show radar chart
        radar_fig = create_radar_chart(measurements)
        st.plotly_chart(radar_fig, use_container_width=True)
        
        # Show measurement table
        measurement_df = pd.DataFrame([
            ['Sepal Length', f"{sepal_length} cm"],
            ['Sepal Width', f"{sepal_width} cm"],
            ['Petal Length', f"{petal_length} cm"],
            ['Petal Width', f"{petal_width} cm"]
        ], columns=['Feature', 'Value'])
        
        st.dataframe(measurement_df, hide_index=True)

    with col2:
        st.markdown("### 🎯 Prediction Results")
        
        if predict_button:
            if selected_model:
                try:
                    # Prepare request data
                    data = {
                        'sepal_length': sepal_length,
                        'sepal_width': sepal_width,
                        'petal_length': petal_length,
                        'petal_width': petal_width,
                        'model_name': selected_model
                    }
                    
                    with st.spinner('🔮 Classifying species...'):
                        # Get detailed prediction
                        response = requests.post(
                            f'{FLASK_BACKEND_ENDPOINT}/predict_detailed',
                            json=data
                        )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display prediction
                        species = result['prediction']
                        confidence = result['confidence']
                        model_used = result['model_used']
                        
                        # Species-specific styling
                        if species == 'setosa':
                            st.success(f"🌸 **Species:** Iris {species.title()}")
                        elif species == 'versicolor':
                            st.info(f"🌺 **Species:** Iris {species.title()}")
                        else:
                            st.warning(f"🌷 **Species:** Iris {species.title()}")
                        
                        st.metric("Confidence", f"{confidence:.1%}")
                        st.text(f"Model Used: {model_used}")
                        
                        # Show probability chart if available
                        if result.get('probabilities'):
                            prob_fig = create_probability_chart(result['probabilities'])
                            if prob_fig:
                                st.plotly_chart(prob_fig, use_container_width=True)
                        
                    else:
                        st.error(f"Prediction failed: {response.status_code}")
                        
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {str(e)}")
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")
            else:
                st.error("Please check backend connection")

    # Information sections
    with st.expander("ℹ️ About Enhanced Flask Iris Classification"):
        st.markdown("""
        This Flask API system demonstrates advanced machine learning capabilities:
        
        **🔬 Multiple Algorithms:**
        - Decision Tree: Simple, interpretable rules
        - Random Forest: Ensemble method with multiple trees
        - Gradient Boosting: Sequential improvement algorithm
        - Support Vector Machine: Optimal boundary classification
        
        **🌸 Iris Species:**
        - **Setosa**: Smaller flowers, distinctive features
        - **Versicolor**: Medium-sized, balanced proportions  
        - **Virginica**: Larger flowers, longer petals
        
        **🚀 Enhanced Features:**
        - Automatic model selection and comparison
        - Real-time model switching
        - Probability confidence scores
        - Interactive visualizations
        """)

    with st.expander("🔧 API Endpoints"):
        st.markdown("""
        **Available Flask Endpoints:**
        - `GET /` - Health check and system info
        - `POST /predict` - Basic species prediction
        - `POST /predict_detailed` - Prediction with probabilities
        - `GET /models` - List available models
        - `GET /model_info` - Detailed model information
        
        **Example JSON Request:**
        ```json
        {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
            "model_name": "Random Forest"
        }
        ```
        """)

if __name__ == '__main__':
    main()