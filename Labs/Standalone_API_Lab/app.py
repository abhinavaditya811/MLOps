import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Enhanced Iris Classification",
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

@st.cache_data
def load_and_prepare_data():
    """Load and enhance the Iris dataset."""
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Add realistic noise
    np.random.seed(42)
    noise_level = 0.15
    feature_stds = np.std(X, axis=0)
    noise = np.random.normal(0, feature_stds * noise_level, X.shape)
    X_noisy = X + noise
    X_noisy = np.maximum(X_noisy, 0.1)
    
    return X_noisy, y, iris.feature_names, iris.target_names

@st.cache_resource
def train_models(X, y):
    """Train multiple models and return them."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=3, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42)
    }
    
    # Train and evaluate models
    trained_models = {}
    model_scores = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        model_scores[name] = cv_scores.mean()
    
    # Find best model
    best_model_name = max(model_scores, key=model_scores.get)
    
    return trained_models, scaler, model_scores, best_model_name

def predict_species(model, scaler, features):
    """Make prediction with the selected model."""
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    species_names = ['Setosa', 'Versicolor', 'Virginica']
    predicted_species = species_names[prediction]
    
    prob_dict = {species_names[i]: probabilities[i] for i in range(3)}
    confidence = max(probabilities)
    
    return predicted_species, prob_dict, confidence

def create_radar_chart(features):
    """Create radar chart of current measurements."""
    feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    
    # Normalize features (0-1 scale)
    ranges = [(4.3, 7.9), (2.0, 4.4), (1.0, 6.9), (0.1, 2.5)]
    normalized_features = []
    
    for i, (value, (min_val, max_val)) in enumerate(zip(features, ranges)):
        normalized = (value - min_val) / (max_val - min_val)
        normalized_features.append(max(0, min(1, normalized)))
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=normalized_features,
        theta=feature_names,
        fill='toself',
        name='Current Measurements',
        line=dict(color='#1f77b4')
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title="Iris Measurements Profile",
        height=400
    )
    
    return fig

def create_probability_chart(probabilities):
    """Create probability bar chart."""
    df = pd.DataFrame(list(probabilities.items()), columns=['Species', 'Probability'])
    
    color_map = {
        'Setosa': '#FF6B6B',
        'Versicolor': '#4ECDC4',
        'Virginica': '#45B7D1'
    }
    
    fig = px.bar(df, x='Species', y='Probability',
                 title='Species Prediction Probabilities',
                 color='Species',
                 color_discrete_map=color_map)
    
    fig.update_layout(
        showlegend=False,
        height=400,
        yaxis=dict(range=[0, 1])
    )
    
    return fig

def create_feature_comparison_chart(features):
    """Create feature comparison chart."""
    feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    ranges = [(4.3, 7.9), (2.0, 4.4), (1.0, 6.9), (0.1, 2.5)]
    defaults = [5.8, 3.1, 3.8, 1.2]
    
    data = []
    for i, (name, value, (min_val, max_val), default) in enumerate(zip(feature_names, features, ranges, defaults)):
        data.append({
            'Feature': name,
            'Current': value,
            'Min': min_val,
            'Max': max_val,
            'Default': default
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

def main():
    # Load data and train models
    X, y, feature_names, target_names = load_and_prepare_data()
    models, scaler, model_scores, best_model_name = train_models(X, y)
    
    # Main header
    st.markdown('<h1 class="main-header">🌸 Enhanced Iris Classification Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🌸 Enhanced Iris Classifier")
        
        # Model selection
        st.markdown("### 🤖 Model Selection")
        selected_model = st.selectbox(
            "Choose Model:",
            options=list(models.keys()),
            index=list(models.keys()).index(best_model_name),
            help="Select which trained model to use for prediction"
        )
        
        # Show if this is the best model
        if selected_model == best_model_name:
            st.success(f"✨ Best performing model (CV Score: {model_scores[selected_model]:.3f})")
        else:
            st.info(f"Current model CV Score: {model_scores[selected_model]:.3f}")
            st.info(f"Best model: {best_model_name} ({model_scores[best_model_name]:.3f})")
        
        st.markdown("### 📏 Iris Measurements")
        
        # Feature inputs
        sepal_length = st.slider('Sepal Length (cm)', 4.3, 7.9, 5.8, 0.1)
        sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.4, 3.1, 0.1)
        petal_length = st.slider('Petal Length (cm)', 1.0, 6.9, 3.8, 0.1)
        petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 1.2, 0.1)
        
        # Predict button
        predict_button = st.button('🔍 Classify Species', type="primary")
    
    # Main content
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("### 📊 Current Measurements")
        features = [sepal_length, sepal_width, petal_length, petal_width]
        
        # Show radar chart
        radar_fig = create_radar_chart(features)
        st.plotly_chart(radar_fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📏 Measurement Ranges")
        
        # Show feature comparison
        comparison_fig = create_feature_comparison_chart(features)
        st.plotly_chart(comparison_fig, use_container_width=True)
    
    with col3:
        st.markdown("### 🎯 Prediction Results")
        
        if predict_button:
            # Make prediction
            species, probabilities, confidence = predict_species(
                models[selected_model], scaler, features
            )
            
            # Display results
            if species == 'Setosa':
                st.success(f"🌸 **Species:** Iris {species}")
            elif species == 'Versicolor':
                st.info(f"🌺 **Species:** Iris {species}")
            else:
                st.warning(f"🌷 **Species:** Iris {species}")
            
            st.metric("Confidence", f"{confidence:.1%}")
            st.text(f"Model Used: {selected_model}")
            
            # Show probability chart
            prob_fig = create_probability_chart(probabilities)
            st.plotly_chart(prob_fig, use_container_width=True)
    
    # Model comparison section
    st.markdown("### 🏆 Model Performance Comparison")
    
    # Create performance comparison chart
    performance_df = pd.DataFrame(list(model_scores.items()), 
                                 columns=['Model', 'CV Score'])
    performance_df = performance_df.sort_values('CV Score', ascending=False)
    
    perf_fig = px.bar(performance_df, x='Model', y='CV Score',
                     title='Model Cross-Validation Scores',
                     color='CV Score',
                     color_continuous_scale='viridis')
    
    perf_fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(perf_fig, use_container_width=True)
    
    # Information sections
    with st.expander("ℹ️ About Enhanced Iris Classification"):
        st.markdown("""
        This system demonstrates advanced machine learning capabilities:
        
        **🔬 Multiple Algorithms:**
        - **Decision Tree**: Simple, interpretable rules
        - **Random Forest**: Ensemble method with multiple trees
        - **Gradient Boosting**: Sequential improvement algorithm
        - **Support Vector Machine**: Optimal boundary classification
        
        **🌸 Iris Species:**
        - **Setosa**: Smaller flowers, distinctive features
        - **Versicolor**: Medium-sized, balanced proportions
        - **Virginica**: Larger flowers, longer petals
        
        **🚀 Enhanced Features:**
        - Automatic model selection and comparison
        - Real-time model switching
        - Probability confidence scores
        - Interactive visualizations
        - Realistic noise added to dataset for better model differentiation
        """)
    
    with st.expander("📊 Dataset Information"):
        st.markdown("""
        **Dataset Details:**
        - **Samples**: 150 iris flowers
        - **Features**: 4 measurements (sepal length/width, petal length/width)
        - **Classes**: 3 species (50 samples each)
        - **Enhancement**: Added 15% realistic noise to make classification more challenging
        
        **Cross-Validation Scores:**
        """)
        
        for model, score in sorted(model_scores.items(), key=lambda x: x[1], reverse=True):
            st.write(f"- **{model}**: {score:.3f}")

if __name__ == "__main__":
    main()