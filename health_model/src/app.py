"""
Streamlit app for interactive health prediction with a simple UI interface.
"""

import streamlit as st
import torch
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import openai
import google.generativeai as genai
import gradio as gr
from dotenv import load_dotenv
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import time
import threading
import queue
import asyncio
import logging
import traceback
from pathlib import Path
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('health_app.log'),
        logging.StreamHandler()
    ]
)

# Load environment variables from .env file
load_dotenv()

# Add the parent directory to the system path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.model import AdvancedHealthModel
from utils.visualization import ModelAnalyzer, explain_predictions

# Set your OpenAI API key here
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set your Gemini API key here
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize session state for dynamic data
if 'user_data' not in st.session_state:
    st.session_state.user_data = pd.DataFrame(columns=[
        'timestamp', 'feature1', 'feature2', 'feature3', 'feature4',
        'feature5', 'feature6', 'feature7', 'feature8', 'feature9',
        'prediction', 'confidence'
    ])

# Initialize real-time data with proper columns
if 'realtime_data' not in st.session_state:
    st.session_state.realtime_data = pd.DataFrame(columns=[
        'timestamp', 'value', 'metric', 'unit'
    ])

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize real-time data queue
data_queue = queue.Queue()

def generate_realtime_data():
    """Generate simulated real-time health data."""
    try:
        current_time = datetime.now()
        metrics = {
            'Heart Rate': {'mean': 75, 'std': 5, 'unit': 'bpm'},
            'Blood Pressure': {'mean': 120, 'std': 10, 'unit': 'mmHg'},
            'Temperature': {'mean': 37, 'std': 0.5, 'unit': '°C'},
            'Oxygen Level': {'mean': 98, 'std': 1, 'unit': '%'},
            'Respiratory Rate': {'mean': 16, 'std': 2, 'unit': 'breaths/min'},
            'Blood Glucose': {'mean': 100, 'std': 10, 'unit': 'mg/dL'}
        }
        
        new_data = []
        for metric, params in metrics.items():
            value = np.random.normal(params['mean'], params['std'])
            new_data.append({
                'timestamp': current_time,
                'value': value,
                'metric': metric,
                'unit': params['unit']
            })
        
        df = pd.DataFrame(new_data)
        logging.info(f"Generated {len(df)} new data points")
        return df
    except Exception as e:
        logging.error(f"Error generating real-time data: {str(e)}")
        return pd.DataFrame(columns=['timestamp', 'value', 'metric', 'unit'])

def real_time_data_generator():
    """Background thread for generating real-time data."""
    while True:
        try:
            new_data = generate_realtime_data()
            if not new_data.empty:
                data_queue.put(new_data)
            time.sleep(1)  # Generate data every second
        except Exception as e:
            logging.error(f"Error in data generation: {str(e)}")
            time.sleep(1)  # Wait before retrying

def create_realtime_chart(data):
    """Create an enhanced real-time line chart using Plotly."""
    try:
        if data.empty:
            logging.warning("Empty data provided for chart creation")
            return go.Figure()
        
        fig = go.Figure()
        
        # Define units for each metric
        units = {
            'Heart Rate': 'bpm',
            'Blood Pressure': 'mmHg',
            'Temperature': '°C',
            'Oxygen Level': '%',
            'Respiratory Rate': 'breaths/min',
            'Blood Glucose': 'mg/dL'
        }
        
        for metric in data['metric'].unique():
            metric_data = data[data['metric'] == metric]
            unit = units.get(metric, '')
            
            fig.add_trace(go.Scatter(
                x=metric_data['timestamp'],
                y=metric_data['value'],
                name=f"{metric} ({unit})",
                mode='lines+markers',
                hovertemplate="<b>%{x}</b><br>Value: %{y:.1f} %{text}<extra></extra>",
                text=[unit] * len(metric_data)
            ))
        
        fig.update_layout(
            title='Real-time Health Metrics',
            xaxis_title='Time',
            yaxis_title='Value',
            hovermode='x unified',
            showlegend=True,
            template='plotly_white',
            height=600
        )
        
        logging.info("Real-time chart created successfully")
        return fig
    except Exception as e:
        logging.error(f"Error creating real-time chart: {str(e)}")
        return go.Figure()

def create_prediction_chart(data):
    """Create an enhanced prediction history chart."""
    fig = px.line(
        data,
        x='timestamp',
        y='prediction',
        title='Prediction History',
        labels={'prediction': 'Prediction Value', 'timestamp': 'Time'},
        template='plotly_white'
    )
    
    fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                  annotation_text="Decision Threshold",
                  annotation_position="bottom right")
    
    fig.update_layout(
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_feature_importance_chart(data):
    """Create an enhanced feature importance chart."""
    features = [f'Feature {i+1}' for i in range(9)]
    importance = np.abs(data.iloc[-1][['feature1', 'feature2', 'feature3',
                                     'feature4', 'feature5', 'feature6',
                                     'feature7', 'feature8', 'feature9']].values)
    
    fig = px.bar(
        x=features,
        y=importance,
        title='Feature Importance',
        labels={'x': 'Features', 'y': 'Importance'},
        template='plotly_white',
        color=importance,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        height=400,
        showlegend=False
    )
    
    return fig

def create_confidence_chart(data):
    """Create an enhanced confidence level chart."""
    fig = px.scatter(
        data,
        x='timestamp',
        y='confidence',
        color='prediction',
        title='Prediction Confidence Over Time',
        labels={'confidence': 'Confidence Level', 'timestamp': 'Time'},
        template='plotly_white',
        color_continuous_scale='RdYlGn'
    )
    
    fig.update_layout(
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_health_dashboard(data):
    """Create a comprehensive health dashboard."""
    metrics = data.groupby('metric')['value'].last()
    
    # Create a grid of metric cards
    cols = st.columns(3)
    for i, (metric, value) in enumerate(metrics.items()):
        with cols[i % 3]:
            st.metric(
                label=metric,
                value=f"{value:.1f}",
                delta=f"{np.random.normal(0, 1):.1f}"
            )

def load_model(model_path, input_size=9):
    """Load the trained model with the correct input size."""
    try:
        model = AdvancedHealthModel(input_size=input_size)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        else:
            st.error(f"Model file not found at {model_path}. Please ensure the model is trained and saved correctly.")
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def predict(model, input_data):
    """Make predictions using the model."""
    try:
        input_data = np.array(input_data)
        input_data = input_data.astype(np.float32)
        input_tensor = torch.FloatTensor(input_data)
        with torch.no_grad():
            output = model(input_tensor)
        return output.numpy()
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def simple_chatbot(user_message):
    """A simple rule-based chatbot for demonstration."""
    user_message = user_message.lower()
    if any(word in user_message for word in ["hello", "hi", "hey"]):
        return "Hello! How can I assist you with your health today?"
    elif "predict" in user_message:
        return "You can upload your data or enter features manually in the Prediction tab."
    elif "thank" in user_message:
        return "You're welcome! If you have more questions, just ask."
    elif "symptom" in user_message:
        return "Please describe your symptoms in detail so I can help you better."
    elif "bye" in user_message:
        return "Goodbye! Stay healthy."
    else:
        return "I'm here to help with your health predictions and questions."

def advanced_chatbot(user_message):
    """Advanced AI chatbot using OpenAI's GPT-3.5-turbo."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful health assistant."},
                {"role": "user", "content": user_message}
            ]
        )
        return response.choices[0].message['content']
    except Exception as e:
        return f"Error: {str(e)}"

def gemini_chatbot(user_message):
    """AI chatbot using Google's Gemini API."""
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(user_message)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def update_user_data(features, prediction, confidence):
    """Update the user's data history."""
    try:
        new_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'feature1': [features[0]],
            'feature2': [features[1]],
            'feature3': [features[2]],
            'feature4': [features[3]],
            'feature5': [features[4]],
            'feature6': [features[5]],
            'feature7': [features[6]],
            'feature8': [features[7]],
            'feature9': [features[8]],
            'prediction': [prediction],
            'confidence': [confidence]
        })
        st.session_state.user_data = pd.concat([st.session_state.user_data, new_data], ignore_index=True)
    except Exception as e:
        st.error(f"Error updating user data: {str(e)}")

def update_realtime_data():
    """Update real-time data from queue."""
    try:
        while not data_queue.empty():
            new_data = data_queue.get_nowait()
            if not new_data.empty:
                # Ensure all required columns are present
                required_columns = ['timestamp', 'value', 'metric', 'unit']
                for col in required_columns:
                    if col not in new_data.columns:
                        if col == 'unit':
                            # Add unit column based on metric
                            units = {
                                'Heart Rate': 'bpm',
                                'Blood Pressure': 'mmHg',
                                'Temperature': '°C',
                                'Oxygen Level': '%',
                                'Respiratory Rate': 'breaths/min',
                                'Blood Glucose': 'mg/dL'
                            }
                            new_data['unit'] = new_data['metric'].map(units)
                        else:
                            new_data[col] = None
                
                # Concatenate with existing data
                st.session_state.realtime_data = pd.concat(
                    [st.session_state.realtime_data, new_data],
                    ignore_index=True
                )
                logging.info(f"Updated real-time data with {len(new_data)} new records")
    except queue.Empty:
        pass
    except Exception as e:
        logging.error(f"Error updating real-time data: {str(e)}")
        st.error(f"Error updating real-time data: {str(e)}")
    
    # Keep only last 100 data points
    if len(st.session_state.realtime_data) > 100:
        st.session_state.realtime_data = st.session_state.realtime_data.tail(100)
        logging.info("Trimmed real-time data to last 100 records")

def validate_dataframe(df, required_columns):
    """Validate DataFrame structure and content."""
    try:
        if df is None or df.empty:
            logging.warning("Empty DataFrame provided")
            return False
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logging.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for NaN values
        nan_columns = df.columns[df.isna().any()].tolist()
        if nan_columns:
            logging.warning(f"NaN values found in columns: {nan_columns}")
        
        # Check data types
        for col in df.columns:
            if df[col].dtype == 'object':
                logging.warning(f"Column {col} has object dtype, may need type conversion")
        
        return True
    except Exception as e:
        logging.error(f"Error validating DataFrame: {str(e)}")
        return False

def main():
    """Main application function."""
    try:
        # Configure Streamlit page
        st.set_page_config(
            page_title="Health Prediction App",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Start real-time data generation in background
        if 'data_thread' not in st.session_state:
            st.session_state.data_thread = threading.Thread(
                target=real_time_data_generator,
                daemon=True
            )
            st.session_state.data_thread.start()
            logging.info("Started real-time data generation thread")
        
        st.title("🩺 Health Prediction & Virtual Doctor Chatbot")
        st.write("Welcome! I am your virtual health assistant. Use the tabs below to predict health outcomes, upload data, or chat with the doctor bot.")

        tabs = st.tabs(["Dashboard", "Prediction", "Upload Data", "Chat with Doctor", 
                       "AI Chat", "Gemini Chat", "Your History", "Real-time Monitoring"])

        # --- Dashboard Tab ---
        with tabs[0]:
            st.header("Health Dashboard")
            
            # Update real-time data
            update_realtime_data()
            
            # Create dashboard
            if not st.session_state.realtime_data.empty:
                create_health_dashboard(st.session_state.realtime_data)
                
                # Create real-time charts
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Real-time Health Metrics")
                    realtime_chart = create_realtime_chart(st.session_state.realtime_data)
                    st.plotly_chart(realtime_chart, use_container_width=True)
                
                with col2:
                    if not st.session_state.user_data.empty:
                        st.subheader("Latest Predictions")
                        prediction_chart = create_prediction_chart(st.session_state.user_data)
                        st.plotly_chart(prediction_chart, use_container_width=True)
            else:
                st.info("Waiting for real-time data...")

        # --- Prediction Tab ---
        with tabs[1]:
            st.header("Manual Prediction")
            st.write("Input your health features below:")
            
            with st.form("prediction_form"):
                col1, col2, col3 = st.columns(3)
                manual_features = []
                
                with col1:
                    for i in range(3):
                        manual_features.append(st.number_input(f"Feature {i+1}", value=0.0, key=f"manual_{i}"))
                with col2:
                    for i in range(3, 6):
                        manual_features.append(st.number_input(f"Feature {i+1}", value=0.0, key=f"manual_{i}"))
                with col3:
                    for i in range(6, 9):
                        manual_features.append(st.number_input(f"Feature {i+1}", value=0.0, key=f"manual_{i}"))
                
                submit_button = st.form_submit_button("Predict")
                
                if submit_button:
                    input_data = np.array([manual_features])
                    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pth')
                    model = load_model(model_path, input_size=len(manual_features))
                    if model is not None:
                        prediction = predict(model, input_data)
                        if prediction is not None:
                            confidence = float(prediction[0][0])
                            update_user_data(manual_features, prediction[0][0], confidence)
                            
                            st.write("Prediction:", prediction)
                            if prediction[0][0] > 0.5:
                                st.success(f"The prediction indicates a positive health outcome with {confidence:.2%} confidence.")
                            else:
                                st.error(f"The prediction indicates a negative health outcome with {(1-confidence):.2%} confidence.")

        # --- Upload Data Tab ---
        with tabs[2]:
            st.header("Batch Prediction via CSV Upload")
            uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type="csv")
            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                    st.write("Data Preview:", data.head())
                    data = data.apply(pd.to_numeric, errors='coerce')
                    if data.isnull().values.any():
                        st.warning("Some values could not be converted to numbers and will be treated as NaN. Rows with NaN will be dropped.")
                        data = data.dropna()
                    if data.empty:
                        st.error("No valid numeric data available for prediction after cleaning. Please check your file.")
                    else:
                        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pth')
                        model = load_model(model_path, input_size=data.shape[1])
                        if model is not None and st.button("Predict Batch", key="predict_batch"):
                            predictions = predict(model, data.values)
                            if predictions is not None:
                                st.write("Predictions:", predictions)
                                
                                # Visualize predictions
                                fig = px.histogram(
                                    x=predictions.flatten(),
                                    title='Distribution of Predictions',
                                    labels={'x': 'Prediction Value', 'y': 'Count'},
                                    template='plotly_white'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

        # --- Chat Tabs (Doctor, AI, Gemini) ---
        for i, tab_name in enumerate(["Chat with Doctor", "AI Chat", "Gemini Chat"]):
            with tabs[i+3]:
                st.header(tab_name)
                chat_history_key = f"{tab_name.lower().replace(' ', '_')}_history"
                if chat_history_key not in st.session_state:
                    st.session_state[chat_history_key] = []
                
                # Chat input
                user_message = st.text_input("You:", key=f"chat_input_{i}")
                if st.button("Send", key=f"send_chat_{i}") and user_message:
                    if tab_name == "Chat with Doctor":
                        response = simple_chatbot(user_message)
                    elif tab_name == "AI Chat":
                        response = advanced_chatbot(user_message)
                    else:
                        response = gemini_chatbot(user_message)
                    
                    st.session_state[chat_history_key].append(("You", user_message))
                    st.session_state[chat_history_key].append(("Assistant", response))
                
                # Display chat history
                chat_container = st.container()
                with chat_container:
                    for sender, message in st.session_state[chat_history_key]:
                        if sender == "You":
                            st.markdown(f"<div style='text-align: right; color: blue;'><b>{sender}:</b> {message}</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div style='text-align: left; color: green;'><b>{sender}:</b> {message}</div>", unsafe_allow_html=True)

        # --- Your History Tab ---
        with tabs[6]:
            st.header("Your Prediction History")
            if not st.session_state.user_data.empty:
                # Interactive prediction history chart
                prediction_chart = create_prediction_chart(st.session_state.user_data)
                st.plotly_chart(prediction_chart, use_container_width=True)
                
                # Feature importance
                feature_chart = create_feature_importance_chart(st.session_state.user_data)
                st.plotly_chart(feature_chart, use_container_width=True)
                
                # Confidence chart
                confidence_chart = create_confidence_chart(st.session_state.user_data)
                st.plotly_chart(confidence_chart, use_container_width=True)
                
                # Data table with sorting and filtering
                st.subheader("Detailed History")
                st.dataframe(
                    st.session_state.user_data.style.background_gradient(
                        subset=['prediction', 'confidence'],
                        cmap='RdYlGn'
                    ),
                    use_container_width=True
                )
            else:
                st.info("No prediction history available yet.")

        # --- Real-time Monitoring Tab ---
        with tabs[7]:
            st.header("Real-time Health Monitoring")
            
            # Get latest real-time data
            try:
                while not data_queue.empty():
                    new_data = data_queue.get_nowait()
                    st.session_state.realtime_data = pd.concat([st.session_state.realtime_data, new_data], ignore_index=True)
            except queue.Empty:
                pass
            
            # Keep only last 100 data points
            if len(st.session_state.realtime_data) > 100:
                st.session_state.realtime_data = st.session_state.realtime_data.tail(100)
            
            # Create and display real-time charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Real-time Health Metrics")
                realtime_chart = create_realtime_chart(st.session_state.realtime_data)
                st.plotly_chart(realtime_chart, use_container_width=True)
                
                # Display current values
                st.subheader("Current Values")
                current_values = st.session_state.realtime_data.groupby('metric')['value'].last()
                for metric, value in current_values.items():
                    st.metric(metric, f"{value:.1f}")
            
            with col2:
                if not st.session_state.user_data.empty:
                    st.subheader("Prediction Confidence")
                    confidence_chart = create_confidence_chart(st.session_state.user_data)
                    st.plotly_chart(confidence_chart, use_container_width=True)
                    
                    st.subheader("Feature Importance")
                    feature_chart = create_feature_importance_chart(st.session_state.user_data)
                    st.plotly_chart(feature_chart, use_container_width=True)

    except Exception as e:
        logging.error(f"Critical error in main function: {str(e)}")
        logging.error(traceback.format_exc())
        st.error("An error occurred while running the application. Please check the logs for details.")
        st.stop()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Fatal error: {str(e)}")
        logging.critical(traceback.format_exc())
        sys.exit(1) 