import os
import sys
import streamlit as st
import pandas as pd
from datetime import datetime
import time

# Set page config
st.set_page_config(
    page_title="Health Model Dashboard",
    page_icon="🏥",
    layout="wide"
)

def main():
    st.title("Health Model Dashboard")
    
    # Sidebar
    st.sidebar.header("Settings")
    refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 5, 60, 30)
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Health Metrics")
        # Sample data
        data = pd.DataFrame({
            'Metric': ['Heart Rate', 'Blood Pressure', 'Temperature'],
            'Value': [75, '120/80', '98.6°F']
        })
        st.table(data)
        
    with col2:
        st.subheader("Predictions")
        # Sample predictions
        predictions = pd.DataFrame({
            'Condition': ['Risk Level', 'Next Check-up'],
            'Prediction': ['Low', '2 weeks']
        })
        st.table(predictions)
    
    # Footer
    st.markdown("---")
    st.markdown("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    main() 