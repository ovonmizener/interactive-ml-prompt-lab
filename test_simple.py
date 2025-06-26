#!/usr/bin/env python3
"""
Simple test to verify the environment is working
"""

import streamlit as st
import pandas as pd
import numpy as np

def main():
    st.title("🧪 Simple Test - ML Playground")
    st.write("If you can see this, your environment is working!")
    
    # Test basic functionality
    st.subheader("📊 Test Data")
    test_data = pd.DataFrame({
        'x': np.random.randn(100),
        'y': np.random.randn(100)
    })
    
    st.dataframe(test_data.head())
    st.line_chart(test_data)
    
    st.success("✅ All systems operational!")

if __name__ == "__main__":
    main() 