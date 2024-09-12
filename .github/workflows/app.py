import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Data
@st.cache
def load_data():
    file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])
    if file is not None:
        data = pd.read_excel(file)
        return data
    return None

data = load_data()

if data is not None:
    st.sidebar.header("Filters")
    
    # Sidebar filters for date range, Fed rate threshold and indices selection
    date_range = st.sidebar.date_input("Select Date Range", [])
    fed_rate_threshold = st.sidebar.slider("Fed Rate Threshold", min_value=float(data['Fed Rate'].min()), max_value=float(data['Fed Rate'].max()), value=(0.0, 5.0))
    indices_selected = st.sidebar.multiselect("Select Indices", data.columns[3:])

    st.title("Financial Analysis Dashboard")

    # Function to find the best-performing indices for the analysis
    def find_best_indices(df, indices, key_metric='mean'):
        performance = df[indices].describe().T
        if key_metric == 'mean':
            best_performing = performance.sort_values(by='mean', ascending=False).index[0]
        elif key_metric == 'corr':
            best_performing = performance['mean'].idxmax()
        return best_performing

    # Home/Overview Page
    if st.sidebar.button("Home/Overview"):
        st.header("Data Overview")
        st.write(data.head())
        st.line_chart(data[['Date', 'Fed Rate', 'S&P 500']])
        
        st.header("Descriptive Statistics")
        st.write(data.describe())
    
    # Page: Rate Cut Impact
    if st.sidebar.button("Rate Cut Impact"):
        st.header("Impact of Fed Rate Cuts on Indices")
        
        # Filter data based on the chosen rate cut threshold
        rate_cut_data = data[data['Fed Rate'] <= fed_rate_threshold[1]]
        
        st.line_chart(rate_cut_data[indices_selected])
        st.write(rate_cut_data.describe())
        
        # Correlation between rate cuts and indices
        correlation = rate_cut_data[indices_selected].corrwith(rate_cut_data['Fed Rate'])
        st.write("Correlation between rate cuts and indices performance:")
        st.write(correlation)
        
        # Highlight the best performing index based on correlation
        best_index = find_best_indices(rate_cut_data, indices_selected, key_metric='corr')
        st.write(f"The best performing index during rate cuts is: {best_index}")
    
    # Page: Recession Analysis
    if st.sidebar.button("Recession Analysis"):
        st.header("Recession vs Rate Cut Analysis")
        
        # Calculate returns for S&P 500
        data['S&P Return'] = data['S&P 500'].pct_change()
        
        # Define a simple recession period (example: S&P negative return)
        recession_data = data[data['S&P Return'] < 0]
        
        st.line_chart(recession_data[indices_selected])
        st.write(recession_data.describe())

        # Test if indices fell during recessionary periods
        correlation_during_recession = recession_data[indices_selected].corrwith(recession_data['Fed Rate'])
        st.write("Correlation between indices and Fed rate during recession:")
        st.write(correlation_during_recession)
        
        # Highlight the best performing index during recessions
        best_index_recession = find_best_indices(recession_data, indices_selected, key_metric='mean')
        st.write(f"The best performing index during recession periods is: {best_index_recession}")
    
    # Page: Custom Analysis
    if st.sidebar.button("Custom Analysis"):
        st.header("Custom Analysis Tool")
        
        # Allow users to select columns to analyze
        columns_selected = st.sidebar.multiselect("Choose Columns for Custom Analysis", data.columns)
        
        if len(columns_selected) > 1:
            st.line_chart(data[columns_selected])
        
        # Show correlation matrix for selected columns
        correlation_matrix = data[columns_selected].corr()
        st.write("Correlation Matrix:")
        st.write(correlation_matrix)

    # Page: Long/Short Strategy Analysis
    if st.sidebar.button("Long/Short Strategy Analysis"):
        st.header("Long/Short Strategy with Indices")
        
        # Define returns for each index
        returns_data = data[indices_selected].pct_change()
        
        # Simple long/short strategy based on returns (buy top 20% performers, short bottom 20%)
        top_percentile = returns_data.quantile(0.8)
        bottom_percentile = returns_data.quantile(0.2)
        
        long_indices = returns_data.columns[(returns_data > top_percentile).mean() > 0.5]
        short_indices = returns_data.columns[(returns_data < bottom_percentile).mean() > 0.5]
        
        st.write(f"Long Indices (Top 20% performers): {long_indices}")
        st.write(f"Short Indices (Bottom 20% performers): {short_indices}")
        
        # Visualize long/short strategies
        st.line_chart(returns_data[long_indices].mean(axis=1), width=0, height=300)
        st.line_chart(returns_data[short_indices].mean(axis=1), width=0, height=300)
        
        # Highlight best performing index based on long/short strategy
        best_long_index = find_best_indices(returns_data[long_indices], long_indices)
        best_short_index = find_best_indices(returns_data[short_indices], short_indices)
        
        st.write(f"Best Long Index: {best_long_index}")
        st.write(f"Best Short Index: {best_short_index}")

else:
    st.write("Please upload your Excel file.")
