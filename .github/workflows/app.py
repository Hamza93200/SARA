import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home/Overview", "Rate Cut Impact", "Recession Analysis", "Custom Analysis", "Long/Short Strategy"])

# File uploader outside the cached function to avoid the error
file = st.sidebar.file_uploader("Upload your Excel file", type=["xlsx", "xls"])

# Function to load the data
@st.cache_data  # Updated from @st.cache to @st.cache_data to align with recent Streamlit versions
def load_data(file):
    if file is not None:
        data = pd.read_excel(file)
        return data
    return None

data = load_data(file)

if data is not None:
    # Sidebar filters for date range, Fed rate threshold and indices selection
    st.sidebar.header("Filters")
    date_range = st.sidebar.date_input("Select Date Range", [])
    fed_rate_threshold = st.sidebar.slider("Fed Rate Threshold", min_value=float(data['Fed Rate'].min()), max_value=float(data['Fed Rate'].max()), value=(0.0, 5.0))
    indices_selected = st.sidebar.multiselect("Select Indices", data.columns[3:])

    # Function to find the best-performing indices for the analysis
    def find_best_indices(df, indices, key_metric='mean'):
        performance = df[indices].describe().T
        if key_metric == 'mean':
            best_performing = performance.sort_values(by='mean', ascending=False).index[0]
        elif key_metric == 'corr':
            best_performing = performance['mean'].idxmax()
        return best_performing

    # Home/Overview Page
    if page == "Home/Overview":
        st.header("Data Overview")
        st.write(data.head())
        st.line_chart(data[['Date', 'Fed Rate', 'S&P 500']])
        st.header("Descriptive Statistics")
        st.write(data.describe())
    
    # Rate Cut Impact Page
    if page == "Rate Cut Impact":
        st.header("Impact of Fed Rate Cuts on Indices")
        rate_cut_data = data[data['Fed Rate'] <= fed_rate_threshold[1]]
        st.line_chart(rate_cut_data[indices_selected])
        st.write(rate_cut_data.describe())
        correlation = rate_cut_data[indices_selected].corrwith(rate_cut_data['Fed Rate'])
        st.write("Correlation between rate cuts and indices performance:")
        st.write(correlation)
        best_index = find_best_indices(rate_cut_data, indices_selected, key_metric='corr')
        st.write(f"The best performing index during rate cuts is: {best_index}")
    
    # Recession Analysis Page
    if page == "Recession Analysis":
        st.header("Recession vs Rate Cut Analysis")
        data['S&P Return'] = data['S&P 500'].pct_change()
        recession_data = data[data['S&P Return'] < 0]
        st.line_chart(recession_data[indices_selected])
        st.write(recession_data.describe())
        correlation_during_recession = recession_data[indices_selected].corrwith(recession_data['Fed Rate'])
        st.write("Correlation between indices and Fed rate during recession:")
        st.write(correlation_during_recession)
        best_index_recession = find_best_indices(recession_data, indices_selected, key_metric='mean')
        st.write(f"The best performing index during recession periods is: {best_index_recession}")
    
    # Custom Analysis Page
    if page == "Custom Analysis":
        st.header("Custom Analysis Tool")
        columns_selected = st.sidebar.multiselect("Choose Columns for Custom Analysis", data.columns)
        if len(columns_selected) > 1:
            st.line_chart(data[columns_selected])
        correlation_matrix = data[columns_selected].corr()
        st.write("Correlation Matrix:")
        st.write(correlation_matrix)
    
    # Long/Short Strategy Page
    if page == "Long/Short Strategy":
        st.header("Long/Short Strategy with Indices")
        returns_data = data[indices_selected].pct_change()
        top_percentile = returns_data.quantile(0.8)
        bottom_percentile = returns_data.quantile(0.2)
        long_indices = returns_data.columns[(returns_data > top_percentile).mean() > 0.5]
        short_indices = returns_data.columns[(returns_data < bottom_percentile).mean() > 0.5]
        st.write(f"Long Indices (Top 20% performers): {long_indices}")
        st.write(f"Short Indices (Bottom 20% performers): {short_indices}")
        st.line_chart(returns_data[long_indices].mean(axis=1))
        st.line_chart(returns_data[short_indices].mean(axis=1))
        best_long_index = find_best_indices(returns_data[long_indices], long_indices)
        best_short_index = find_best_indices(returns_data[short_indices], short_indices)
        st.write(f"Best Long Index: {best_long_index}")
        st.write(f"Best Short Index: {best_short_index}")

else:
    st.write("Please upload your Excel file to proceed.")
