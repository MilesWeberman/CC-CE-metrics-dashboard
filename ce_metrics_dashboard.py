import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Customer Enigneering Metrics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Customer Metrics Dashboard")

# File upload section
st.header("📁 Upload CSV File")
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type="csv",
    help="Upload a CSV file containing customer data with the required columns"
)

if uploaded_file is not None:
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Display basic info about the dataset
        st.success(f"✅ File uploaded successfully! Dataset contains {len(df)} rows and {len(df.columns)} columns.")
        
        # Check if required columns exist
        required_columns = ['Customers', 'Delta data received to prod', 'Estimated days', 'FTE Days', 'External Deadline', 'Uploaded to prod', 'Created time']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {missing_columns}")
            st.info("Please ensure your CSV file contains all required columns.")
        else:
            # Preprocessing: Extract customer name (remove link part)
            df['Customer_Name'] = df['Customers'].str.split(' ').str[0]
            
            # Convert Created time to datetime and extract date
            df['Created_Date'] = pd.to_datetime(df['Created time']).dt.date
            
            # Convert External Deadline and Uploaded to prod to datetime for comparison
            df['External_Deadline_Date'] = pd.to_datetime(df['External Deadline']).dt.date
            df['Uploaded_to_Prod_Date'] = pd.to_datetime(df['Uploaded to prod']).dt.date
            
            # Get unique customers for dropdown
            unique_customers = sorted(df['Customer_Name'].unique())
            
            if len(unique_customers) == 0:
                st.warning("⚠️ No customers found in the dataset.")
            else:
                # Customer selection dropdown
                st.subheader("👥 Customer Selection")
                # Set default customer to Heneken if available, otherwise use first customer
                default_index = 0
                if "Heneken" in unique_customers:
                    default_index = unique_customers.index("Heneken")
                
                selected_customer = st.selectbox(
                    "Select a customer to view metrics:",
                    options=unique_customers,
                    index=default_index,
                    help="Choose a customer from the dropdown to see their specific metrics"
                )
                
                if selected_customer:
                    # Filter data for selected customer
                    customer_data = df[df['Customer_Name'] == selected_customer]
                    
                    st.subheader(f"📈 Metrics for {selected_customer}")
                    
                    # Create columns for metrics display
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        # Count of completed recurring uploads
                        completed_uploads = len(customer_data)
                        st.metric(
                            label="Completed Recurring Uploads",
                            value=completed_uploads,
                            help="Total number of entries for this customer"
                        )
                    
                    with col2:
                        # Average Delta data received to prod (ignoring missing values)
                        delta_data = customer_data['Delta data received to prod'].dropna()
                        if len(delta_data) > 0:
                            avg_delta = delta_data.mean()
                            st.metric(
                                label="Avg Delta Data Received to Prod",
                                value=f"{avg_delta:.2f}",
                                help="Average of 'Delta data received to prod' (missing values excluded)"
                            )
                        else:
                            st.metric(
                                label="Avg Delta Data Received to Prod",
                                value="N/A",
                                help="No valid data available"
                            )
                    
                    with col3:
                        # Average Estimated days (ignoring missing values)
                        estimated_days = customer_data['Estimated days'].dropna()
                        if len(estimated_days) > 0:
                            avg_estimated = estimated_days.mean()
                            st.metric(
                                label="Avg Estimated Days",
                                value=f"{avg_estimated:.2f}",
                                help="Average of 'Estimated days' (missing values excluded)"
                            )
                        else:
                            st.metric(
                                label="Avg Estimated Days",
                                value="N/A",
                                help="No valid data available"
                            )
                    
                    with col4:
                        # Average FTE Days (ignoring missing values)
                        fte_days = customer_data['FTE Days'].dropna()
                        if len(fte_days) > 0:
                            avg_fte = fte_days.mean()
                            st.metric(
                                label="Avg FTE Days",
                                value=f"{avg_fte:.2f}",
                                help="Average of 'FTE Days' (missing values excluded)"
                            )
                        else:
                            st.metric(
                                label="Avg FTE Days",
                                value="N/A",
                                help="No valid data available"
                            )
                    
                    with col5:
                        # External deadline percentage
                        deadline_data = customer_data.dropna(subset=['External_Deadline_Date', 'Uploaded_to_Prod_Date'])
                        if len(deadline_data) > 0:
                            deadline_met = (deadline_data['Uploaded_to_Prod_Date'] <= deadline_data['External_Deadline_Date']).sum()
                            deadline_percentage = (deadline_met / len(deadline_data)) * 100
                            st.metric(
                                label="External Deadline Met",
                                value=f"{deadline_percentage:.1f}%",
                                help=f"Percentage of deadlines met ({deadline_met}/{len(deadline_data)} data points)"
                            )
                        else:
                            st.metric(
                                label="External Deadline Met",
                                value="N/A",
                                help="No valid deadline data available"
                            )
                    
                    # Interactive Time Series Graphs
                    st.subheader("📈 Interactive Time Series Analysis")
                    
                    # Prepare data for time series
                    time_series_data = customer_data.dropna(subset=['Created_Date', 'Delta data received to prod', 'FTE Days'])
                    time_series_data = time_series_data.sort_values('Created_Date')
                    
                    if len(time_series_data) > 0:
                        # Create interactive time series chart
                        fig = make_subplots(
                            rows=2, cols=1,
                            subplot_titles=('Delta Data Received to Prod Over Time', 'FTE Days Over Time'),
                            vertical_spacing=0.1
                        )
                        
                        # Add Delta data line
                        fig.add_trace(
                            go.Scatter(
                                x=time_series_data['Created_Date'],
                                y=time_series_data['Delta data received to prod'],
                                mode='lines+markers',
                                name='Delta Data Received to Prod',
                                line=dict(color='#1f77b4', width=2),
                                marker=dict(size=6),
                                hovertemplate='<b>Date:</b> %{x}<br><b>Delta Data:</b> %{y}<extra></extra>'
                            ),
                            row=1, col=1
                        )
                        
                        # Add FTE Days line
                        fig.add_trace(
                            go.Scatter(
                                x=time_series_data['Created_Date'],
                                y=time_series_data['FTE Days'],
                                mode='lines+markers',
                                name='FTE Days',
                                line=dict(color='#ff7f0e', width=2),
                                marker=dict(size=6),
                                hovertemplate='<b>Date:</b> %{x}<br><b>FTE Days:</b> %{y}<extra></extra>'
                            ),
                            row=2, col=1
                        )
                        
                        # Update layout
                        fig.update_layout(
                            height=600,
                            showlegend=True,
                            title=f"Time Series Analysis for {selected_customer}",
                            title_x=0.5
                        )
                        
                        # Update x-axis labels
                        fig.update_xaxes(title_text="Date", row=1, col=1)
                        fig.update_xaxes(title_text="Date", row=2, col=1)
                        
                        # Update y-axis labels
                        fig.update_yaxes(title_text="Delta Data Received to Prod", row=1, col=1)
                        fig.update_yaxes(title_text="FTE Days", row=2, col=1)
                        
                        # Display the chart
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Additional insights
                        col_insight1, col_insight2 = st.columns(2)
                        
                        with col_insight1:
                            st.write("**Delta Data Insights:**")
                            st.write(f"- Min: {time_series_data['Delta data received to prod'].min():.2f}")
                            st.write(f"- Max: {time_series_data['Delta data received to prod'].max():.2f}")
                            st.write(f"- Trend: {'📈 Increasing' if time_series_data['Delta data received to prod'].iloc[-1] > time_series_data['Delta data received to prod'].iloc[0] else '📉 Decreasing'}")
                        
                        with col_insight2:
                            st.write("**FTE Days Insights:**")
                            st.write(f"- Min: {time_series_data['FTE Days'].min():.2f}")
                            st.write(f"- Max: {time_series_data['FTE Days'].max():.2f}")
                            st.write(f"- Trend: {'📈 Increasing' if time_series_data['FTE Days'].iloc[-1] > time_series_data['FTE Days'].iloc[0] else '📉 Decreasing'}")
                    
                    else:
                        st.warning("⚠️ No time series data available for this customer (missing Created time, Delta data, or FTE Days values).")
                    
                    # Detailed data table for selected customer
                    st.subheader("📋 Detailed Data")
                    st.write(f"Showing all {len(customer_data)} records for {selected_customer}:")
                    st.dataframe(customer_data)
                    
                    # Summary statistics
                    st.subheader("📊 Summary Statistics")
                    summary_col1, summary_col2 = st.columns(2)
                    
                    with summary_col1:
                        st.write("**Data Quality Summary:**")
                        st.write(f"- Total records: {len(customer_data)}")
                        st.write(f"- Delta data records (non-null): {len(customer_data['Delta data received to prod'].dropna())}")
                        st.write(f"- Estimated days records (non-null): {len(customer_data['Estimated days'].dropna())}")
                        st.write(f"- FTE Days records (non-null): {len(customer_data['FTE Days'].dropna())}")
                        st.write(f"- Deadline data records (non-null): {len(customer_data.dropna(subset=['External_Deadline_Date', 'Uploaded_to_Prod_Date']))}")
                    
                    with summary_col2:
                        st.write("**Missing Values:**")
                        st.write(f"- Delta data missing: {customer_data['Delta data received to prod'].isnull().sum()}")
                        st.write(f"- Estimated days missing: {customer_data['Estimated days'].isnull().sum()}")
                        st.write(f"- FTE Days missing: {customer_data['FTE Days'].isnull().sum()}")
                        st.write(f"- External deadline missing: {customer_data['External Deadline'].isnull().sum()}")
                        st.write(f"- Uploaded to prod missing: {customer_data['Uploaded to prod'].isnull().sum()}")
    
    except Exception as e:
        st.error(f"❌ Error reading the CSV file: {str(e)}")
        st.info("Please ensure the file is a valid CSV format.")

else:
    st.info("👆 Please upload a CSV file to get started.")
    
    # Show expected format
    st.subheader("📝 Expected CSV Format")
    st.write("Your CSV file should contain the following columns:")
    st.write("- **Customer**: Customer name followed by a space and link")
    st.write("- **Delta data received to prod**: Numeric values")
    st.write("- **Estimated days**: Numeric values") 
    st.write("- **FTE Days**: Numeric values")
    st.write("- **External Deadline**: Date values (for deadline tracking)")
    st.write("- **Uploaded to prod**: Date values (for deadline tracking)")
    st.write("- **Created time**: Date/time values (for time series analysis)")
    
    # Example data
    example_data = {
        'Customer': ['CustomerA https://example.com', 'CustomerB https://example2.com', 'CustomerA https://example.com'],
        'Delta data received to prod': [10.5, 15.2, 8.7],
        'Estimated days': [5, 7, 3],
        'FTE Days': [2.5, 3.0, 1.8],
        'External Deadline': ['2024-01-15', '2024-01-20', '2024-01-25'],
        'Uploaded to prod': ['2024-01-14', '2024-01-22', '2024-01-24'],
        'Created time': ['2024-01-10 09:00:00', '2024-01-12 14:30:00', '2024-01-15 11:15:00']
    }
    example_df = pd.DataFrame(example_data)
    st.write("**Example format:**")
    st.dataframe(example_df)
