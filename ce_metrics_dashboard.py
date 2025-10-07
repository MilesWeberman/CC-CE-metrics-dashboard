import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

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
                # Create tabs
                tab1, tab2 = st.tabs(["📊 Customer Metrics", "📈 General Statistics"])
                
                with tab1:
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
                
                with tab2:
                    st.subheader("📈 General Statistics")
                    
                    # Time period selection and threshold
                    col_time1, col_time2, col_threshold = st.columns([1, 2, 1])
                    
                    with col_time1:
                        time_period = st.radio(
                            "Select time period:",
                            ["All Time", "Rolling Window"],
                            help="Choose between all-time data or a rolling window analysis"
                        )
                    
                    with col_time2:
                        if time_period == "Rolling Window":
                            weeks = st.slider(
                                "Number of weeks for rolling window:",
                                min_value=1,
                                max_value=52,
                                value=4,
                                help="Select the number of weeks for the rolling window (default: 4 weeks)"
                            )
                    
                    with col_threshold:
                        threshold = st.number_input(
                            "Threshold (days):",
                            min_value=0.0,
                            max_value=75.0,
                            value=10.0,
                            step=0.5,
                            help="Set threshold to see percentage of ingestions under this value"
                        )
                    
                    # Get data for distribution plot
                    delta_data_all = df['Delta data received to prod'].dropna()
                    
                    if len(delta_data_all) == 0:
                        st.warning("⚠️ No valid 'Delta data received to prod' data available for distribution analysis.")
                    else:
                        if time_period == "All Time":
                            # All time distribution
                            st.subheader("📊 All-Time Distribution of Delta Data Received to Prod")
                            
                            # Create histogram with 1-day buckets
                            fig = px.histogram(
                                delta_data_all,
                                nbins=int(delta_data_all.max() - delta_data_all.min() + 1),  # 1-day buckets
                                title="Distribution of Delta Data Received to Prod (All Time) - Click on a bar to see datapoints",
                                labels={'value': 'Delta Data Received to Prod (Days)', 'count': 'Frequency'},
                                color_discrete_sequence=['#1f77b4']
                            )
                            
                            # Add vertical line for threshold
                            fig.add_vline(
                                x=threshold,
                                line_dash="dash",
                                line_color="red",
                                annotation_text=f"Threshold: {threshold} days",
                                annotation_position="top"
                            )
                            
                            fig.update_layout(
                                height=500,
                                showlegend=False,
                                title_x=0.5,
                                xaxis=dict(range=[0, 75])  # Fixed x-axis range
                            )
                            
                            # Display the chart
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Calculate percentage under threshold
                            under_threshold = (delta_data_all <= threshold).sum()
                            percentage_under = (under_threshold / len(delta_data_all)) * 100
                            
                            # Statistics
                            col_stats1, col_stats2, col_stats3, col_stats4, col_stats5 = st.columns(5)
                            
                            with col_stats1:
                                st.metric("Mean", f"{delta_data_all.mean():.2f}")
                            with col_stats2:
                                st.metric("Median", f"{delta_data_all.median():.2f}")
                            with col_stats3:
                                st.metric("Std Dev", f"{delta_data_all.std():.2f}")
                            with col_stats4:
                                st.metric("Count", f"{len(delta_data_all)}")
                            with col_stats5:
                                st.metric(
                                    f"Under {threshold} days",
                                    f"{percentage_under:.1f}%",
                                    help=f"{under_threshold} out of {len(delta_data_all)} ingestions"
                                )
                            
                            # Add bin selection for detailed view
                            st.write("**🔍 Select a bin to see detailed datapoints:**")
                            
                            # Create bins for selection (5-day intervals)
                            min_val = int(delta_data_all.min())
                            max_val = int(delta_data_all.max())
                            bin_options = [f"{i}-{i+5} days" for i in range(min_val, max_val + 1, 5)]
                            
                            selected_bin = st.selectbox(
                                "Choose a bin to explore:",
                                options=bin_options,
                                help="Select a 5-day bin to see all datapoints within that range"
                            )
                            
                            if selected_bin:
                                # Parse the selected bin
                                bin_start = int(selected_bin.split('-')[0])
                                
                                # Filter data for the selected bin (5-day range)
                                bin_data = df[
                                    (df['Delta data received to prod'] >= bin_start) & 
                                    (df['Delta data received to prod'] < bin_start + 5)
                                ][['Customer_Name', 'Task name','Delta data received to prod', 'Created_Date', 'Estimated days', 'FTE Days']]
                                
                                if len(bin_data) > 0:
                                    st.subheader(f"📋 Datapoints in {selected_bin} range")
                                    st.dataframe(bin_data, use_container_width=True, hide_index=True)
                                else:
                                    st.info(f"No datapoints found in the {selected_bin} range.")
                        
                        else:
                            # Rolling window distribution
                            st.subheader(f"📊 Rolling {weeks}-Week Distribution of Delta Data Received to Prod")
                            
                            # Get the full date range of the data
                            min_data_date = df['Created_Date'].min()
                            max_data_date = df['Created_Date'].max()
                            
                            # Calculate how many time periods we can slide through
                            total_days = (max_data_date - min_data_date).days
                            window_days = weeks * 7
                            max_start_day = max(0, total_days - window_days)
                            
                            if max_start_day == 0:
                                st.warning(f"⚠️ Not enough data to create a {weeks}-week rolling window.")
                            else:
                                # Time slider to scroll through the rolling window
                                st.write("**🕒 Time Slider - Scroll through time to see distribution evolution:**")
                                
                                # Create list of possible start dates
                                possible_start_dates = []
                                for day in range(0, max_start_day + 1):
                                    possible_start_dates.append(min_data_date + timedelta(days=day))
                                
                                # Use select_slider to show dates
                                selected_start_date = st.select_slider(
                                    "Select starting date for rolling window:",
                                    options=possible_start_dates,
                                    value=possible_start_dates[-1],  # Default to most recent window
                                    format_func=lambda x: x.strftime("%Y-%m-%d"),
                                    help=f"Slide to see how the {weeks}-week distribution changes over time"
                                )
                                
                                # Convert selected date back to start_day
                                start_day = (selected_start_date - min_data_date).days
                                
                                # Calculate the date range for the selected window
                                window_start_date = min_data_date + timedelta(days=start_day)
                                window_end_date = window_start_date + timedelta(days=window_days)
                                
                                # Filter data for the selected rolling window
                                rolling_data = df[
                                    (df['Created_Date'] >= window_start_date) & 
                                    (df['Created_Date'] <= window_end_date)
                                ]['Delta data received to prod'].dropna()
                                
                                if len(rolling_data) == 0:
                                    st.warning(f"⚠️ No data available for the selected {weeks}-week period.")
                                else:
                                    # Create histogram for rolling window with 1-day buckets
                                    fig = px.histogram(
                                        rolling_data,
                                        nbins=75,  # 1-day buckets for 0-75 days
                                        title=f"Distribution of Delta Data Received to Prod ({weeks}-Week Window) - Select a bin to see datapoints",
                                        labels={'value': 'Delta Data Received to Prod (Days)', 'count': 'Frequency'},
                                        color_discrete_sequence=['#ff7f0e']
                                    )
                                    
                                    # Add vertical line for threshold
                                    fig.add_vline(
                                        x=threshold,
                                        line_dash="dash",
                                        line_color="red",
                                        annotation_text=f"Threshold: {threshold} days",
                                        annotation_position="top"
                                    )
                                    
                                    fig.update_layout(
                                        height=500,
                                        showlegend=False,
                                        title_x=0.5,
                                        xaxis=dict(range=[0, 75])  # Fixed x-axis range
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Calculate percentage under threshold for rolling window
                                    under_threshold_rolling = (rolling_data <= threshold).sum()
                                    percentage_under_rolling = (under_threshold_rolling / len(rolling_data)) * 100
                                    
                                    # Statistics for rolling window
                                    col_stats1, col_stats2, col_stats3, col_stats4, col_stats5 = st.columns(5)
                                    
                                    with col_stats1:
                                        st.metric("Mean", f"{rolling_data.mean():.2f}")
                                    with col_stats2:
                                        st.metric("Median", f"{rolling_data.median():.2f}")
                                    with col_stats3:
                                        st.metric("Std Dev", f"{rolling_data.std():.2f}")
                                    with col_stats4:
                                        st.metric("Count", f"{len(rolling_data)}")
                                    with col_stats5:
                                        st.metric(
                                            f"Under {threshold} days",
                                            f"{percentage_under_rolling:.1f}%",
                                            help=f"{under_threshold_rolling} out of {len(rolling_data)} ingestions"
                                        )
                                    
                                    # Add bin selection for detailed view
                                    st.write("**🔍 Select a bin to see detailed datapoints:**")
                                    
                                    # Create bins for selection based on rolling data (5-day intervals)
                                    if len(rolling_data) > 0:
                                        min_val_rolling = int(rolling_data.min())
                                        max_val_rolling = int(rolling_data.max())
                                        bin_options_rolling = [f"{i}-{i+5} days" for i in range(min_val_rolling, max_val_rolling + 1, 5)]
                                        
                                        selected_bin_rolling = st.selectbox(
                                            "Choose a bin to explore:",
                                            options=bin_options_rolling,
                                            help="Select a 5-day bin to see all datapoints within that range for the selected time window",
                                            key="rolling_bin_selector"
                                        )
                                        
                                        if selected_bin_rolling:
                                            # Parse the selected bin
                                            bin_start_rolling = int(selected_bin_rolling.split('-')[0])
                                            
                                            # Filter data for the selected bin within the rolling window (5-day range)
                                            bin_data_rolling = df[
                                                (df['Created_Date'] >= window_start_date) & 
                                                (df['Created_Date'] <= window_end_date) &
                                                (df['Delta data received to prod'] >= bin_start_rolling) & 
                                                (df['Delta data received to prod'] < bin_start_rolling + 5)
                                            ][['Customer_Name', 'Delta data received to prod', 'Created_Date', 'Estimated days', 'FTE Days']]
                                            
                                            if len(bin_data_rolling) > 0:
                                                st.subheader(f"📋 Datapoints in {selected_bin_rolling} range ({weeks}-week window)")
                                                st.dataframe(bin_data_rolling, use_container_width=True, hide_index=True)
                                            else:
                                                st.info(f"No datapoints found in the {selected_bin_rolling} range for the selected time window.")
                                    
                                    # Date range info
                                    st.info(f"📅 Data range: {window_start_date.strftime('%Y-%m-%d')} to {window_end_date.strftime('%Y-%m-%d')}")
                                
                                # Evolution over time (rolling window comparison)
                                st.subheader("📈 Evolution Over Time")
                                
                                # Create a time series showing how the distribution changes over time
                                # Group by week and calculate statistics
                                df_with_week = df.copy()
                                df_with_week['Week'] = df_with_week['Created_Date'].apply(lambda x: x - timedelta(days=x.weekday()))
                                
                                weekly_stats = df_with_week.groupby('Week')['Delta data received to prod'].agg(['mean', 'count']).reset_index()
                                weekly_stats = weekly_stats[weekly_stats['count'] > 0]  # Only weeks with data
                                
                                if len(weekly_stats) > 0:
                                    # Create line chart showing weekly mean evolution
                                    fig_evolution = px.line(
                                        weekly_stats,
                                        x='Week',
                                        y='mean',
                                        title="Weekly Average Delta Data Received to Prod Over Time",
                                        labels={'mean': 'Average Delta Data', 'Week': 'Week Starting'},
                                        markers=True
                                    )
                                    
                                    fig_evolution.update_layout(
                                        height=400,
                                        title_x=0.5
                                    )
                                    
                                    st.plotly_chart(fig_evolution, use_container_width=True)
                                
                                else:
                                    st.warning("⚠️ No weekly data available for evolution analysis.")
                    
                    # Customer average table
                    st.subheader("📊 Average Delta Data by Customer")
                    
                    # Calculate average delta per customer
                    customer_avg = df.groupby('Customer_Name')['Delta data received to prod'].agg(['mean', 'count']).reset_index()
                    customer_avg = customer_avg.rename(columns={'mean': 'Average Delta (Days)', 'count': 'Number of Ingestions'})
                    customer_avg = customer_avg.sort_values('Average Delta (Days)', ascending=True)
                    customer_avg['Average Delta (Days)'] = customer_avg['Average Delta (Days)'].round(2)
                    
                    # Display the table
                    st.dataframe(
                        customer_avg,
                        use_container_width=True,
                        hide_index=True
                    )
    
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