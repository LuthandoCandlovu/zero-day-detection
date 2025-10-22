import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="Zero-Day Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .anomaly-alert {
        background-color: #ffcccc;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff0000;
    }
</style>
""", unsafe_allow_html=True)

class ZeroDayDashboard:
    def __init__(self):
        self.api_base = "http://localhost:8000/api/v1"
        self.sample_data = self._generate_sample_data()
    
    def _generate_sample_data(self):
        """Generate sample network data for demonstration"""
        protocols = ['TCP', 'UDP', 'HTTP', 'HTTPS', 'DNS']
        sources = [f'192.168.1.{i}' for i in range(1, 11)]
        destinations = ['8.8.8.8', '1.1.1.1', '10.0.0.1', '172.16.0.1']
        
        data = []
        for i in range(100):
            # Create some anomalies
            if i % 20 == 0:
                # Anomalous data
                data.append({
                    'timestamp': (datetime.now() - timedelta(hours=i)).isoformat(),
                    'source_ip': np.random.choice(sources),
                    'destination_ip': np.random.choice(destinations),
                    'packet_size': np.random.randint(1500, 10000),  # Large packets
                    'duration': np.random.uniform(10, 30),  # Long duration
                    'protocol': np.random.choice(protocols)
                })
            else:
                # Normal data
                data.append({
                    'timestamp': (datetime.now() - timedelta(hours=i)).isoformat(),
                    'source_ip': np.random.choice(sources),
                    'destination_ip': np.random.choice(destinations),
                    'packet_size': np.random.randint(64, 1500),
                    'duration': np.random.uniform(0.1, 5.0),
                    'protocol': np.random.choice(protocols)
                })
        
        return data
    
    def check_api_status(self):
        """Check if the API is running"""
        try:
            response = requests.get(f"{self.api_base}/status", timeout=5)
            return response.status_code == 200, response.json() if response.status_code == 200 else {}
        except:
            return False, {}
    
    def detect_anomalies(self, data):
        """Send data to API for anomaly detection"""
        try:
            response = requests.post(
                f"{self.api_base}/detect",
                json={"data": data},
                timeout=10
            )
            return response.status_code == 200, response.json() if response.status_code == 200 else {}
        except Exception as e:
            return False, {"error": str(e)}
    
    def get_monitoring_data(self):
        """Get monitoring data from API"""
        try:
            response = requests.get(f"{self.api_base}/monitoring", timeout=5)
            return response.status_code == 200, response.json() if response.status_code == 200 else {}
        except:
            return False, {}

def main():
    dashboard = ZeroDayDashboard()
    
    # Header
    st.markdown('<h1 class="main-header">🛡️ Zero-Day Detection Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # API Status
    api_status, status_data = dashboard.check_api_status()
    status_color = "🟢" if api_status else "🔴"
    st.sidebar.markdown(f"### API Status: {status_color} {'Connected' if api_status else 'Disconnected'}")
    
    if api_status:
        st.sidebar.json(status_data)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔍 Detection", "📈 Analytics", "⚙️ Settings"])
    
    with tab1:
        st.header("Real-time Monitoring")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Samples", len(dashboard.sample_data))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Simulate anomaly detection for dashboard
            success, result = dashboard.detect_anomalies(dashboard.sample_data[:10])
            anomaly_count = result.get('summary', {}).get('anomalies_detected', 0) if success else 0
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Anomalies Detected", anomaly_count)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Detection Rate", f"{(anomaly_count/10)*100:.1f}%" if success else "0%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("System Status", "Healthy" if api_status else "Offline")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Real-time charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Network Traffic Overview")
            
            # Convert sample data to DataFrame for visualization
            df = pd.DataFrame(dashboard.sample_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Packet size distribution
            fig_packets = px.histogram(
                df, 
                x='packet_size',
                title='Packet Size Distribution',
                color_discrete_sequence=['#1f77b4']
            )
            st.plotly_chart(fig_packets, use_container_width=True)
        
        with col2:
            st.subheader("Protocol Distribution")
            
            protocol_counts = df['protocol'].value_counts()
            fig_protocol = px.pie(
                values=protocol_counts.values,
                names=protocol_counts.index,
                title='Network Protocols'
            )
            st.plotly_chart(fig_protocol, use_container_width=True)
        
        # Recent alerts
        st.subheader("Recent Alerts")
        if anomaly_count > 0:
            st.markdown('<div class="anomaly-alert">', unsafe_allow_html=True)
            st.warning(f"🚨 {anomaly_count} potential anomalies detected in recent traffic!")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("✅ No anomalies detected in recent traffic")
    
    with tab2:
        st.header("Anomaly Detection")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Input Data")
            
            # Data input options
            input_method = st.radio(
                "Choose input method:",
                ["Use Sample Data", "Upload CSV", "Manual Input"]
            )
            
            if input_method == "Use Sample Data":
                sample_size = st.slider("Sample size", 10, 100, 20)
                data_to_analyze = dashboard.sample_data[:sample_size]
                st.info(f"Using {sample_size} sample records")
            
            elif input_method == "Upload CSV":
                uploaded_file = st.file_uploader("Upload network data CSV", type="csv")
                if uploaded_file:
                    df_uploaded = pd.read_csv(uploaded_file)
                    data_to_analyze = df_uploaded.to_dict('records')
                    st.success(f"Uploaded {len(data_to_analyze)} records")
                else:
                    data_to_analyze = []
            
            else:  # Manual Input
                st.info("Enter network data manually")
                manual_data = []
                
                with st.form("manual_data_form"):
                    source_ip = st.text_input("Source IP", "192.168.1.100")
                    dest_ip = st.text_input("Destination IP", "8.8.8.8")
                    packet_size = st.number_input("Packet Size", min_value=64, max_value=10000, value=1500)
                    duration = st.number_input("Duration (seconds)", min_value=0.1, max_value=60.0, value=2.5)
                    protocol = st.selectbox("Protocol", ["TCP", "UDP", "HTTP", "HTTPS", "DNS"])
                    
                    if st.form_submit_button("Add to Analysis"):
                        manual_data.append({
                            'timestamp': datetime.now().isoformat(),
                            'source_ip': source_ip,
                            'destination_ip': dest_ip,
                            'packet_size': packet_size,
                            'duration': duration,
                            'protocol': protocol
                        })
                        st.success("Data added to analysis queue")
                
                data_to_analyze = manual_data
            
            if st.button("🔍 Detect Anomalies", type="primary") and data_to_analyze:
                with st.spinner("Analyzing data for anomalies..."):
                    success, results = dashboard.detect_anomalies(data_to_analyze)
                    
                    if success:
                        st.success("Analysis completed!")
                        
                        # Display results
                        anomalies = results.get('anomalies', [])
                        summary = results.get('summary', {})
                        
                        # Summary metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Records", summary.get('total_records', 0))
                        with col2:
                            st.metric("Anomalies Found", summary.get('anomalies_detected', 0))
                        with col3:
                            st.metric("Anomaly Rate", f"{summary.get('anomaly_rate', 0)*100:.1f}%")
                        
                        # Anomalies table
                        if anomalies:
                            st.subheader("Detected Anomalies")
                            anomalies_df = pd.DataFrame(anomalies)
                            st.dataframe(anomalies_df)
                            
                            # Highlight anomalies in original data
                            st.subheader("Analysis Results")
                            original_df = pd.DataFrame(data_to_analyze)
                            original_df['is_anomaly'] = [anom['is_anomaly'] for anom in anomalies]
                            original_df['anomaly_score'] = [anom['score'] for anom in anomalies]
                            
                            # Color code anomalies
                            def highlight_anomalies(row):
                                return ['background-color: #ffcccc' if row['is_anomaly'] else '' for _ in row]
                            
                            st.dataframe(original_df.style.apply(highlight_anomalies, axis=1))
                        
                        else:
                            st.info("✅ No anomalies detected in the provided data")
                    
                    else:
                        st.error("Analysis failed. Please check if the API is running.")
        
        with col2:
            st.subheader("Visual Analysis")
            
            if 'data_to_analyze' in locals() and data_to_analyze:
                df_analysis = pd.DataFrame(data_to_analyze)
                
                # Create scatter plot
                if len(df_analysis) > 0:
                    fig_scatter = px.scatter(
                        df_analysis,
                        x='packet_size',
                        y='duration',
                        color='protocol',
                        title='Packet Size vs Duration',
                        size_max=10
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Show data preview
                st.subheader("Data Preview")
                st.dataframe(df_analysis.head(10))
    
    with tab3:
        st.header("Analytics & Monitoring")
        
        if api_status:
            success, monitoring_data = dashboard.get_monitoring_data()
            
            if success:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Performance Metrics")
                    
                    # Create performance timeline
                    performance_data = monitoring_data.get('performance_metrics', {})
                    if performance_data:
                        metrics_df = pd.DataFrame([performance_data])
                        st.dataframe(metrics_df)
                    
                    # Simulate performance history
                    st.subheader("Anomaly Detection History")
                    dates = [datetime.now() - timedelta(hours=x) for x in range(24)]
                    anomaly_rates = [max(0, min(0.3, np.random.normal(0.1, 0.05))) for _ in range(24)]
                    
                    fig_history = px.line(
                        x=dates,
                        y=anomaly_rates,
                        title='Anomaly Rate Over Time',
                        labels={'x': 'Time', 'y': 'Anomaly Rate'}
                    )
                    st.plotly_chart(fig_history, use_container_width=True)
                
                with col2:
                    st.subheader("System Alerts")
                    
                    alerts = monitoring_data.get('recent_alerts', [])
                    if alerts:
                        for alert in alerts:
                            st.error(f"🚨 {alert.get('type', 'Alert')} - {alert.get('timestamp', 'Unknown time')}")
                    else:
                        st.info("No recent alerts")
            
            else:
                st.warning("Could not fetch monitoring data")
        else:
            st.error("API is not connected. Please start the Flask server.")
    
    with tab4:
        st.header("System Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Configuration")
            
            model_type = st.selectbox(
                "Detection Model",
                ["Isolation Forest", "One-Class SVM", "Local Outlier Factor"]
            )
            
            contamination = st.slider(
                "Expected Anomaly Rate",
                min_value=0.01,
                max_value=0.5,
                value=0.1,
                help="Expected proportion of anomalies in the data"
            )
            
            sensitivity = st.slider(
                "Detection Sensitivity",
                min_value=1,
                max_value=10,
                value=5,
                help="Higher values detect more subtle anomalies"
            )
            
            if st.button("Update Model Configuration", type="primary"):
                st.success("Configuration updated successfully!")
        
        with col2:
            st.subheader("API Configuration")
            
            api_host = st.text_input("API Host", "localhost")
            api_port = st.number_input("API Port", min_value=1000, max_value=9999, value=8000)
            
            st.subheader("Data Sources")
            
            data_sources = st.multiselect(
                "Active Data Sources",
                ["Network Traffic", "System Logs", "Application Logs", "Security Events"],
                default=["Network Traffic"]
            )
            
            update_frequency = st.selectbox(
                "Update Frequency",
                ["Real-time", "Every 5 minutes", "Every 15 minutes", "Every hour"]
            )

if __name__ == "__main__":
    main()
