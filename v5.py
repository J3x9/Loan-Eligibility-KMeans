import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="Loan App", layout="centered")
st.title("💳 Loan Eligibility Clustering")

# New: Load demo data or upload
st.header("📁 Choose Your Data Source")
data_source = st.radio("Select input method:",
                       ["Upload your own CSV", "Use Demo Data"])

if data_source == "Use Demo Data":
    try:
        df = pd.read_csv("Demo_Data.csv")
        st.success("✅ Demo data loaded successfully.")
    except FileNotFoundError:
        st.error("❌ 'Demo_Data.csv' not found in the app directory.")
        df = None
else:
    uploaded_file = st.file_uploader("Upload a CSV file with customer data", type=["csv"])
    df = pd.read_csv(uploaded_file) if uploaded_file is not None else None

# If we have a dataframe (from demo or upload)
if df is not None:
    st.subheader("📋 Loaded Data")
    st.dataframe(df)

    # Feature selector
    st.header("✅ Select Features for Clustering")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    default_features = [f for f in ['CreditScore', 'BankBalance'] if f in numeric_cols]
    selected_features = st.multiselect("Choose features to use in clustering:",
                                       options=numeric_cols,
                                       default=default_features)

    # Cluster count dropdown
    st.header("🔢 Select Number of Clusters")
    cluster_count = st.selectbox("Number of clusters:", options=[2, 3, 4, 5], index=0)

    # Criteria for "Eligible" cluster
    st.header("🎯 Define 'Eligible' Cluster By")
    criteria = st.radio("Pick method to define eligibility:",
                        options=["Highest average credit score", "Highest credit score + balance combined"])

    # Run clustering
    if st.button("🚀 Run Clustering"):
        if len(selected_features) < 2:
            st.warning("Please select at least two features.")
        else:
            features = df[selected_features]
            scaler = StandardScaler()
            scaled = scaler.fit_transform(features)

            kmeans = KMeans(n_clusters=cluster_count, random_state=42)
            df['Cluster'] = kmeans.fit_predict(scaled)
            cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=selected_features)

            # Determine eligible cluster
            if criteria == "Highest average credit score":
                if "CreditScore" not in selected_features:
                    st.error("You must include 'CreditScore' to use this method.")
                else:
                    eligible_cluster = cluster_centers['CreditScore'].idxmax()
            else:
                if not all(x in selected_features for x in ['CreditScore', 'BankBalance']):
                    st.error("You must include both 'CreditScore' and 'BankBalance'.")
                else:
                    eligible_cluster = (cluster_centers['CreditScore'] + cluster_centers['BankBalance']).idxmax()

            df['LoanStatus'] = df['Cluster'].apply(
                lambda x: 'Eligible' if x == eligible_cluster else 'Not Eligible'
            )

            st.subheader("📊 Clustered Data")
            st.dataframe(df[['CustomerID'] + selected_features + ['LoanStatus']])

            # Scatter plot
            st.subheader("🧿 Scatter Plot")
            if len(selected_features) >= 2:
                x_feature, y_feature = selected_features[:2]
                colors = {'Eligible': 'green', 'Not Eligible': 'red'}
                fig, ax = plt.subplots(figsize=(10, 6))
                for status in df['LoanStatus'].unique():
                    subset = df[df['LoanStatus'] == status]
                    ax.scatter(subset[x_feature], subset[y_feature],
                               c=colors[status], label=status, alpha=0.7)
                ax.set_xlabel(x_feature)
                ax.set_ylabel(y_feature)
                ax.set_title('Loan Eligibility Clustering')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
            else:
                st.info("Select at least two features to view scatter plot.")

            # Download
            st.subheader("⬇️ Download Result")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Clustered CSV", csv, "clustered_data.csv", "text/csv")
else:
    st.info("👈 Select demo or upload CSV file to continue.")
