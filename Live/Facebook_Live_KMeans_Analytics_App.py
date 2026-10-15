import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Page Configuration
st.set_page_config(
    page_title="Facebook Live Engagement Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Facebook Live Engagement Analytics Dashboard")
st.markdown("Unsupervised Clustering & Dimensionality Reduction using **K-Means** & **PCA**")

# Sidebar - Data Upload & Controls
st.sidebar.header("📁 Data & Controls")
uploaded_file = st.sidebar.file_uploader("Upload Facebook Live CSV File", type=["csv"])

@st.cache_data
def load_data(file):
    if file is not None:
        df = pd.read_csv(file)
    else:
        # Load default or sample if user hasn't uploaded
        try:
            df = pd.read_csv("Live.csv")
        except FileNotFoundError:
            return None
    return df

df_raw = load_data(uploaded_file)

if df_raw is None:
    st.info("👆 कृपया Sidebar मधून 'Live.csv' फाईल अपलोड करा.")
    st.stop()

# ---------------------------------------------------------
# Data Preprocessing
# ---------------------------------------------------------
# Engagement Features List
engagement_cols = [
    'num_reactions', 'num_comments', 'num_shares',
    'num_likes', 'num_loves', 'num_wows', 
    'num_hahas', 'num_sads', 'num_angrys'
]

# Ensure required columns exist
available_features = [col for col in engagement_cols if col in df_raw.columns]

if len(available_features) < len(engagement_cols):
    st.warning("⚠️ फाईलमध्ये सर्व आवश्यक Engagement Columns सापडले नाहीत. उपलब्ध columns वापरले जात आहेत.")

df_clean = df_raw.copy()

# Drop unnecessary columns if present
drop_cols = ['status_id', 'status_published', 'Column1', 'Column2', 'Column3', 'Column4']
df_clean = df_clean.drop(columns=[col for col in drop_cols if col in df_clean.columns])

# Handle missing values in engagement numerical features
for col in available_features:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# Log1p transformation to handle skewness
X_log = np.log1p(df_clean[available_features])

# Standard Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

# ---------------------------------------------------------
# Sidebar Controls for K-Means
# ---------------------------------------------------------
st.sidebar.subheader("⚙️ K-Means Hyperparameters")
max_k = st.sidebar.slider("Select Max K for Evaluation", min_value=3, max_value=10, value=6)

# Pre-calculate Silhouette Scores & Inertia for range of K
k_range = list(range(2, max_k + 1))
inertias = []
silhouette_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

best_k_auto = k_range[np.argmax(silhouette_scores)]

auto_select = st.sidebar.checkbox(f"Auto-select best K (Best Silhouette: K={best_k_auto})", value=True)

if auto_select:
    selected_k = best_k_auto
else:
    selected_k = st.sidebar.slider("Select number of clusters (K)", min_value=2, max_value=max_k, value=best_k_auto)

# Final K-Means Model
kmeans_final = KMeans(n_clusters=selected_k, random_state=42, n_init=10)
df_clean['Cluster'] = kmeans_final.fit_predict(X_scaled)

# PCA Transformation
pca = PCA(n_components=2)
pca_coords = pca.fit_transform(X_scaled)
df_clean['PC1'] = pca_coords[:, 0]
df_clean['PC2'] = pca_coords[:, 1]

# ---------------------------------------------------------
# Multi-Tab Dashboard Interface
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Analytics & Evaluation",
    "🗺️ PCA Visualization",
    "📊 Cluster Profiles",
    "📌 Post Type Insights",
    "🔍 Dataset Explorer",
    "💡 Executive Summary"
])

# ---------------------------------------------------------
# TAB 1: Analytics & Evaluation
# ---------------------------------------------------------
with tab1:
    st.subheader("K-Means Model Evaluation")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(k_range, inertias, marker='o', color='b')
        ax.axvline(x=selected_k, color='r', linestyle='--', label=f'Selected K = {selected_k}')
        ax.set_title("Elbow Method (Inertia)")
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Inertia")
        ax.legend()
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(k_range, silhouette_scores, marker='s', color='g')
        ax.axvline(x=selected_k, color='r', linestyle='--', label=f'Selected K = {selected_k}')
        ax.set_title("Silhouette Score vs. K")
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Silhouette Score")
        ax.legend()
        st.pyplot(fig)

    st.subheader("Cluster Size Distribution")
    cluster_counts = df_clean['Cluster'].value_counts().sort_index()
    st.bar_chart(cluster_counts)

# ---------------------------------------------------------
# TAB 2: PCA Visualization
# ---------------------------------------------------------
with tab2:
    st.subheader("PCA 2D Cluster Scatter Plot")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        x='PC1', y='PC2', hue='Cluster', palette='tab10', 
        data=df_clean, alpha=0.7, ax=ax
    )
    ax.set_title(f"2D PCA Projection (Explained Variance: {sum(pca.explained_variance_ratio_)*100:.2f}%)")
    st.pyplot(fig)

    st.subheader("PCA Feature Loadings")
    loadings_df = pd.DataFrame(
        pca.components_.T, 
        columns=['PC1', 'PC2'], 
        index=available_features
    )
    st.dataframe(loadings_df.style.background_gradient(cmap='coolwarm'))

# ---------------------------------------------------------
# TAB 3: Cluster Profiles
# ---------------------------------------------------------
with tab3:
    st.subheader("Cluster Feature Profiles (Original Scale)")
    metric_type = st.radio("Select Metric:", ["Mean", "Median"])

    if metric_type == "Mean":
        profile_df = df_clean.groupby('Cluster')[available_features].mean()
    else:
        profile_df = df_clean.groupby('Cluster')[available_features].median()

    st.dataframe(profile_df.style.highlight_max(axis=0, color='lightgreen'))

# ---------------------------------------------------------
# TAB 4: Post Type Insights
# ---------------------------------------------------------
with tab4:
    st.subheader("Post Type Distribution across Clusters")
    if 'status_type' in df_clean.columns:
        crosstab = pd.crosstab(df_clean['Cluster'], df_clean['status_type'], normalize='index') * 100
        st.dataframe(crosstab.style.format("{:.2f}%").background_gradient(cmap='YlGnBu'))

        fig, ax = plt.subplots(figsize=(8, 4))
        crosstab.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
        ax.set_title("Status Type Proportion per Cluster (%)")
        ax.set_ylabel("Percentage")
        st.pyplot(fig)
    else:
        st.info("डेटासेटमध्ये 'status_type' Column उपलब्ध नाही.")

# ---------------------------------------------------------
# TAB 5: Dataset Explorer
# ---------------------------------------------------------
with tab5:
    st.subheader("Clustered Dataset Explorer")
    selected_cluster = st.selectbox("Filter by Cluster:", options=["All"] + list(range(selected_k)))

    if selected_cluster == "All":
        filtered_df = df_clean
    else:
        filtered_df = df_clean[df_clean['Cluster'] == selected_cluster]

    st.write(f"Showing {len(filtered_df)} records")
    st.dataframe(filtered_df)

    # Export CSV
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Clustered Data as CSV",
        data=csv_data,
        file_name="clustered_facebook_data.csv",
        mime="text/csv"
    )

# ---------------------------------------------------------
# TAB 6: Executive Summary
# ---------------------------------------------------------
with tab6:
    st.subheader("💡 Key Insights & Model Summary")
    
    current_score = silhouette_scores[k_range.index(selected_k)]
    top_cluster = df_clean.groupby('Cluster')['num_reactions'].mean().idxmax()

    st.markdown(f"""
    - **Total Analyzed Posts:** {len(df_clean):,}
    - **Optimal Clusters Selected (K):** `{selected_k}`
    - **Silhouette Score:** `{current_score:.4f}`
    - **Highest Engagement Cluster:** `Cluster {top_cluster}` (Highest average reactions/likes)
    - **PCA Variance Explained:** `{sum(pca.explained_variance_ratio_)*100:.2f}%` variance is captured by the first two principal components.
    """)