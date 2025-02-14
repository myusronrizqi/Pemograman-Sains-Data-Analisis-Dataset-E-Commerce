import os
import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
import joblib
import numpy as np

# ====================
# CONFIGURASI DASHBOARD
# ====================
st.set_page_config(page_title="E-Commerce Dashboard", layout="wide")

# Judul Utama
st.sidebar.title("📊 DASHBOARD HASIL ANALISIS")

# Logo UNIKOM
st.sidebar.image("logo_unikom_kuning.png", 
                 caption="Universitas Komputer Indonesia (UNIKOM)", 
                 use_container_width=True)

# Informasi Kelompok
st.sidebar.markdown("""
**Kelompok : 4**
\n**Anggota :**
1. 10123224 - Diaz Garcia Pratama
2. 10123236 - Naelza Febrian
3. 10123217 - Muhammad Fathan Fadilah Ihsan
4. 10123245 - Nandyto Rizval Zilbran
5. 10123239 - Benyamin Benedecthus Nikolaus Maryen
6. 10123227 - Muhammad Yusron Rizqi Adhie Putra
""")

# Sidebar untuk navigasi
st.sidebar.header("Navigasi")
menu = st.sidebar.radio(
    "Pilih Menu",
    [
        "Exploratory Data Analysis (EDA)",
        "Seller Performance",
        "Revenue Prediction",
        "Seller Activity Classification",
        "Insights & Knowledge",
    ],
)

# Path ke folder dataset dan model
dataset_path = "E-Commerce Public Dataset/Feature Engineered Dataset/feature_engineered_cleaned_sellers_dataset.csv"
regression_model_path = "E-Commerce Public Dataset/Models/random_forest_regression_model.pkl"
classification_model_path = "E-Commerce Public Dataset/Models/xgboost_classification_model.pkl"

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv(dataset_path)

df = load_data()

# ====================
# EXPLORATORY DATA ANALYSIS (EDA)
# ====================
if menu == "Exploratory Data Analysis (EDA)":
    st.header("📊 Exploratory Data Analysis (EDA)")
    st.write("""
    **Penjelasan:** Bagian ini bertujuan untuk memahami karakteristik dasar dataset melalui statistik deskriptif, distribusi data, dan korelasi antar variabel.
    """)

    # Tab untuk visualisasi
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Data Preview", "Statistik Deskriptif", "Missing Values", "Distribusi Numerik", "Matriks Korelasi"]
    )

    with tab1:
        st.subheader("Data Preview")
        st.write(df.head())
        st.caption("Menampilkan 5 baris pertama dataset untuk memahami struktur data.")

    with tab2:
        st.subheader("Statistik Deskriptif")
        st.write(df.describe())
        st.caption("Ringkasan statistik numerik seperti rata-rata, standar deviasi, minimum, dan maksimum.")

    with tab3:
        st.subheader("Missing Values")
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        if not missing_data.empty:
            fig = px.bar(
                x=missing_data.index,
                y=missing_data.values,
                labels={"x": "Kolom", "y": "Jumlah Missing Values"},
                title="Missing Values per Kolom",
            )
            st.plotly_chart(fig)
            st.caption("Visualisasi jumlah missing values untuk setiap kolom.")
        else:
            st.write("Tidak ada missing values dalam dataset.")

    with tab4:
        st.subheader("Distribusi Data Numerik")
        num_cols = df.select_dtypes(include=["number"]).columns
        selected_num_col = st.selectbox("Pilih kolom untuk histogram:", num_cols)
        fig = px.histogram(df, x=selected_num_col, nbins=30, title=f"Distribusi {selected_num_col}")
        st.plotly_chart(fig)
        st.caption("Histogram untuk memahami distribusi nilai dalam kolom numerik.")

    with tab5:
        st.subheader("Matriks Korelasi")
        num_cols = df.select_dtypes(include=["number"]).columns
        plt.figure(figsize=(10, 6))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        st.pyplot(plt)
        st.caption("Heatmap untuk menunjukkan hubungan antar variabel numerik.")

# ====================
# SELLER PERFORMANCE
# ====================
elif menu == "Seller Performance":
    st.header("📈 Seller Performance Dashboard")
    st.write("""
    **Penjelasan:** Bagian ini menampilkan Key Performance Indicators (KPI) dan visualisasi terkait performa penjual, termasuk pendapatan, waktu pengiriman, dan distribusi geografis.
    """)

    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Orders", df["total_orders"].sum())
    col2.metric("Total Revenue", f"${df['total_revenue'].sum():,.2f}")
    col3.metric("Average Delivery Time", f"{df['average_delivery_time'].mean():.1f} days")
    col4.metric("Avg Monthly Transactions", f"{df['average_monthly_transactions'].mean():.1f}")

    # Visualisasi
    st.subheader("Total Revenue per Seller")
    fig_revenue = px.bar(
        df,
        x="seller_id",
        y="total_revenue",
        title="Total Revenue per Seller",
        labels={"seller_id": "Seller ID", "total_revenue": "Total Revenue"},
    )
    
    st.plotly_chart(fig_revenue, use_container_width=True)
    st.caption("Bar chart pendapatan total per penjual.")

    st.subheader("Orders vs Unique Customers")
    fig_orders = px.scatter(
        df,
        x="total_orders",
        y="seller_count_by_city",
        size="total_revenue",
        color="seller_id",
        title="Orders vs Unique Customers",
    )
    st.plotly_chart(fig_orders, use_container_width=True)
    st.caption("Scatter plot hubungan antara total pesanan dan jumlah pelanggan unik.")

    st.subheader("Average Delivery Time per Seller")
    fig_delivery = px.bar(
        df,
        x="seller_id",
        y="average_delivery_time",
        title="Average Delivery Time per Seller",
        labels={"seller_id": "Seller ID", "average_delivery_time": "Avg Delivery Time (days)"},
    )
    st.plotly_chart(fig_delivery, use_container_width=True)
    st.caption("Bar chart waktu pengiriman rata-rata per penjual.")

    st.subheader("Sellers Distribution by City")
    fig_city = px.bar(
        df,
        x="seller_city",
        y="seller_count_by_city",
        title="Number of Sellers per City",
        labels={"seller_city": "City", "seller_count_by_city": "Number of Sellers"},
        color="seller_city",
    )
    st.plotly_chart(fig_city, use_container_width=True)
    st.caption("Bar chart distribusi penjual berdasarkan kota.")

    st.subheader("Sellers Distribution by State")
    fig_state = px.bar(
        df,
        x="seller_state",
        y="seller_count_by_state",
        title="Number of Sellers per State",
        labels={"seller_state": "State", "seller_count_by_state": "Number of Sellers"},
        color="seller_state",
    )
    st.plotly_chart(fig_state, use_container_width=True)
    st.caption("Bar chart distribusi penjual berdasarkan negara bagian.")

    st.subheader("Revenue vs Distance to Customers")
    fig_distance = px.scatter(
        df,
        x="average_distance_to_customers",
        y="total_revenue",
        size="total_orders",
        color="seller_id",
        title="Revenue vs Distance to Customers",
    )
    st.plotly_chart(fig_distance, use_container_width=True)
    st.caption("Scatter plot hubungan antara pendapatan dan jarak ke pelanggan.")

    st.subheader("Monthly Transactions per Seller")
    fig_transactions = px.histogram(
        df,
        x="average_monthly_transactions",
        nbins=20,
        title="Distribution of Monthly Transactions per Seller",
    )
    st.plotly_chart(fig_transactions, use_container_width=True)
    st.caption("Histogram distribusi transaksi bulanan per penjual.")

# ====================
# REVENUE PREDICTION
# ====================
elif menu == "Revenue Prediction":
    st.header("🔮 Revenue Prediction")
    st.write("""
    **Penjelasan:** Model regresi Random Forest digunakan untuk memprediksi pendapatan penjual berdasarkan fitur-fitur seperti total pesanan, harga rata-rata, dan jarak ke pelanggan.
    """)

    # Load model
    @st.cache_resource
    def load_regression_model():
        return joblib.load(regression_model_path)

    model = load_regression_model()

    # Prediksi pendapatan
    def predict_revenue(model, df):
        features = [
            "total_orders",
            "average_price",
            "average_monthly_transactions",
            "average_distance_to_customers",
            "seller_count_by_city",
            "seller_count_by_state",
        ]
        X = df[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return model.predict(X_scaled)

    df["Predicted Revenue"] = predict_revenue(model, df)

    # Visualisasi: Actual vs Predicted
    st.subheader("Perbandingan Pendapatan Aktual vs Prediksi")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["total_revenue"], df["Predicted Revenue"], alpha=0.7)
    ax.plot(
        [df["total_revenue"].min(), df["total_revenue"].max()],
        [df["total_revenue"].min(), df["total_revenue"].max()],
        "r--",
        label="Ideal Fit",
    )
    ax.set_xlabel("Pendapatan Aktual")
    ax.set_ylabel("Pendapatan Prediksi")
    ax.set_title("Pendapatan Aktual vs Pendapatan Prediksi")
    ax.legend()
    st.pyplot(fig)
    st.caption("Scatter plot untuk membandingkan pendapatan aktual dengan prediksi.")

    # Residual Plot
    st.subheader("Analisis Residual")
    residuals = df["total_revenue"] - df["Predicted Revenue"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["Predicted Revenue"], residuals, alpha=0.7)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Pendapatan Prediksi")
    ax.set_ylabel("Residual")
    ax.set_title("Plot Residual")
    st.pyplot(fig)
    st.caption("Scatter plot residual untuk menganalisis kesalahan prediksi.")

    # Histogram Residual
    st.subheader("Distribusi Residual")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(residuals, kde=True, bins=30, color="blue", ax=ax)
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frekuensi")
    ax.set_title("Histogram Residual")
    st.pyplot(fig)
    st.caption("Histogram distribusi residual.")

    # Feature Importance
    st.subheader("Kepentingan Fitur dalam Model")
    importances = model.feature_importances_
    feature_names = [
        "total_orders",
        "average_price",
        "average_monthly_transactions",
        "average_distance_to_customers",
        "seller_count_by_city",
        "seller_count_by_state",
    ]
    feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(
        by="Importance", ascending=False
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feature_importance_df, x="Importance", y="Feature", palette="viridis", ax=ax)
    ax.set_title("Kepentingan Fitur")
    ax.set_xlabel("Skor Kepentingan")
    ax.set_ylabel("Fitur")
    st.pyplot(fig)
    st.caption("Bar chart pentingnya fitur dalam model.")

    # Evaluasi model
    st.subheader("Evaluasi Model Regresi")
    mse = mean_squared_error(df["total_revenue"], df["Predicted Revenue"])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(df["total_revenue"], df["Predicted Revenue"])
    r2 = r2_score(df["total_revenue"], df["Predicted Revenue"])
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"**R-squared (R²):** {r2:.2f}")
    st.caption("Metrik evaluasi untuk mengukur performa model regresi.")

# ====================
# SELLER ACTIVITY CLASSIFICATION
# ====================
elif menu == "Seller Activity Classification":
    st.header("👥 Seller Activity Classification")
    st.write("""
    **Penjelasan:** Model klasifikasi XGBoost digunakan untuk memprediksi apakah seorang penjual aktif atau tidak berdasarkan fitur-fitur seperti total pesanan dan transaksi bulanan.
    """)

    # Load model
    @st.cache_resource
    def load_classification_model():
        return joblib.load(classification_model_path)

    model = load_classification_model()

    # Preprocessing
    X_classification = df[
        [
            "total_orders",
            "average_price",
            "average_monthly_transactions",
            "average_distance_to_customers",
            "seller_count_by_city",
            "seller_count_by_state",
        ]
    ]
    df["activity_status"] = df["total_orders"].apply(lambda x: "Active" if x > 0 else "Not Active")
    y_classification = df["activity_status"]
    label_encoder = LabelEncoder()
    y_classification_encoded = label_encoder.fit_transform(y_classification)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_classification)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    y_pred = model.predict(X_scaled)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    y_actual_labels = label_encoder.inverse_transform(y_classification_encoded)
    conf_matrix = confusion_matrix(y_actual_labels, y_pred_labels)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Active", "Active"], yticklabels=["Not Active", "Active"])
    st.pyplot(plt)
    st.caption("Heatmap confusion matrix untuk mengevaluasi performa model klasifikasi.")

    # ROC Curve
    st.subheader("ROC Curve")
    y_pred_prob = model.predict_proba(X_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_classification_encoded, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    st.pyplot(plt)
    st.caption("Kurva ROC dengan nilai AUC.")

    # Feature Importance
    st.subheader("Feature Importance")
    importances = model.feature_importances_
    feature_names = X_classification.columns
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(
        by="Importance", ascending=False
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=importance_df, x="Importance", y="Feature", palette="viridis")
    st.pyplot(plt)
    st.caption("Bar chart pentingnya fitur dalam model.")

    # Evaluasi Model
    st.subheader("Evaluasi Model Klasifikasi")
    classification_rep = classification_report(
        y_classification_encoded, y_pred, target_names=["Not Active", "Active"], output_dict=True
    )
    classification_df = pd.DataFrame(classification_rep).transpose()

    # Menampilkan hasil evaluasi dalam tabel
    st.write("### Classification Report")
    st.dataframe(classification_df.style.format("{:.2f}"))

    # Menampilkan metrik evaluasi utama
    accuracy = classification_rep["accuracy"]
    precision = classification_rep["Active"]["precision"]
    recall = classification_rep["Active"]["recall"]
    f1_score = classification_rep["Active"]["f1-score"]

    st.write(f"**Accuracy:** {accuracy:.2f}")
    st.write(f"**Precision:** {precision:.2f}")
    st.write(f"**Recall:** {recall:.2f}")
    st.write(f"**F1-Score:** {f1_score:.2f}")

# ====================
# INSIGHTS & KNOWLEDGE
# ====================
elif menu == "Insights & Knowledge":
    st.header("💡 Insights & Knowledge")
    st.write("""
    **Penjelasan:** Berikut adalah beberapa insight penting yang dapat diambil dari analisis data dan prediksi:
    - **EDA:** Dataset memiliki distribusi yang cukup merata untuk variabel numerik, namun ada beberapa kolom dengan missing values yang perlu ditangani.
    - **Seller Performance:** Penjual dengan jumlah pesanan tinggi cenderung memiliki pendapatan lebih besar dan waktu pengiriman lebih cepat.
    - **Revenue Prediction:** Model regresi Random Forest berhasil memprediksi pendapatan dengan akurasi yang baik (R² > 0.8).
    - **Seller Activity Classification:** Model XGBoost efektif dalam membedakan penjual aktif dan tidak aktif berdasarkan fitur transaksi.
    """)

    st.subheader("Rekomendasi Bisnis")
    st.write("""
    - Fokus pada penjual dengan potensi pendapatan tinggi untuk meningkatkan kontribusi mereka.
    - Tingkatkan efisiensi logistik untuk mengurangi waktu pengiriman.
    - Identifikasi penjual tidak aktif dan berikan insentif untuk meningkatkan aktivitas mereka.
    """)