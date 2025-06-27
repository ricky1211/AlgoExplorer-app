import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Set page config
st.set_page_config(page_title="Machine Learning App", layout="wide")

# Title dengan kredit kelompok
st.title("Machine Learning App")
st.markdown("""
**Dibuat oleh Kelompok 4**  
Anggota:
- Dafa Alfiana Erlangga
- Diska Kurnia Azzahra
- Akram Satya Ramdhani Putra
""")

st.write("""
Aplikasi ini memungkinkan Anda untuk mengupload dataset dan menerapkan berbagai algoritma machine learning.
""")

# Sidebar for user input
with st.sidebar:
    st.header("Pengaturan")
    uploaded_file = st.file_uploader("Upload dataset (CSV)", type=["csv"])
    
    st.subheader("Pilihan Algoritma")
    algorithm_type = st.selectbox("Tipe Algoritma", 
                                ["Klasifikasi", "Regresi", "Clustering"])
    
    algorithm = None
    if algorithm_type == "Klasifikasi":
        algorithm = st.selectbox("Pilih Algoritma", 
                                ["Logistic Regression", "Naive Bayes", "SVM", "K-NN", "Decision Tree"])
    elif algorithm_type == "Regresi":
        algorithm = st.selectbox("Pilih Algoritma", ["Linear Regression"])
    elif algorithm_type == "Clustering":
        algorithm = st.selectbox("Pilih Algoritma", ["K-Means"])
    
    if algorithm_type != "Clustering":
        test_size = st.slider("Ukuran Data Testing (%)", 10, 40, 20)
        random_state = st.slider("Random State", 0, 100, 42)
    
    if algorithm == "K-Means":
        n_clusters = st.slider("Jumlah Cluster", 2, 10, 3)
    
    if algorithm in ["SVM", "K-NN"]:
        if algorithm == "SVM":
            kernel = st.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
            C = st.slider("C (Regularization)", 0.1, 10.0, 1.0)
        elif algorithm == "K-NN":
            n_neighbors = st.slider("Jumlah Tetangga", 1, 15, 5)

# Main content
if uploaded_file is not None:
    # Load data
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Preview Dataset")
        st.write(df.head())
        
        # Show data info
        st.subheader("Informasi Dataset")
        st.write(f"Jumlah Baris: {df.shape[0]}")
        st.write(f"Jumlah Kolom: {df.shape[1]}")
        
        # Show columns
        st.write("\nKolom:")
        st.write(df.columns.tolist())
        
        # Select features and target
        st.subheader("Pilih Fitur dan Target")
        cols = df.columns.tolist()
        
        if algorithm_type != "Clustering":
            target_col = st.selectbox("Pilih Kolom Target", cols)
            feature_cols = st.multiselect("Pilih Kolom Fitur", [col for col in cols if col != target_col])
        else:
            feature_cols = st.multiselect("Pilih Kolom Fitur", cols)
        
        if (algorithm_type != "Clustering" and len(feature_cols) > 0 and target_col) or (algorithm_type == "Clustering" and len(feature_cols) > 0):
            # Prepare data
            X = df[feature_cols]
            
            if algorithm_type != "Clustering":
                y = df[target_col]
                
                # Encode categorical target for classification
                if algorithm_type == "Klasifikasi" and y.dtype == 'object':
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size/100, random_state=random_state)
                
                # Scale features
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            
            else:
                # Scale features for clustering
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            
            # Initialize and train model
            model = None
            if algorithm == "Logistic Regression":
                model = LogisticRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
            elif algorithm == "Naive Bayes":
                model = GaussianNB()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
            elif algorithm == "SVM":
                model = SVC(kernel=kernel, C=C)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
            elif algorithm == "K-NN":
                model = KNeighborsClassifier(n_neighbors=n_neighbors)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
            elif algorithm == "Decision Tree":
                model = DecisionTreeClassifier(random_state=random_state)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
            elif algorithm == "Linear Regression":
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
            elif algorithm == "K-Means":
                model = KMeans(n_clusters=n_clusters, random_state=random_state)
                clusters = model.fit_predict(X_scaled)
                df['Cluster'] = clusters
                
            # Show results
            st.subheader("Hasil")
            
            if algorithm_type != "Clustering":
                # Show accuracy
                if algorithm_type == "Klasifikasi":
                    accuracy = accuracy_score(y_test, y_pred)
                    st.write(f"Akurasi: {accuracy:.2f}")
                    
                    # Confusion matrix
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    st.pyplot(fig)
                    
                    # Classification report
                    st.subheader("Classification Report")
                    report = classification_report(y_test, y_pred, output_dict=True)
                    st.table(pd.DataFrame(report).transpose())
                
                elif algorithm_type == "Regresi":
                    # Regression metrics
                    from sklearn.metrics import mean_squared_error, r2_score
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    st.write(f"Mean Squared Error: {mse:.2f}")
                    st.write(f"R-squared: {r2:.2f}")
                    
                    # Plot actual vs predicted
                    st.subheader("Actual vs Predicted")
                    fig, ax = plt.subplots()
                    ax.scatter(y_test, y_pred)
                    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
                    ax.set_xlabel('Actual')
                    ax.set_ylabel('Predicted')
                    st.pyplot(fig)
                
                # Feature importance for some models
                if algorithm in ["Logistic Regression", "Decision Tree", "Linear Regression"]:
                    st.subheader("Feature Importance")
                    
                    if algorithm == "Logistic Regression":
                        importance = model.coef_[0]
                    elif algorithm == "Decision Tree":
                        importance = model.feature_importances_
                    elif algorithm == "Linear Regression":
                        importance = model.coef_
                    
                    fig, ax = plt.subplots()
                    ax.barh(feature_cols, importance)
                    ax.set_xlabel('Importance')
                    st.pyplot(fig)
                    
                # Decision Tree visualization
                if algorithm == "Decision Tree":
                    st.subheader("Decision Tree Visualization")
                    fig, ax = plt.subplots(figsize=(12, 8))
                    plot_tree(model, feature_names=feature_cols, class_names=[str(c) for c in model.classes_], 
                             filled=True, ax=ax, max_depth=3)
                    st.pyplot(fig)
            
            else:
                # Clustering results
                st.write(f"Silhouette Score: {silhouette_score(X_scaled, clusters):.2f}")
                
                # Cluster distribution
                st.subheader("Distribusi Cluster")
                fig, ax = plt.subplots()
                sns.countplot(x='Cluster', data=df, ax=ax)
                st.pyplot(fig)
                
                # Cluster visualization (first 2 features)
                if len(feature_cols) >= 2:
                    st.subheader("Visualisasi Cluster (2 Fitur Pertama)")
                    fig, ax = plt.subplots()
                    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis')
                    ax.set_xlabel(feature_cols[0])
                    ax.set_ylabel(feature_cols[1])
                    st.pyplot(fig)
                
                # Show data with clusters
                st.subheader("Data dengan Cluster")
                st.write(df)
                
    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.info("Silakan upload dataset CSV untuk memulai analisis.")

# Footer dengan kredit kelompok
st.markdown("---")
st.markdown("""
**Aplikasi Machine Learning dengan Streamlit**  
Dikembangkan oleh Kelompok 4 TI.22.A4
Praktikum Data Mining 2025
""")