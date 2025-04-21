import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Konfigurasi Halaman
st.set_page_config(page_title="Prediksi Penyakit Jantung", page_icon="🫀", layout="wide")

st.title("🫀 Prediksi Penyakit Jantung: KNN vs Logistic Regression")
st.markdown("Aplikasi ini membandingkan dua model machine learning untuk memprediksi risiko penyakit jantung.")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("heart.csv")  # Ganti jika nama file berbeda
    return df

df = load_data()

st.subheader("📊 Data Sekilas")
st.dataframe(df.head(), use_container_width=True)

# Preprocessing
X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Models
knn = KNeighborsClassifier(n_neighbors=5)
logreg = LogisticRegression(max_iter=1000)

knn.fit(X_train, y_train)
logreg.fit(X_train, y_train)

# Predictions
y_pred_knn = knn.predict(X_test)
y_pred_logreg = logreg.predict(X_test)

acc_knn = accuracy_score(y_test, y_pred_knn)
acc_logreg = accuracy_score(y_test, y_pred_logreg)

# Akurasi
st.subheader("📈 Akurasi Model")
col1, col2 = st.columns(2)
col1.metric("KNN (K=5)", f"{acc_knn*100:.2f}%", delta=None)
col2.metric("Logistic Regression", f"{acc_logreg*100:.2f}%", delta=None)

# Confusion Matrix
def plot_conf_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

st.markdown("---")
st.subheader("📌 Confusion Matrix")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**KNN**")
    plot_conf_matrix(y_test, y_pred_knn, "KNN")
with col2:
    st.markdown("**Logistic Regression**")
    plot_conf_matrix(y_test, y_pred_logreg, "Logistic Regression")

# Classification Report
st.markdown("---")
st.subheader("📋 Classification Report")
with st.expander("Klik untuk melihat laporan lengkap"):
    st.text("KNN:\n" + classification_report(y_test, y_pred_knn))
    st.text("Logistic Regression:\n" + classification_report(y_test, y_pred_logreg))

# Input Manual
st.markdown("---")
st.subheader("🧪 Prediksi dari Input Manual")

with st.form("prediction_form"):
    st.markdown("Masukkan data pasien:")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Umur", 20, 100, 50)
        sex = st.selectbox("Jenis Kelamin", ["Perempuan (0)", "Laki-laki (1)"])
        cp = st.selectbox("Tipe Nyeri Dada", ["0", "1", "2", "3"])
        trestbps = st.number_input("Tekanan Darah", 80, 200, 120)
        chol = st.number_input("Kolesterol", 100, 600, 200)
        fbs = st.selectbox("Gula Darah Puasa > 120", ["Tidak (0)", "Ya (1)"])
        restecg = st.selectbox("Hasil EKG", ["0", "1", "2"])

    with col2:
        thalach = st.number_input("Detak Jantung Maks", 70, 210, 150)
        exang = st.selectbox("Nyeri Dada saat Olahraga", ["Tidak (0)", "Ya (1)"])
        oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 6.0, 1.0, step=0.1)
        slope = st.selectbox("Kemiringan ST", ["0", "1", "2"])
        ca = st.selectbox("Jumlah Pembuluh Tersumbat", ["0", "1", "2", "3"])
        thal = st.selectbox("Thal", ["Normal (1)", "Fixed Defect (2)", "Reversable Defect (3)"])

    submitted = st.form_submit_button("🔍 Prediksi")

if submitted:
    input_data = pd.DataFrame([[age,
                                int(sex.split()[-1]),
                                int(cp),
                                trestbps,
                                chol,
                                int(fbs.split()[-1]),
                                int(restecg),
                                thalach,
                                int(exang.split()[-1]),
                                oldpeak,
                                int(slope),
                                int(ca),
                                int(thal.split()[-1])
                                ]], columns=X.columns)

    input_scaled = scaler.transform(input_data)

    pred_knn = knn.predict(input_scaled)[0]
    pred_logreg = logreg.predict(input_scaled)[0]
    proba_knn = knn.predict_proba(input_scaled)[0][1]
    proba_logreg = logreg.predict_proba(input_scaled)[0][1]

    st.markdown("### 🔮 Hasil Prediksi")
    col1, col2 = st.columns(2)
    col1.success(f"**Logistic Regression**: {'🟥 Berisiko' if pred_logreg==1 else '✅ Tidak Berisiko'} ({proba_logreg*100:.2f}%)")
    col2.info(f"**KNN**: {'🟥 Berisiko' if pred_knn==1 else '✅ Tidak Berisiko'} ({proba_knn*100:.2f}%)")

# Footer
st.markdown("---")
st.caption("© 2025 | Dibuat oleh [Diky Juni Purwanto] | Powered by Streamlit & Scikit-Learn")
