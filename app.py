
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

st.set_page_config(page_title="Prediksi Penyakit Jantung", layout="wide")
st.markdown("# 🪀 Prediksi Penyakit Jantung")
st.markdown("Perbandingan algoritma **K-Nearest Neighbors** dan **Logistic Regression** berdasarkan dataset kesehatan.")

# --- Load Dataset
@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")  # Pastikan file tersedia

df = load_data()

st.subheader("📊 Data Sekilas")
st.dataframe(df.head(), use_container_width=True)

# --- Preprocessing
X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Training Model
knn = KNeighborsClassifier(n_neighbors=5)
logreg = LogisticRegression(max_iter=1000)

knn.fit(X_train, y_train)
logreg.fit(X_train, y_train)

# --- Evaluasi Model
y_pred_knn = knn.predict(X_test)
y_pred_logreg = logreg.predict(X_test)

acc_knn = accuracy_score(y_test, y_pred_knn)
acc_logreg = accuracy_score(y_test, y_pred_logreg)

roc_knn = roc_auc_score(y_test, knn.predict_proba(X_test)[:, 1])
roc_logreg = roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1])

st.subheader("📈 Evaluasi Model (ROC-AUC)")
col1, col2 = st.columns(2)
with col1:
    st.metric("KNN (K=5)", f"{roc_knn:.3f}")
with col2:
    st.metric("Logistic Regression", f"{roc_logreg:.3f}")

# --- ROC Curve
fpr_knn, tpr_knn, _ = roc_curve(y_test, knn.predict_proba(X_test)[:, 1])
fpr_logreg, tpr_logreg, _ = roc_curve(y_test, logreg.predict_proba(X_test)[:, 1])

fig, ax = plt.subplots()
ax.plot(fpr_knn, tpr_knn, label=f"KNN (AUC = {roc_knn:.3f})")
ax.plot(fpr_logreg, tpr_logreg, label=f"Logistic Regression (AUC = {roc_logreg:.3f})")
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()
st.pyplot(fig)

# --- Confusion Matrix
st.subheader("🧹 Confusion Matrix")
col1, col2 = st.columns(2)

def plot_conf_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with col1:
    st.markdown("##### KNN")
    plot_conf_matrix(y_test, y_pred_knn, "KNN")
with col2:
    st.markdown("##### Logistic Regression")
    plot_conf_matrix(y_test, y_pred_logreg, "Logistic Regression")

# --- Classification Report
st.subheader("📋 Classification Report")
with st.expander("Tampilkan Detail"):
    st.text("KNN:\n" + classification_report(y_test, y_pred_knn))
    st.text("Logistic Regression:\n" + classification_report(y_test, y_pred_logreg))

# --- Input Manual untuk Prediksi
st.subheader("🪪 Coba Prediksi dari Input Manual")
with st.form("prediction_form"):
    st.markdown("Masukkan data pasien:")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Umur", 20, 100, 50)
        sex = st.radio("Jenis Kelamin", [0, 1], format_func=lambda x: "Perempuan" if x == 0 else "Laki-laki")
        cp = st.selectbox("Tipe Nyeri Dada (0=Typical Angina, 3=Asymptomatic)", [0, 1, 2, 3])
        trestbps = st.number_input("Tekanan Darah", 80, 200, 120)
        chol = st.number_input("Kolesterol", 100, 600, 200)
        fbs = st.radio("Gula Darah Puasa > 120", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
        restecg = st.selectbox("Hasil EKG", [0, 1, 2])
    with col2:
        thalach = st.number_input("Detak Jantung Maks", 70, 210, 150)
        exang = st.radio("Nyeri saat Olahraga", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
        oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 6.0, 1.0, step=0.1)
        slope = st.selectbox("Kemiringan ST", [0, 1, 2])
        ca = st.selectbox("Jumlah Pembuluh Tersumbat", [0, 1, 2, 3])
        thal = st.selectbox("Thal", [1, 2, 3])

    submitted = st.form_submit_button("Prediksi")

if submitted:
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                                 thalach, exang, oldpeak, slope, ca, thal]], columns=X.columns)

    input_scaled = scaler.transform(input_data)

    pred_knn = knn.predict(input_scaled)[0]
    pred_logreg = logreg.predict(input_scaled)[0]

    proba_knn = knn.predict_proba(input_scaled)[0][1]
    proba_logreg = logreg.predict_proba(input_scaled)[0][1]

    st.success(f"📌 Logistic Regression Prediksi: {'Berisiko' if pred_logreg==1 else 'Tidak Berisiko'} ({proba_logreg*100:.2f}%)")
    st.info(f"📌 KNN Prediksi: {'Berisiko' if pred_knn==1 else 'Tidak Berisiko'} ({proba_knn*100:.2f}%)")
