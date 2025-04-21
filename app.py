import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Prediksi Penyakit Jantung", layout="wide")
st.title("🫀 Prediksi Penyakit Jantung: KNN vs Logistic Regression")

# --- Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("heart.csv")  # Ubah ini jika nama file berbeda
    return df

df = load_data()

st.subheader("📊 Data Sekilas")
st.dataframe(df.head())

# --- Preprocessing
X = df.drop("target", axis=1)  # Pastikan kolom label bernama 'target'
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

st.subheader("📈 Akurasi Model")
st.write(f"**KNN (K=5):** {acc_knn:.2f}")
st.write(f"**Logistic Regression:** {acc_logreg:.2f}")

# --- Confusion Matrix
def plot_conf_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(plt.gcf())
    plt.clf()

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Confusion Matrix - KNN")
    plot_conf_matrix(y_test, y_pred_knn, "KNN")
with col2:
    st.markdown("#### Confusion Matrix - Logistic Regression")
    plot_conf_matrix(y_test, y_pred_logreg, "Logistic Regression")

# --- Classification Report
st.subheader("📋 Classification Report")
with st.expander("Lihat Detail"):
    st.text("KNN:\n" + classification_report(y_test, y_pred_knn))
    st.text("Logistic Regression:\n" + classification_report(y_test, y_pred_logreg))

# --- Input Manual untuk Prediksi
st.subheader("🧪 Coba Prediksi dari Input Manual")

with st.form("prediction_form"):
    st.write("Masukkan data pasien:")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Umur", 20, 100, 50)
        sex = st.selectbox("Jenis Kelamin (0=Perempuan, 1=Laki-laki)", [0, 1])
        cp = st.selectbox("Tipe Nyeri Dada (0-3)", [0, 1, 2, 3])
        trestbps = st.number_input("Tekanan Darah", 80, 200, 120)
        chol = st.number_input("Kolesterol", 100, 600, 200)
        fbs = st.selectbox("Gula Darah Puasa > 120 (0=Tidak, 1=Ya)", [0, 1])
        restecg = st.selectbox("Hasil EKG (0-2)", [0, 1, 2])

    with col2:
        thalach = st.number_input("Detak Jantung Maks", 70, 210, 150)
        exang = st.selectbox("Nyeri Dada saat Olahraga (0=Tidak, 1=Ya)", [0, 1])
        oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 6.0, 1.0, step=0.1)
        slope = st.selectbox("Kemiringan ST (0-2)", [0, 1, 2])
        ca = st.selectbox("Jumlah Pembuluh Tersumbat (0-3)", [0, 1, 2, 3])
        thal = st.selectbox("Thal (1=Normal, 2=Fixed, 3=Reversable)", [1, 2, 3])

    submitted = st.form_submit_button("Prediksi")

if submitted:
    input_data = pd.DataFrame([[
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    ]], columns=X.columns)

    input_scaled = scaler.transform(input_data)

    # Prediksi kelas
    pred_knn = knn.predict(input_scaled)[0]
    pred_logreg = logreg.predict(input_scaled)[0]

    # Prediksi probabilitas
    proba_knn = knn.predict_proba(input_scaled)[0][1]  # Probabilitas kelas 1
    proba_logreg = logreg.predict_proba(input_scaled)[0][1]

    # Tampilkan hasil dengan persen
    st.success(f"📌 Logistic Regression Prediksi: {'Berisiko' if pred_logreg==1 else 'Tidak Berisiko'} ({proba_logreg*100:.2f}%)")
    st.info(f"📌 KNN Prediksi: {'Berisiko' if pred_knn==1 else 'Tidak Berisiko'} ({proba_knn*100:.2f}%)")
