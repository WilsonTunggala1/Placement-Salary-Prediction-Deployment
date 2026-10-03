# 🎓 Student Placement & Salary Prediction

Aplikasi **Machine Learning end-to-end** untuk memprediksi peluang seorang mahasiswa mendapatkan pekerjaan (*placement*) setelah lulus, sekaligus mengestimasi besaran gaji yang berpotensi diterima.

Project ini mengimplementasikan **two-stage machine learning prediction**, yaitu:

1. **Classification** — memprediksi apakah mahasiswa akan mendapatkan pekerjaan (`Placed` / `Not Placed`).
2. **Regression** — jika mahasiswa diprediksi `Placed`, sistem akan mengestimasi besaran gaji dalam **LPA (Lakhs Per Annum)**.

Project ini dibangun menggunakan arsitektur **client-server**, dengan **FastAPI** sebagai backend/API server, **Streamlit** sebagai frontend, dan **MLflow** untuk experiment tracking serta model management.

---

## 📌 Project Overview

Proses rekrutmen mahasiswa dapat dipengaruhi oleh berbagai faktor, seperti kemampuan akademik, kemampuan komunikasi, kemampuan coding, pengalaman internship, aktivitas ekstrakurikuler, dan kemampuan aptitude.

Project ini bertujuan untuk memanfaatkan data tersebut untuk membangun sistem prediksi yang dapat menjawab dua pertanyaan:

> **"Apakah mahasiswa ini berpotensi mendapatkan pekerjaan?"**

dan apabila jawabannya `Placed`:

> **"Berapa estimasi gaji yang mungkin diterima mahasiswa tersebut?"**

Sistem tidak hanya menghasilkan prediksi, tetapi juga menyediakan visualisasi kemampuan mahasiswa sehingga pengguna dapat memperoleh gambaran mengenai faktor-faktor yang berkaitan dengan hasil prediksi.

---

# 🚀 Key Features

## 1. Two-Stage Prediction

Sistem menggunakan dua model Machine Learning yang bekerja secara berurutan.

### Stage 1 — Placement Classification

Model klasifikasi memprediksi status placement mahasiswa:

```text
Student Data
     │
     ▼
Classification Model
     │
     ├── Not Placed
     │
     └── Placed
              │
              ▼
        Regression Model
              │
              ▼
        Estimated Salary
```

Output klasifikasi:

* `Placed`
* `Not Placed`

Jika hasil prediksi adalah `Not Placed`, proses berhenti dan sistem tidak melakukan prediksi gaji.

Jika hasilnya `Placed`, data akan diteruskan ke model regresi untuk mendapatkan estimasi gaji.

---

## 2. Salary Prediction

Untuk mahasiswa yang diprediksi `Placed`, sistem menjalankan model regresi untuk memperkirakan:

```text
Estimated Salary = X LPA
```

Contoh:

```text
Placement Status : Placed
Estimated Salary : 7.25 LPA
```

Prediksi salary dilakukan secara conditional sehingga model regresi hanya digunakan ketika hasil klasifikasi menunjukkan bahwa mahasiswa berpotensi mendapatkan placement.

---

## 3. Automated ML Pipeline

Project menggunakan `ColumnTransformer` untuk menangani berbagai jenis fitur secara otomatis.

Pipeline dapat menangani:

* Numerical features
* Categorical features
* Ordinal features

Contoh preprocessing:

```text
Raw Data
   │
   ├── Numerical Features
   │       └── Scaling / Transformation
   │
   ├── Categorical Features
   │       └── OneHotEncoder
   │
   └── Ordinal Features
           └── Ordinal Encoding
                │
                ▼
         Machine Learning Model
```

Dengan pendekatan ini, proses preprocessing menjadi lebih konsisten antara:

* training
* validation
* testing
* API inference
* Streamlit application

Hal ini juga membantu mengurangi risiko **data preprocessing mismatch** antara data training dan data prediction.

---

# 🖥️ Interactive Streamlit UI

Frontend dibangun menggunakan **Streamlit** agar pengguna dapat melakukan prediksi melalui interface yang sederhana dan interaktif.

Input mahasiswa dibagi ke dalam beberapa bagian menggunakan **Tabs** dan **Forms**, sehingga pengguna tidak perlu mengisi seluruh fitur dalam satu tampilan yang panjang.

Contoh struktur UI:

```text
Student Placement Prediction
│
├── Personal Information
│
├── Academic Information
│
├── Skills & Performance
│
├── Experience & Activities
│
└── Prediction
```

Setelah user melakukan submit, Streamlit akan mengirimkan data ke backend FastAPI.

---

# ⚡ FastAPI Backend

FastAPI digunakan sebagai REST API server yang bertanggung jawab untuk:

* menerima input mahasiswa
* melakukan validation
* menjalankan preprocessing
* melakukan placement prediction
* melakukan salary prediction jika diperlukan
* mengembalikan hasil prediction ke Streamlit

Contoh flow:

```text
Streamlit
    │
    │ HTTP Request
    ▼
FastAPI
    │
    ├── Validate Input
    │
    ├── Classification Model
    │
    └── Regression Model
    │
    ▼
Prediction Response
    │
    ▼
Streamlit
```

---

# 📊 MLflow Experiment Tracking

Project ini menggunakan **MLflow** untuk melakukan experiment tracking.

MLflow digunakan untuk mencatat informasi penting selama proses training model, seperti:

### Parameters

Contohnya:

```text
model_type
max_depth
learning_rate
n_estimators
random_state
```

### Metrics

Contohnya:

Untuk classification:

```text
accuracy
precision
recall
f1_score
roc_auc
```

Untuk regression:

```text
MAE
MSE
RMSE
R²
```

### Model Artifacts

Model dan pipeline yang telah ditraining dapat disimpan sebagai artifact sehingga model terbaik dapat digunakan kembali pada proses inference.

Contoh experiment flow:

```text
Training
   │
   ▼
MLflow Experiment
   │
   ├── Parameters
   ├── Metrics
   ├── Model
   └── Artifacts
```

---

# 📈 Student Skill Visualization

Setelah prediction dilakukan, aplikasi juga menampilkan visualisasi kemampuan mahasiswa.

Beberapa kemampuan yang dapat divisualisasikan antara lain:

* Coding skill
* Communication skill
* Aptitude score

Visualisasi ini membantu pengguna memahami profil mahasiswa secara lebih intuitif dibandingkan hanya melihat angka prediction.

Contoh:

```text
Coding        ████████████████  80
Communication ██████████████    70
Aptitude      █████████████████ 85
```

---

# 🏗️ System Architecture

Secara keseluruhan, arsitektur aplikasi dapat digambarkan sebagai berikut:

```text
                    ┌─────────────────────┐
                    │      User           │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Streamlit UI     │
                    │     Frontend        │
                    └──────────┬──────────┘
                               │
                         HTTP Request
                               │
                               ▼
                    ┌─────────────────────┐
                    │      FastAPI        │
                    │      Backend        │
                    └──────────┬──────────┘
                               │
                     ┌─────────┴─────────┐
                     ▼                   ▼
             ┌──────────────┐    ┌──────────────┐
             │ Classification│    │  Preprocessing│
             │     Model     │    │   Pipeline    │
             └──────┬───────┘    └──────────────┘
                    │
          ┌─────────┴──────────┐
          │                    │
          ▼                    ▼
    ┌────────────┐       ┌─────────────┐
    │Not Placed  │       │   Placed    │
    └────────────┘       └──────┬──────┘
                                 │
                                 ▼
                         ┌──────────────┐
                         │  Regression  │
                         │     Model    │
                         └──────┬───────┘
                                │
                                ▼
                         Estimated Salary
```

---

# 🧠 Machine Learning Pipeline

## Classification Pipeline

Classification pipeline digunakan untuk memprediksi status placement.

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
```

Secara konseptual:

```text
Input Features
      │
      ▼
ColumnTransformer
      │
      ├── Numerical → Scaling
      │
      ├── Categorical → OneHotEncoder
      │
      └── Ordinal → Ordinal Encoding
      │
      ▼
Classification Model
      │
      ▼
Placed / Not Placed
```

Pipeline ini memastikan preprocessing yang digunakan ketika training sama dengan preprocessing ketika prediction.

---

# 💰 Regression Pipeline

Regression pipeline digunakan untuk memprediksi salary mahasiswa yang telah diprediksi `Placed`.

```text
Student Features
      │
      ▼
Preprocessing Pipeline
      │
      ▼
Best Regression Model
      │
      ▼
Salary Prediction
      │
      ▼
Salary in LPA
```

Model regresi dipilih berdasarkan performa pada evaluation dataset menggunakan metrik regresi yang sesuai.

---

# 📁 Project Structure

Contoh struktur repository:

```text
student-placement-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── eda.ipynb
│   └── model_training.ipynb
│
├── src/
│   ├── preprocessing/
│   ├── training/
│   ├── prediction/
│   └── utils/
│
├── models/
│   ├── classification_model/
│   └── regression_model/
│
├── api/
│   └── main.py
│
├── streamlit/
│   └── app.py
│
├── mlruns/
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

> Struktur di atas dapat disesuaikan dengan struktur aktual repository.

---

# 🛠️ Tech Stack

Project ini menggunakan beberapa teknologi utama:

| Technology           | Purpose                                |
| -------------------- | -------------------------------------- |
| Python               | Programming Language                   |
| Scikit-learn         | Machine Learning & preprocessing       |
| Pandas               | Data manipulation                      |
| NumPy                | Numerical computation                  |
| MLflow               | Experiment tracking & model management |
| FastAPI              | Backend REST API                       |
| Uvicorn              | ASGI server                            |
| Streamlit            | Interactive frontend                   |
| Matplotlib / Seaborn | Data visualization                     |

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/your-username/student-placement-prediction.git

cd student-placement-prediction
```

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

Project terdiri dari tiga komponen utama:

```text
MLflow
   │
   ├── FastAPI Backend
   │
   └── Streamlit Frontend
```

## 1. Start MLflow

```bash
mlflow ui
```

MLflow UI biasanya dapat diakses melalui:

```text
http://localhost:5000
```

---

## 2. Start FastAPI Backend

Contoh:

```bash
uvicorn api.main:app --reload
```

Backend akan berjalan pada:

```text
http://localhost:8000
```

FastAPI juga menyediakan interactive API documentation:

```text
http://localhost:8000/docs
```

---

## 3. Start Streamlit

```bash
streamlit run streamlit/app.py
```

Kemudian buka:

```text
http://localhost:8501
```

---

# 🔌 API Prediction

Backend menyediakan endpoint untuk melakukan prediction.

Contoh request:

```http
POST /predict
```

Request body:

```json
{
  "gender": "Male",
  "ssc_percentage": 75.5,
  "ssc_board": "Central",
  "hsc_percentage": 72.3,
  "hsc_board": "Central",
  "hsc_subject": "Science",
  "degree_percentage": 70.4,
  "degree_type": "Sci&Tech",
  "work_experience": "No",
  "specialisation": "Mkt&Fin",
  "mba_percentage": 65.5,
  "communication_skill": 75,
  "coding_skill": 80,
  "aptitude_score": 85
}
```

Contoh response:

```json
{
  "placement_status": "Placed",
  "predicted_salary": 7.25
}
```

Apabila mahasiswa diprediksi tidak mendapatkan placement:

```json
{
  "placement_status": "Not Placed",
  "predicted_salary": null
}
```

> Nama dan jumlah field pada contoh API harus disesuaikan dengan schema aktual yang digunakan pada project.

---

# 🔄 Prediction Workflow

Ketika pengguna melakukan prediction, sistem menjalankan workflow berikut:

```text
1. User mengisi data mahasiswa
              │
              ▼
2. Streamlit melakukan validation
              │
              ▼
3. Data dikirim ke FastAPI
              │
              ▼
4. FastAPI melakukan preprocessing
              │
              ▼
5. Classification Model
              │
       ┌──────┴──────┐
       ▼             ▼
  Not Placed       Placed
       │             │
       │             ▼
       │      Regression Model
       │             │
       │             ▼
       │       Salary Prediction
       │             │
       └──────┬──────┘
              ▼
6. Response dikirim ke Streamlit
              │
              ▼
7. Prediction + Visualization
```

---

# 📊 Model Evaluation

Model classification dievaluasi menggunakan beberapa metrik, seperti:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

Sedangkan model regression dievaluasi menggunakan:

* MAE
* MSE
* RMSE
* R² Score

Pemilihan model terbaik dilakukan berdasarkan hasil evaluasi pada validation/test dataset.

---

# 🔬 Experiment Tracking

Setiap eksperimen training dapat dicatat menggunakan MLflow.

Contoh:

```python
import mlflow

with mlflow.start_run():

    mlflow.log_param("model", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(
        model,
        "model"
    )
```

Dengan MLflow, eksperimen yang berbeda dapat dibandingkan tanpa harus mencatat hasil secara manual.

---

# 🎯 Why This Project?

Project ini dibuat bukan hanya sebagai model Machine Learning sederhana, tetapi sebagai contoh implementasi **end-to-end ML application**.

Project mencakup seluruh lifecycle:

```text
Data
 │
 ▼
Exploratory Data Analysis
 │
 ▼
Data Preprocessing
 │
 ▼
Feature Engineering
 │
 ▼
Model Training
 │
 ▼
Model Evaluation
 │
 ▼
MLflow Experiment Tracking
 │
 ▼
Model Deployment
 │
 ├── FastAPI
 │
 └── Streamlit
 │
 ▼
Real-time Prediction
```

Dengan pendekatan tersebut, project ini dapat menjadi contoh penerapan Machine Learning dari tahap eksperimen hingga deployment.

---

# 🌟 Highlights

* ✅ Two-stage Machine Learning prediction
* ✅ Classification untuk placement prediction
* ✅ Regression untuk salary prediction
* ✅ Automated preprocessing menggunakan `ColumnTransformer`
* ✅ One-hot encoding untuk categorical features
* ✅ Handling numerical dan ordinal features
* ✅ MLflow experiment tracking
* ✅ REST API menggunakan FastAPI
* ✅ Interactive UI menggunakan Streamlit
* ✅ Real-time prediction
* ✅ Student skill visualization
* ✅ Modular ML pipeline
* ✅ Separation antara frontend dan backend

---

# 🚧 Future Improvements

Beberapa pengembangan yang dapat ditambahkan:

* [ ] Docker & Docker Compose deployment
* [ ] Model Registry menggunakan MLflow
* [ ] Automated model retraining
* [ ] CI/CD pipeline
* [ ] Cloud deployment
* [ ] Authentication untuk API
* [ ] Prediction history
* [ ] Model monitoring
* [ ] Data drift detection
* [ ] SHAP untuk model explainability
* [ ] Batch prediction menggunakan CSV
* [ ] Hyperparameter optimization dengan Optuna

---

# 👨‍💻 Author

**Wilson Pranajaya Tunggala**

Machine Learning | Data Science | Backend Development
