import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Student Placement Predictor", page_icon="🎓", layout="wide")

def make_prediction(features):
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=features)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error from API: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("Gagal terhubung ke backend. Apakah server FastAPI Anda sudah berjalan di port 8000?")
        return None

def main():
    st.title('🎓 Student Placement & Salary Predictor (API Version)')
    st.markdown("""
    Aplikasi ini memprediksi apakah seorang mahasiswa akan mendapatkan pekerjaan (Placed) atau tidak. 
    Aplikasi ini bertindak sebagai Frontend yang berkomunikasi dengan FastAPI Backend.
    """)

    # Sidebar untuk Informasi Aplikasi
    with st.sidebar:
        st.header("Informasi Sistem")
        st.write("Arsitektur Client-Server:")
        st.info("Frontend: Streamlit")
        st.success("Backend: FastAPI (Two-Stage Model)")
        st.divider()

    with st.form("prediction_form"):
        st.subheader("Masukkan Data Mahasiswa")
        
        tab1, tab2, tab3 = st.tabs(["📚 Akademik & Keahlian", "🚀 Pengalaman & Proyek", "👤 Profil Pribadi"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                cgpa = st.number_input("CGPA (0.0 - 10.0)", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
                attendance_percentage = st.slider("Persentase Kehadiran (%)", 0, 100, 85)
                study_hours_per_day = st.number_input("Jam Belajar Per Hari", 0, 24, 4)
                backlogs = st.number_input("Jumlah Backlog", 0, 10, 0)
                branch = st.selectbox("Jurusan (Branch)", ["CSE", "ECE", "IT", "ME", "CE"])
            with col2:
                coding_skill_rating = st.slider("Rating Skill Coding (1-100)", 1, 100, 50)
                communication_skill_rating = st.slider("Rating Skill Komunikasi (1-100)", 1, 100, 50)
                aptitude_skill_rating = st.slider("Rating Skill Aptitude (1-100)", 1, 100, 50)

        with tab2:
            col3, col4 = st.columns(2)
            with col3:
                projects_completed = st.number_input("Jumlah Proyek Diselesaikan", 0, 50, 2)
                internships_completed = st.number_input("Jumlah Magang Diselesaikan", 0, 10, 1)
                hackathons_participated = st.number_input("Jumlah Hackathon Diikuti", 0, 50, 0)
            with col4:
                certifications_count = st.number_input("Jumlah Sertifikasi", 0, 50, 1)
                extracurricular_involvement = st.select_slider("Keterlibatan Ekstrakurikuler", options=["Low", "Medium", "High"], value="Medium")

        with tab3:
            col5, col6 = st.columns(2)
            with col5:
                gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
                city_tier = st.selectbox("Asal Kota", ["Tier 1", "Tier 2", "Tier 3"])
                family_income_level = st.selectbox("Tingkat Pendapatan Keluarga", ["Low", "Medium", "High"])
            with col6:
                part_time_job = st.radio("Memiliki Kerja Part-Time?", ["Yes", "No"])
                internet_access = st.radio("Akses Internet di Rumah?", ["Yes", "No"])
                sleep_hours = st.number_input("Jam Tidur Rata-rata", 0, 24, 7)
                stress_level = st.slider("Tingkat Stres (1-10)", 1, 10, 5)

        submit_button = st.form_submit_button(label="Prediksi via API")

    if submit_button:
        features = {
            'gender': gender,
            'branch': branch,
            'part_time_job': part_time_job,
            'internet_access': internet_access,
            'cgpa': float(cgpa),
            'backlogs': int(backlogs),
            'study_hours_per_day': float(study_hours_per_day),
            'attendance_percentage': float(attendance_percentage),
            'projects_completed': int(projects_completed),
            'internships_completed': int(internships_completed),
            'coding_skill_rating': int(coding_skill_rating),
            'communication_skill_rating': int(communication_skill_rating),
            'aptitude_skill_rating': int(aptitude_skill_rating),
            'hackathons_participated': int(hackathons_participated),
            'certifications_count': int(certifications_count),
            'sleep_hours': float(sleep_hours),
            'stress_level': int(stress_level),
            'family_income_level': family_income_level,
            'city_tier': city_tier,
            'extracurricular_involvement': extracurricular_involvement
        }

        st.divider()
        st.subheader("Hasil Prediksi")

        with st.spinner("Menghubungi FastAPI Backend"):
            result = make_prediction(features)
            
            if result is not None:
                # Mengambil data dari response JSON FastAPI
                placement_status = result.get('placement_status')
                confidence = result.get('placement_probability')
                salary_prediction = result.get('estimated_salary_lpa')

                col_res1, col_res2 = st.columns(2)

                if placement_status == 1:
                    with col_res1:
                        st.success(f"Status: PLACED (Diterima Bekerja)")
                        st.write(f"Tingkat Kepercayaan Model: {confidence*100:.2f}%")
                    
                    with col_res2:
                        st.info("Estimasi Gaji (Salary):")
                        st.metric(label="Lakhs Per Annum (LPA)", value=f"{salary_prediction:.2f} LPA")
                        
                    st.write("Analisis Profil Keahlian Anda:")
                    chart_data = pd.DataFrame({
                        "Keahlian": ["Coding", "Komunikasi", "Aptitude"],
                        "Skor Anda": [coding_skill_rating, communication_skill_rating, aptitude_skill_rating]
                    }).set_index("Keahlian")
                    st.bar_chart(chart_data, color="#2ECC71")

                else:
                    with col_res1:
                        st.error(f"Status: NOT PLACED (Belum Mendapatkan Pekerjaan)")
                        st.write(f"Tingkat Kepercayaan Model: {confidence*100:.2f}%")
                    
                    with col_res2:
                        st.warning("Estimasi Gaji (Salary):")
                        st.metric(label="Lakhs Per Annum (LPA)", value="0.00 LPA")

if __name__ == "__main__":
    main()