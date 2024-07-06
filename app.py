# app.py
#import library

import pickle
import streamlit as st
import pandas as pd
import numpy as np
import keras
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import math
import joblib
import tensorflow as tf
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.models import load_model

def main():
    st.set_page_config(
    page_title="PREDIKSI CURAH HUJAN MENGGUNAKAN LSTM DAN K-NN DALAM IMPUTASI MISSING VALUE"
)
    st.title('PREDIKSI CURAH HUJAN MENGGUNAKAN LSTM DAN K-NN DALAM IMPUTASI MISSING VALUE')

    tab1, tab2, tab3 = st.tabs(["Data Understanding", "Imputasi Missing Value Menggunakan KNN", "Prediksi Selanjutnya"])
    
    with tab1:
        st.write("""
        <h5>Data Understanding</h5>
        <br>
        """, unsafe_allow_html=True)

        st.markdown("""
        Link Dataset:
        https://dataonline.bmkg.go.id
        """, unsafe_allow_html=True)


        st.write('Dataset ini berisi tentang curah hujan')
        missing_values = ['8888']
        df = pd.read_excel('Dataset_Curah_Hujan.xlsx', na_values = missing_values)
        st.write("Dataset Curah Hujan : ")
        st.write(df)
        
    with tab2:
        st.write("""
        <h5>Imputasi Missing Value Menggunakan KNN</h5>
        <br>
        """, unsafe_allow_html=True)
        
        model_knn = st.radio("Pemodelan", ('Imputasi Missing Value', 'Normalisasi Data', 'Prediksi Menggunakan LSTM', 'Grafik Perbandingan Data Asli dengan Hasil Prediksi'))
        if model_knn == 'Imputasi Missing Value':
            missing_values = ['8888']
            df = pd.read_excel('Dataset_Curah_Hujan.xlsx', na_values = missing_values)
            missing_data = df[df.isna().any(axis=1)]
            st.write('Data yang Mempunyai Missing Value :')
            st.write(missing_data)
            k = st.selectbox("Pilih nilai k (jumlah tetangga terdekat) :", [3, 4, 5])
            preprocessing = KNNImputer(n_neighbors=k)
            data_imputed = preprocessing.fit_transform(df[['RR']])
            df_imputed = df.copy()
            df_imputed['RR_Imputed'] = data_imputed
            df_comparison = df_imputed[['Tanggal', 'RR', 'RR_Imputed']]
            st.write('Data yang telah dilakukan Proses Imputasi Missing Value dengan KNN')
            st.write(df_comparison)
            
        elif model_knn == 'Normalisasi Data':
            st.write('Data yang telah Dilakukan Proses Normalisasi Data :')
            df_normalisasi = pd.read_csv('normalisasi_n_4_fix.csv')
            df_normalisasi = df_normalisasi.round(2)
            st.write(df_normalisasi)

        elif model_knn == 'Prediksi Menggunakan LSTM':
            df_imputed = pd.read_csv('imputasi_n_4_fix.csv')
            df_imputed = df_imputed.round(2)
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df_imputed[['RR']])
            values = scaled_data
            training_data_len = math.ceil(len(values) * 0.7)
            # Menampilkan hasil prediksi
            st.write("Hasil Prediksi:")
            df_prediksi = pd.read_csv('predictions_fix.csv')
            df_prediksi = df_prediksi.round(2)
            st.write(df_prediksi)
            df_prediksi_de = scaler.inverse_transform(df_prediksi)
            df_prediksi_de = df_prediksi_de.round(2)
            
            # Menampilkan MAPE
            y_test = pd.read_csv('interpolasi_n_4_splitdata_0.7_epochs_50_lr_0.01_ts_50.csv')
            nilai_mape_uji = np.mean(np.abs((df_imputed['RR'][training_data_len:] - y_test['RR'][training_data_len:]) / df_imputed['RR'][training_data_len:])) * 100
            nilai_mape_uji = nilai_mape_uji.round(2)
            st.write('MAPE : ')
            st.write(nilai_mape_uji)

        elif model_knn == 'Grafik Perbandingan Data Asli dengan Hasil Prediksi':
            df_imputed = pd.read_csv('imputasi_n_4_fix.csv')
            df_imputed = df_imputed.round(2)
            df_imputed['Tanggal'] = pd.to_datetime(df_imputed['Tanggal'])
            df_prediksi = pd.read_csv('predictions_fix.csv')
            df_prediksi = df_prediksi.round(2)
            
            plt.figure(figsize=(20, 7))
            plt.plot(df_imputed['Tanggal'], df_imputed['RR'], color='blue', label='Curah Hujan Asli')
            plt.plot(df_imputed['Tanggal'][1161:], df_prediksi['prediksi'], color='red', label='Prediksi Curah Hujan')
            plt.title('Prediksi Curah Hujan')
            plt.xlabel('Tanggal')
            plt.ylabel('Curah Hujan (mm)')
            plt.legend()
            # Menampilkan plot di Streamlit
            st.pyplot(plt)
         
    with tab3:
        n = 356  # Example: Predict the next 10 time steps
        future_predictions = []
        x_test = pd.read_csv('xtest_knn_n_4_epochs_50_lr_0.01_ts_50_fix.csv')
        x_test = x_test.round(2)
        df_imputed = pd.read_csv('imputasi_n_4_fix.csv')
        df_imputed = df_imputed.round(2)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df_imputed[['RR']])
        scaled_data_df = pd.DataFrame(scaled_data)
        values = scaled_data_df.values
        df_normalisasi = pd.read_csv('normalisasi_n_4_fix.csv')
        df_normalisasi = df_normalisasi.round(2)
        df_imputed['Tanggal'] = pd.to_datetime(df_imputed['Tanggal'])
        model_path = 'model_knn_n_4_epochs_50_lr_0.01_ts_50_fix.h5'
        model = tf.keras.models.load_model(model_path)
        df_prediksi = pd.read_csv('predictions_fix.csv')
        df_prediksi = df_prediksi.round(2)
        x_last_window = np.array(x_test['x_test'].values[-50:], dtype=np.float32).reshape((1, 50, 1))
        
        for _ in range(n):
            # Predict the next time step
            prediction = model.predict(x_last_window)

            # Append the prediction to the list of future predictions
            future_predictions.append(prediction[0])
        
            # Update the last window by removing the first element and appending the prediction
            x_last_window = np.append(x_last_window[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)
        
        # Convert the list of future predictions to a numpy array
        future_predictions = np.array(future_predictions)
        future_predictions = future_predictions.round(2)
        
        # Inverse transform predictions to get the original scale
        future_predictions_denormalisasi = scaler.inverse_transform(future_predictions)
        future_predictions_denormalisasi = future_predictions_denormalisasi.round(2)
        future_predictions_df = pd.DataFrame(future_predictions_denormalisasi, columns=['Prediksi'])
        st.write('Prediksi Selanjutnya : ')
        st.write(future_predictions_df)
        
        # Plotting the predictions
        plt.figure(figsize=(12, 6))
        plt.plot(df_imputed['Tanggal'].iloc[-50:], df_imputed['RR'].iloc[-50:], label='Curah Hujan Asli', color='green')
        plt.plot(df_imputed['Tanggal'].iloc[-50:], df_prediksi[-50:], label='Hasil Prediksi', color='orange')
        future_dates = pd.date_range(start=df_imputed['Tanggal'].iloc[-1], periods=n+1, closed='right')
        plt.plot(future_dates, future_predictions_df, color='red', label='Prediksi 2 Hari Selanjutnya')
        
        plt.title('Prediksi Curah Hujan Selanjutnya')
        plt.xlabel('Tanggal')
        plt.ylabel('Curah Hujan (mm)')
        plt.legend()
        st.pyplot(plt)
        
if __name__ == "__main__":
    main()
