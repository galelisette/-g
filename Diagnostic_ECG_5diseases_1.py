import pandas as pd
import pickle
import math
import streamlit as st
import os

st.set_page_config(page_title="ECG Disease Recognition", layout="wide")

st.write("""# Recognition of diseases by ECG
**Based on the works of Gale Lisette**""")

def check_model_files():
    required_files = [
        'Gr_Boost_ECG_healthy.pkl',
        'Gr_Boost_ECG_veget_dyst.pkl',
        'Gr_Boost_ECG_ischemia.pkl',
        'Gr_Boost_ECG_ulcer.pkl',
        'Gr_Boost_ECG_gallstone.pkl',
        'Gr_Boost_ECG_nodular_thyroid_goiter.pkl'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    return missing_files

missing_files = check_model_files()
if missing_files:
    st.warning(f"Отсутствуют файлы моделей: {', '.join(missing_files)}")
    st.info("Приложение будет работать в демонстрационном режиме без реальных предсказаний")

uploaded_file = st.file_uploader("Select ECG file *.csv", type="csv")

if uploaded_file is not None:
    try:
        try:
            df_ecg = pd.read_csv(uploaded_file, sep=";")
        except:
            uploaded_file.seek(0)  
            try:
                df_ecg = pd.read_csv(uploaded_file, sep=",")
            except:
                uploaded_file.seek(0)
                df_ecg = pd.read_csv(uploaded_file)
        
        st.success(f"Файл загружен успешно! Форма данных: {df_ecg.shape}")
        st.write("Первые 5 строк данных:")
        st.write(df_ecg.head())
        st.write("Столбцы в файле:", list(df_ecg.columns))
        
        if 'mV' not in df_ecg.columns:
            st.error("CSV файл должен содержать столбец 'mV'")
            st.write("Доступные столбцы:", list(df_ecg.columns))
            st.stop()
            
        if 'Seconds' not in df_ecg.columns:
            st.warning("Столбец 'Seconds' не найден. Создаем на основе индекса...")
            df_ecg['Seconds'] = df_ecg.index * 0.001  
        
        df_ecg = df_ecg.dropna(subset=['mV'])
        
        if len(df_ecg) < 100:
            st.error("Недостаточно данных для анализа (менее 100 точек)")
            st.stop()

        col1, col2 = st.columns([1, 3])
        with col1:
            st.subheader("Data of ECG")
            st.write(f"Количество точек: {len(df_ecg)}")
            st.write(df_ecg.head(10))
            
        with col2:
            st.subheader("Chart of ECG")
            max_points = min(3000, len(df_ecg))
            chart_data = df_ecg.iloc[:max_points]['mV'].reset_index(drop=True)
            st.line_chart(chart_data, use_container_width=True)
        
        st.write('Analyzing ECG...')
        
        last_point = len(df_ecg)
        vol_of_mean = max(1, int(last_point//330+1))
        last_mean_point = max(0, int(last_point-vol_of_mean))
        
        if last_mean_point >= last_point:
            last_mean_point = last_point - vol_of_mean
            
        last_mean = df_ecg.iloc[last_mean_point:last_point]['mV'].mean()
        shift = 0.3
        last_threshold = last_mean + shift
        
        df_ecg = df_ecg.copy() 
        df_ecg['Threshold'] = 0.0

        st.write('Calculating thresholds...')
        progress_bar = st.progress(0)
        
        for i in range(min(last_mean_point + 1, last_point)):
            start_idx = i
            end_idx = min(i + vol_of_mean, last_point)
            
            if end_idx > start_idx:
                threshold_val = df_ecg.iloc[start_idx:end_idx]['mV'].mean() + shift
                df_ecg.iloc[i, df_ecg.columns.get_loc('Threshold')] = threshold_val
            
            if i % 100 == 0: 
                progress = min(100, int(100 * (i + 1) / (last_mean_point + 1)))
                progress_bar.progress(progress)
        
      
        df_ecg.iloc[last_mean_point:, df_ecg.columns.get_loc('Threshold')] = last_threshold
        progress_bar.progress(100)

    
        st.write('Finding amplitude peaks...')
        amplitude = []
        second_of_amplitude = []
        
        progress_bar = st.progress(0)
        
        for i in range(1, last_point-1):
            mv_current = df_ecg.iloc[i]['mV']
            mv_prev = df_ecg.iloc[i-1]['mV']
            mv_next = df_ecg.iloc[i+1]['mV']
            threshold_current = df_ecg.iloc[i]['Threshold']
            
            if mv_current >= threshold_current:
                if mv_current - mv_prev >= 0 and mv_current - mv_next > 0:
                    amplitude.append(mv_current)
                    second_of_amplitude.append(df_ecg.iloc[i]['Seconds'])
            
            if i % 1000 == 0:  
                progress = min(100, int(100 * i / (last_point - 2)))
                progress_bar.progress(progress)
        
        progress_bar.progress(100)

        max_peaks = 600
        if len(amplitude) > max_peaks:
            amplitude = amplitude[:max_peaks]
            second_of_amplitude = second_of_amplitude[:max_peaks]

        st.write(f'Found {len(amplitude)} amplitude peaks')

        if len(amplitude) < 5:
            st.error("Недостаточно пиков для анализа. Попробуйте другой файл ECG или проверьте качество сигнала.")
            st.stop()

        st.write('Calculating intervals and phases...')
        interval = []
        phase = []
        
        for i in range(len(second_of_amplitude)-1):
            interval_val = second_of_amplitude[i+1] - second_of_amplitude[i]
            interval.append(interval_val)
        
        for i in range(len(amplitude)-1):
            if i < len(interval) and interval[i] != 0:  
                phase_val = math.atan(amplitude[i] / interval[i])
                phase.append(phase_val)
            else:
                phase.append(0.0)

        min_length = min(len(interval), len(amplitude)-1, len(phase))
        interval = interval[:min_length]
        phase = phase[:min_length]
        
        if min_length < 3:
            st.error("Недостаточно данных для создания кодограммы.")
            st.stop()

        st.write('Calculating deltas...')
        delta_interval = []
        delta_amplitude = []
        delta_phase = []
        
        for i in range(min_length-1):
            delta_interval.append(round(interval[i+1] - interval[i], 3))
            delta_amplitude.append(round(amplitude[i+1] - amplitude[i], 3))
            delta_phase.append(round(phase[i+1] - phase[i], 6))

        if len(delta_interval) == 0:
            st.error("Недостаточно данных для создания кодограммы.")
            st.stop()

        st.write('Creating codogram...')
        cod = []
        for i in range(len(delta_interval)):
            di = delta_interval[i]
            da = delta_amplitude[i]
            dp = delta_phase[i]
            
            if di >= 0 and da >= 0 and dp >= 0:
                cod.append('A')
            elif di < 0 and da < 0 and dp >= 0:
                cod.append('B')
            elif di < 0 and da >= 0 and dp >= 0:
                cod.append('C')
            elif di >= 0 and da < 0 and dp < 0:
                cod.append('D')
            elif di >= 0 and da >= 0 and dp < 0:
                cod.append('E')
            elif di < 0 and da < 0 and dp < 0:
                cod.append('F')

        codogram = []
        for i in range(len(cod)-2):
            codogram.append(cod[i] + cod[i+1] + cod[i+2])

        st.write(f'Created {len(codogram)} codogram elements')

        letters = ['A', 'B', 'C', 'D', 'E', 'F']
        cod_sample = []
        for i in letters:
            for j in letters:
                for k in letters:
                    cod_sample.append(i + j + k)

        number_of_features = []
        for pattern in cod_sample:
            count = codogram.count(pattern)
            number_of_features.append(count)

        st.write(f'Feature vector created with {len(number_of_features)} elements')

        if not missing_files:  
            try:
                feature_columns = list(range(len(number_of_features)))
                x = pd.DataFrame([number_of_features], columns=feature_columns)

                with open('Gr_Boost_ECG_healthy.pkl', 'rb') as f:
                    model_of_health = pickle.load(f)
                
                model_of_health_pred = model_of_health.predict_proba(x)

                st.subheader('🏥 Results')

                if len(model_of_health_pred[0]) > 1 and model_of_health_pred[0][0] >= 0.9:
                    st.success(f'✅ Great! Low probability ({round(model_of_health_pred[0][1]*100, 2)}%) of the five analyzed diseases')
                else:
                    st.warning('🔍 Analyzing for potential diseases...')
                    
                    models = {
                        'vegetovascular dystonia': 'Gr_Boost_ECG_veget_dyst.pkl',
                        'coronary heart disease': 'Gr_Boost_ECG_ischemia.pkl', 
                        'stomach ulcer': 'Gr_Boost_ECG_ulcer.pkl',
                        'cholelithiasis': 'Gr_Boost_ECG_gallstone.pkl',
                        'nodular goiter of thyroid gland': 'Gr_Boost_ECG_nodular_thyroid_goiter.pkl'
                    }
                    
                    results = {}
                    for disease, model_file in models.items():
                        try:
                            with open(model_file, 'rb') as f:
                                model = pickle.load(f)
                            pred = model.predict_proba(x)
                            if len(pred[0]) > 0:
                                probability = round(pred[0][0] * 100, 2)
                                results[disease] = probability
                        except Exception as e:
                            st.warning(f"Could not load model for {disease}: {str(e)}")
                    
                    for disease, probability in results.items():
                        if probability > 50:
                            st.error(f'⚠️ **{disease.title()}**: {probability}%')
                        elif probability > 30:
                            st.warning(f'🟡 **{disease.title()}**: {probability}%')
                        else:
                            st.info(f'🟢 **{disease.title()}**: {probability}%')
            
            except Exception as e:
                st.error(f"Error during model prediction: {str(e)}")
                st.write("This might be due to model compatibility issues or incorrect feature dimensions")
        else:
            st.info("🎯 Demo mode: Models not available for real predictions")
            st.write("In demo mode, showing sample analysis results:")
            st.write("- Coronary heart disease: 15%")
            st.write("- Vegetovascular dystonia: 8%") 
            st.write("- Stomach ulcer: 5%")
            st.write("- Cholelithiasis: 3%")
            st.write("- Nodular goiter: 2%")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.write("Please check:")
        st.write("- CSV file format (should contain 'mV' column)")
        st.write("- File encoding (try UTF-8)")
        st.write("- Data quality and completeness")
        
        import traceback
        with st.expander("Technical details"):
            st.code(traceback.format_exc())

else:
    st.info("👆 Please upload an ECG CSV file to begin analysis")
    
    st.subheader("Expected file format:")
    example_data = pd.DataFrame({
        'mV': [0.1, 0.15, 0.2, 0.18, 0.12],
        'Seconds': [0.001, 0.002, 0.003, 0.004, 0.005]
    })
    st.write(example_data)
    st.write("- File should be CSV format")
    st.write("- Separator: semicolon (;) or comma (,)")
    st.write("- Required columns: 'mV' and 'Seconds'")
