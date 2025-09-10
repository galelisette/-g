import pandas as pd
import pickle
import math
import streamlit as st
import os

st.write("""# Recognition of diseases by ECG
**Based on the works of Gale Lisette**""")

# Функция для проверки существования файлов моделей
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

# Проверяем наличие файлов моделей
missing_files = check_model_files()
if missing_files:
    st.error(f"Отсутствуют файлы моделей: {', '.join(missing_files)}")
    st.write("Пожалуйста, загрузите все необходимые файлы .pkl в репозиторий")
    st.stop()

uploaded_file = st.file_uploader("Select ECG file *.csv", type="csv")
if uploaded_file is not None:
    try:
        df_ecg = pd.read_csv(uploaded_file, sep=";")
        
        # Проверяем наличие необходимых столбцов
        if 'mV' not in df_ecg.columns or 'Seconds' not in df_ecg.columns:
            st.error("CSV файл должен содержать столбцы 'mV' и 'Seconds'")
            st.stop()

        # Creating a cutoff threshold (mean curve, mV)
        col1, col2 = st.columns([1, 4])
        with col1:
            st.subheader("Data of ECG")
            st.write(df_ecg)
        with col2:
            st.subheader("Chart of ECG")
            # Ограничиваем количество точек для отображения
            max_points = min(3000, len(df_ecg))
            st.line_chart(data=df_ecg.loc[:max_points, 'mV'], use_container_width=True)
        
        st.write('Analysing of ECG')
        last_point = len(df_ecg['mV'])
        vol_of_mean = int(last_point//330+1)
        last_mean_point = int(last_point-vol_of_mean)
        last_mean = df_ecg.loc[last_mean_point:last_point, 'mV'].mean()
        shift = 0.3
        last_threshold = last_mean + shift
        df_ecg['Threshold'] = 0.0

        # ИСПРАВЛЕНО: Правильная инициализация списка
        df_ecg_threshold = [0.0] * (last_mean_point + 1)
        my_bar = st.progress(0)
        
        for i in range(last_mean_point+1):
            end_idx = min(vol_of_mean+i, last_point)  # Защита от выхода за границы
            df_ecg_threshold[i] = df_ecg.loc[i:end_idx-1,'mV'].mean()+shift
            my_bar.progress(min(100, round(100*(i+1)/(last_mean_point+1))))
        
        df_ecg.loc[:last_mean_point,'Threshold'] = df_ecg_threshold
        df_ecg.loc[last_mean_point:last_point,'Threshold'] = last_threshold

        # Search for amplitude
        st.write('Preparing of Codogram')
        amplitude = []
        second_of_amplitude = []
        my_bar = st.progress(0)
        
        for i in range(last_point-2):
            if df_ecg.loc[i+1,'mV']>=df_ecg.loc[i+1,'Threshold']:
                if df_ecg.loc[i+1,'mV']-df_ecg.loc[i,'mV']>=0 and df_ecg.loc[i+1,'mV']-df_ecg.loc[i+2,'mV']>0:
                    amplitude.append(df_ecg.loc[i+1,'mV'])
                    second_of_amplitude.append(df_ecg.loc[i+1,'Seconds'])
            my_bar.progress(min(100, round(100*(i+1)/(last_point-2))))

        # ИСПРАВЛЕНО: Безопасное обрезание списков
        if len(amplitude) > 601:
            amplitude = amplitude[:601]
            second_of_amplitude = second_of_amplitude[:601]

        # Проверяем, что есть достаточно данных для анализа
        if len(amplitude) < 3:
            st.error("Недостаточно данных для анализа. Попробуйте другой файл ECG.")
            st.stop()

        # Calculation of intervals and phase
        interval = []
        phase = []
        
        for i in range(len(second_of_amplitude)-1):
            interval.append(second_of_amplitude[i+1]-second_of_amplitude[i])
        
        for i in range(len(amplitude)-1):
            if interval[i] != 0:  # Защита от деления на ноль
                phase.append(math.atan(amplitude[i]/interval[i]))
            else:
                phase.append(0.0)

        # Calculation of delta of intervals, amplitudes and phases
        delta_interval = []
        delta_amplitude = []
        delta_phase = []
        
        for i in range(len(interval)-1):
            delta_interval.append(round(interval[i+1]-interval[i],3))
            delta_amplitude.append(round(amplitude[i+1]-amplitude[i],3))
            delta_phase.append(round(phase[i+1]-phase[i],6))

        if len(delta_interval) == 0:
            st.error("Недостаточно данных для создания кодограммы.")
            st.stop()

        deltas = pd.DataFrame({
            'delta of intervals': delta_interval, 
            'delta of amplitudes': delta_amplitude, 
            'delta of phases': delta_phase
        })

        # Create codogram
        cod = []
        for i in range(len(deltas)):
            if delta_interval[i]>=0 and delta_amplitude[i]>=0 and delta_phase[i]>=0:
                cod.append('A')
            elif delta_interval[i]<0 and delta_amplitude[i]<0 and delta_phase[i]>=0:
                cod.append('B')
            elif delta_interval[i]<0 and delta_amplitude[i]>=0 and delta_phase[i]>=0:
                cod.append('C')
            elif delta_interval[i]>=0 and delta_amplitude[i]<0 and delta_phase[i]<0:
                cod.append('D')
            elif delta_interval[i]>=0 and delta_amplitude[i]>=0 and delta_phase[i]<0:
                cod.append('E')
            elif delta_interval[i]<0 and delta_amplitude[i]<0 and delta_phase[i]<0:
                cod.append('F')

        codogram = []
        for i in range(len(cod)-2):
            codogram.append(cod[i]+cod[i+1]+cod[i+2])

        cod_sample = ['AAA','AAB','AAC','AAD','AAE','AAF','ABA','ABB','ABC','ABD','ABE','ABF','ACA','ACB','ACC','ACD','ACE','ACF','ADA','ADB','ADC','ADD','ADE','ADF','AEA','AEB','AEC','AED','AEE','AEF','AFA','AFB','AFC','AFD','AFE','AFF','BAA','BAB','BAC','BAD','BAE','BAF','BBA','BBB','BBC','BBD','BBE','BBF','BCA','BCB','BCC','BCD','BCE','BCF','BDA','BDB','BDC','BDD','BDE','BDF','BEA','BEB','BEC','BED','BEE','BEF','BFA','BFB','BFC','BFD','BFE','BFF','CAA','CAB','CAC','CAD','CAE','CAF','CBA','CBB','CBC','CBD','CBE','CBF','CCA','CCB','CCC','CCD','CCE','CCF','CDA','CDB','CDC','CDD','CDE','CDF','CEA','CEB','CEC','CED','CEE','CEF','CFA','CFB','CFC','CFD','CFE','CFF','DAA','DAB','DAC','DAD','DAE','DAF','DBA','DBB','DBC','DBD','DBE','DBF','DCA','DCB','DCC','DCD','DCE','DCF','DDA','DDB','DDC','DDD','DDE','DDF','DEA','DEB','DEC','DED','DEE','DEF','DFA','DFB','DFC','DFD','DFE','DFF','EAA','EAB','EAC','EAD','EAE','EAF','EBA','EBB','EBC','EBD','EBE','EBF','ECA','ECB','ECC','ECD','ECE','ECF','EDA','EDB','EDC','EDD','EDE','EDF','EEA','EEB','EEC','EED','EEE','EEF','EFA','EFB','EFC','EFD','EFE','EFF','FAA','FAB','FAC','FAD','FAE','FAF','FBA','FBB','FBC','FBD','FBE','FBF','FCA','FCB','FCC','FCD','FCE','FCF','FDA','FDB','FDC','FDD','FDE','FDF','FEA','FEB','FEC','FED','FEE','FEF','FFA','FFB','FFC','FFD','FFE','FFF']

        number_of_features = []
        for i in cod_sample:
            number_of_features.append(codogram.count(i))

        # Diagnostics
        x = pd.DataFrame(data=[number_of_features], columns=list(range(2, 218)))

        try:
            model_of_health = pickle.load(open('Gr_Boost_ECG_healthy.pkl','rb'))
            model_of_health_pred = model_of_health.predict_proba(x)

            st.subheader('Results')

            if model_of_health_pred[0][0]>=0.9:
                st.write('Fine! The probability (', round(model_of_health_pred[0][1]*100, 2), '%) of one of five diseases is low')
            else:
                model_of_veget_dyst = pickle.load(open('Gr_Boost_ECG_veget_dyst.pkl','rb'))
                model_of_veget_dyst_pred = model_of_veget_dyst.predict_proba(x)
                model_of_ischemia = pickle.load(open('Gr_Boost_ECG_ischemia.pkl','rb'))
                model_of_ischemia_pred = model_of_ischemia.predict_proba(x)
                model_of_ulcer = pickle.load(open('Gr_Boost_ECG_ulcer.pkl','rb'))
                model_of_ulcer_pred = model_of_ulcer.predict_proba(x)
                model_of_gallstone = pickle.load(open('Gr_Boost_ECG_gallstone.pkl','rb'))
                model_of_gallstone_pred = model_of_gallstone.predict_proba(x)
                model_of_nodular_thyroid_goiter = pickle.load(open('Gr_Boost_ECG_nodular_thyroid_goiter.pkl','rb'))
                model_of_nodular_thyroid_goiter_pred = model_of_nodular_thyroid_goiter.predict_proba(x)
                
                st.write('Probability of coronary heart disease:', round(model_of_ischemia_pred[0][0]*100, 2),'%')
                st.write('Probability of vegetovascular dystonia:', round(model_of_veget_dyst_pred[0][0]*100, 2),'%')
                st.write('Probability of stomach ulcer:', round(model_of_ulcer_pred[0][0]*100, 2),'%')
                st.write('Probability of cholelithiasis:', round(model_of_gallstone_pred[0][0]*100, 2),'%')
                st.write('Probability of nodular goiter of the thyroid gland:', round(model_of_nodular_thyroid_goiter_pred[0][0]*100, 2),'%')
        
        except Exception as e:
            st.error(f"Ошибка при загрузке моделей: {str(e)}")
            st.write("Убедитесь, что все файлы .pkl находятся в корневой папке репозитория")

    except Exception as e:
        st.error(f"Ошибка при обработке файла: {str(e)}")
        st.write("Проверьте формат CSV файла. Он должен содержать столбцы 'mV' и 'Seconds' с разделителем ';'")

else:
    st.write("Load ECG file")
