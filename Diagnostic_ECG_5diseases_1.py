import pandas as pd
import pickle
import math
import streamlit as st

st.write("""# Recognition of diseases by ECG
**Based on the works of Gale Lisette**""")

# st.subheader("Ввод данных")
# df_ecg = pd.read_csv('C:\\Users\Андрей\Desktop\DS\My_Projects\Datasets\ЭКГ\Датасет_2\Полн_Voron-2014-task-ekg-data\samples.csv', sep=";")

uploaded_file = st.file_uploader("Select ECG file *.csv", type="csv")
if uploaded_file != None:
    df_ecg = pd.read_csv(uploaded_file, sep=";")

    # Creating a cutoff threshold (mean curve, mV)

    col1, col2 = st.columns([1, 4])
    with col1:
        st.subheader("Data of ECG")
        st.write(df_ecg)
    with col2:
        st.subheader("Chart of ECG")
        st.line_chart(data=df_ecg.loc[:3000, 'mV'], width=0, height=0, use_container_width=True)
    st.write('Analysing of ECG')
    last_point = len(df_ecg['mV'])
    vol_of_mean = int(last_point//330+1)
    last_mean_point = int(last_point-vol_of_mean)
    last_mean = df_ecg.loc[last_mean_point:last_point, 'mV'].mean()
    shift = 0.3
    last_threshold = last_mean + shift
    df_ecg['Threshold'] = 0
    shift = 0.3

    df_ecg_threshold = []
    my_bar = st.progress(0)
    i = 0
    for i in range(last_mean_point+1):
        df_ecg_threshold[i:] = [df_ecg.loc[i:vol_of_mean+i-1,'mV'].mean()+shift]
        my_bar.progress(round(100*(i+1)/(last_mean_point+1)))
    df_ecg.loc[:last_mean_point,'Threshold'] = df_ecg_threshold
    df_ecg.loc[last_mean_point:last_point,'Threshold'] = last_threshold

    # Search for amplitude

    st.write('Prepearing of Codogram')
    amplitude = []
    second_of_amplitude = []
    my_bar = st.progress(0)
    i = 0
    for i in range(last_point-2):
        if df_ecg.loc[i+1,'mV']>=df_ecg.loc[i+1,'Threshold']:
            if df_ecg.loc[i+1,'mV']-df_ecg.loc[i,'mV']>=0 and df_ecg.loc[i+1,'mV']-df_ecg.loc[i+2,'mV']>0:
                amplitude.append(df_ecg.loc[i+1,'mV'])
                second_of_amplitude.append(df_ecg.loc[i+1,'Seconds'])
        my_bar.progress(round(100*(i+1)/(last_point-2)))
    del amplitude[601:]
    del second_of_amplitude[601:]

    # Calculation of intervals and phase

    interval = []
    phase = []
    i = 0
    for i in range(len(second_of_amplitude)-1):
        interval.append(second_of_amplitude[i+1]-second_of_amplitude[i])
    i = 0
    for i in range(len(amplitude)-1):
        phase.append(math.atan(amplitude[i]/interval[i]))

    # Сalculation of delta of intervals, amplitudes and phases

    delta_interval = []
    delta_amplitude = []
    delta_phase = []
    i = 0
    for i in range(len(interval)-1):
        delta_interval.append(round(interval[i+1]-interval[i],3))
        delta_amplitude.append(round(amplitude[i+1]-amplitude[i],3))
        delta_phase.append(round(phase[i+1]-phase[i],6))
    deltas = pd.DataFrame({'delta of intervals':delta_interval, 'delta of amplitudes':delta_amplitude, 'delta of phases':delta_phase})

    # Create of codogram

    cod = []
    i = 0
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
    i = 0
    for i in range(len(cod)-2):
        codogram.append(cod[i]+cod[i+1]+cod[i+2])
    cod_sample = ['AAA','AAB','AAC','AAD','AAE','AAF','ABA','ABB','ABC','ABD','ABE','ABF','ACA','ACB','ACC','ACD','ACE','ACF','ADA','ADB','ADC','ADD','ADE','ADF','AEA','AEB','AEC','AED','AEE','AEF','AFA','AFB','AFC','AFD','AFE','AFF','BAA','BAB','BAC','BAD','BAE','BAF','BBA','BBB','BBC','BBD','BBE','BBF','BCA','BCB','BCC','BCD','BCE','BCF','BDA','BDB','BDC','BDD','BDE','BDF','BEA','BEB','BEC','BED','BEE','BEF','BFA','BFB','BFC','BFD','BFE','BFF','CAA','CAB','CAC','CAD','CAE','CAF','CBA','CBB','CBC','CBD','CBE','CBF','CCA','CCB','CCC','CCD','CCE','CCF','CDA','CDB','CDC','CDD','CDE','CDF','CEA','CEB','CEC','CED','CEE','CEF','CFA','CFB','CFC','CFD','CFE','CFF','DAA','DAB','DAC','DAD','DAE','DAF','DBA','DBB','DBC','DBD','DBE','DBF','DCA','DCB','DCC','DCD','DCE','DCF','DDA','DDB','DDC','DDD','DDE','DDF','DEA','DEB','DEC','DED','DEE','DEF','DFA','DFB','DFC','DFD','DFE','DFF','EAA','EAB','EAC','EAD','EAE','EAF','EBA','EBB','EBC','EBD','EBE','EBF','ECA','ECB','ECC','ECD','ECE','ECF','EDA','EDB','EDC','EDD','EDE','EDF','EEA','EEB','EEC','EED','EEE','EEF','EFA','EFB','EFC','EFD','EFE','EFF','FAA','FAB','FAC','FAD','FAE','FAF','FBA','FBB','FBC','FBD','FBE','FBF','FCA','FCB','FCC','FCD','FCE','FCF','FDA','FDB','FDC','FDD','FDE','FDF','FEA','FEB','FEC','FED','FEE','FEF','FFA','FFB','FFC','FFD','FFE','FFF']

    number_of_features = []
    i = 0
    k = 0
    for i in cod_sample:
        number_of_features.append(codogram.count(i))

    # Diagnostics

    x = pd.DataFrame(data=[number_of_features], columns=[list(range(2, 218, 1))])
    model_of_helth = pickle.load(open('Gr_Boost_ECG_healthy.pkl','rb'))
    model_of_helth_pred = model_of_helth.predict_proba(x)

    st.subheader('Rezults')

    if model_of_helth_pred[0][0]>=0.9 :
        st.write('Fine! The probability (', round(model_of_helth_pred[0][1]*100, 2), '%) of one of five diseases is low')
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
        st.write('Probability of coronary heart disease', round(model_of_ischemia_pred[0][0]*100, 2),'%')
        st.write('Probability of vegetovascular dystonia', round(model_of_veget_dyst_pred[0][0]*100, 2),'%')
        st.write('Probability of stomach ulcer', round(model_of_ulcer_pred[0][0]*100, 2),'%')
        st.write('Probability of cholelithiasis', round(model_of_gallstone_pred[0][0]*100, 2),'%')
        st.write('Probability of nodular goiter of the thyroid gland', round(model_of_nodular_thyroid_goiter_pred[0][0]*100, 2),'%')
else:
    st.write("Load ECG file")