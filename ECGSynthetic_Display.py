import math
import numpy as np
import plotly.graph_objs as go
import streamlit as st
import random


st.title("ECG Synthetic Project💓")

st.sidebar.title("Input Parameter Variabel")
st.sidebar.subheader("Mayer and RSA Parameter")
f1 = st.sidebar.number_input("f1 (Hz)", value=0.1)  
f2 = st.sidebar.number_input("f2 (Hz)", value=0.25) 

c1 = st.sidebar.number_input("c1", value=0.01)  
c2 = st.sidebar.number_input("c2", value=0.01)  

ratio = st.sidebar.number_input("ratio", value=0.5) 
Nrr = st.sidebar.number_input("Nrr", value=256)  

mode_ecg = st.selectbox("Choose the ECG Model", ["Normal","PVC"])

magsf = 1.7 / Nrr 
cycle = int(st.number_input("ECG Cycle", value=5))

st.sidebar.subheader("Noise Random Seed")
h_mean = st.sidebar.number_input("h mean", value=60)
offset = 60/h_mean
scalling = st.sidebar.number_input("Scalling", value=1)
st.sidebar.subheader("Parameter fecg")
fecg = st.sidebar.number_input("fecg", value=256)

st.sidebar.title("NOTE❕")
st.sidebar.write("All parameters have been set to auto to make it easier for users. Changing the parameters even slightly can affect the final synthetic ECG results.")
dt = 1/fecg
hr_fact = math.sqrt(h_mean/60) #modulation factor alpha
hr_fact2 = math.sqrt(hr_fact)
if mode_ecg == "PVC":
    st.subheader("Parameter for Normal ECG")
    col1, col2, col3 = st.columns(3)
    with col1:
        input_ti_p = st.number_input("Normal Teta P" ,value=-60)
        input_ti_q = st.number_input("Normal Teta Q" ,value=-15)
        input_ti_r = st.number_input("Normal Teta R" ,value=0)
        input_ti_s = st.number_input("Normal Teta S" ,value=15)
        input_ti_t = st.number_input("Normal Teta T" ,value=90)
    with col2:
        input_ai_p = st.number_input("Normal a P" ,value=1.2)
        input_ai_q = st.number_input("Normal a Q" ,value=-5)
        input_ai_r = st.number_input("Normal a R" ,value=30)
        input_ai_s = st.number_input("Normal a S" ,value=-7.5)
        input_ai_t = st.number_input("Normal a T" ,value=0.75)
    with col3:
        input_bi_p = st.number_input("Normal b P" ,value=0.25)
        input_bi_q = st.number_input("Normal b Q" ,value=0.1)
        input_bi_r = st.number_input("Normal b R" ,value=0.1)
        input_bi_s = st.number_input("Normal b S" ,value=0.1)
        input_bi_t = st.number_input("Normal b T" ,value=0.4)
    
    st.subheader("Parameter for Modification ECG")
    st.write("The parameters set on this display are for PVC (Premature Ventricular Contraction), if you want to change it to the ECG signal morphology of other cardiovascular diseases, the existing parameters can be changed.")
    colom1, colom2, colom3 = st.columns(3)
    with colom1:
        input_ti_p_modif = st.number_input("Modif Teta P" ,value=180)
        input_ti_q_modif = st.number_input("Modif Teta Q" ,value=-135)
        input_ti_r_modif = st.number_input("Modif Teta R" ,value=-120)
        input_ti_s_modif = st.number_input("Modif Teta S" ,value=-105)
        input_ti_t_modif = st.number_input("Modif Teta T" ,value=-30)
    with colom2:
        input_ai_p_modif = st.number_input("Modif a P" ,value=0)
        input_ai_q_modif = st.number_input("Modif a Q" ,value=5)
        input_ai_r_modif = st.number_input("Modif a R" ,value=5)
        input_ai_s_modif = st.number_input("Modif a S" ,value=5)
        input_ai_t_modif = st.number_input("Modif a T" ,value=-0.75)
    with colom3:
        input_bi_p_modif = st.number_input("Modif b P" ,value=0.25)
        input_bi_q_modif = st.number_input("Modif b Q" ,value=0.15)
        input_bi_r_modif = st.number_input("Modif b R" ,value=0.10)
        input_bi_s_modif = st.number_input("Modif b S" ,value=0.27)
        input_bi_t_modif = st.number_input("Modif b T" ,value=-0.40)
    #Parameter Teta
    ti_p = (input_ti_p*math.pi/180)*hr_fact2
    ti_q = (input_ti_q*math.pi/180)*hr_fact
    ti_r = (input_ti_r*math.pi/180)
    ti_s = (input_ti_s*math.pi/180)*hr_fact
    ti_t = (input_ti_t*math.pi/180)*hr_fact2
    ti = [ti_p, ti_q, ti_r, ti_s, ti_t]
    #Parameter a
    ai_p = input_ai_p
    ai_q = input_ai_q
    ai_r = input_ai_r
    ai_s = input_ai_s
    ai_t = input_ai_t
    ai = [ai_p, ai_q, ai_r, ai_s, ai_t]
    #Parameter b
    bi_p = input_bi_p*hr_fact2
    bi_q = input_bi_q*hr_fact2
    bi_r = input_bi_r*hr_fact2
    bi_s = input_bi_s*hr_fact2
    bi_t = input_bi_t*hr_fact2
    bi = [bi_p, bi_q, bi_r, bi_s, bi_t]
    #Modification variabel 
    #Parameter Teta
    ti_p_modif = (input_ti_p_modif*math.pi/180)*hr_fact2
    ti_q_modif = (input_ti_q_modif*math.pi/180)*hr_fact
    ti_r_modif = (input_ti_r_modif*math.pi/180)
    ti_s_modif = (input_ti_s_modif*math.pi/180)*hr_fact
    ti_t_modif = (input_ti_t_modif*math.pi/180)*hr_fact2
    ti_modif = [ti_p_modif, ti_q_modif, ti_r_modif, ti_s_modif, ti_t_modif]
    #Parameter a
    ai_p_modif =input_ai_p_modif
    ai_q_modif =input_ai_q_modif
    ai_r_modif =input_ai_r_modif
    ai_s_modif =input_ai_s_modif
    ai_t_modif =input_ai_t_modif
    ai_modif = [ai_p_modif, ai_q_modif, ai_r_modif, ai_s_modif, ai_t_modif]
    #Parameter b
    bi_p_modif = input_bi_p_modif  *hr_fact2
    bi_q_modif = input_bi_q_modif  *hr_fact2
    bi_r_modif = input_bi_r_modif  *hr_fact2
    bi_s_modif = input_bi_s_modif  *hr_fact2
    bi_t_modif = input_bi_t_modif  *hr_fact2
    bi_modif = [bi_p_modif, bi_q_modif, bi_r_modif, bi_s_modif, bi_t_modif]
elif mode_ecg== "Normal":
    #Parameter Teta
    ti_p = (-60*math.pi/180)*hr_fact2
    ti_q = (-15*math.pi/180)*hr_fact
    ti_r = (0  *math.pi/180)
    ti_s = (15 *math.pi/180)*hr_fact
    ti_t = (90 *math.pi/180)*hr_fact2
    ti = [ti_p, ti_q, ti_r, ti_s, ti_t]
    #Parameter a
    ai_p = 1.2
    ai_q = -5
    ai_r = 30
    ai_s = -7.5
    ai_t = 0.75
    ai = [ai_p, ai_q, ai_r, ai_s, ai_t]
    #Parameter b
    bi_p = 0.25*hr_fact2
    bi_q = 0.1*hr_fact2
    bi_r = 0.1*hr_fact2
    bi_s = 0.1*hr_fact2
    bi_t = 0.4*hr_fact2
    bi = [bi_p, bi_q, bi_r, bi_s, bi_t]
    #Modification variabel 
    #Parameter Teta
    ti_p_modif = (-60*math.pi/180)*hr_fact2
    ti_q_modif = (-15*math.pi/180)*hr_fact
    ti_r_modif = (0*math.pi/180)
    ti_s_modif = (15*math.pi/180)*hr_fact
    ti_t_modif = (90*math.pi/3)*hr_fact2
    ti_modif = [ti_p_modif, ti_q_modif, ti_r_modif, ti_s_modif, ti_t_modif]
    #Parameter a
    ai_p_modif =1.2
    ai_q_modif =-5
    ai_r_modif =30
    ai_s_modif =-7.5
    ai_t_modif =0.75
    ai_modif = [ai_p_modif, ai_q_modif, ai_r_modif, ai_s_modif, ai_t_modif]
    #Parameter b
    bi_p_modif = 0.25  *hr_fact2
    bi_q_modif = 0.1  *hr_fact2
    bi_r_modif = 0.1  *hr_fact2
    bi_s_modif = 0.1  *hr_fact2 
    bi_t_modif = 0.4  *hr_fact2
    bi_modif = [bi_p_modif, bi_q_modif, bi_r_modif, bi_s_modif, bi_t_modif]

    
if st.button("Start Simulation"):
    sf = []

    for i in range(0, Nrr):
        f = i * 1 / Nrr
        sf1 = ratio * magsf * math.exp(
            -((f - f1)**2) / (2 * c1**2)
        ) / math.sqrt(2 * math.pi * (c1**2))
        sf2 = magsf * math.exp(
            -((f - f2) ** 2) / (2 * (c2**2))
        ) / math.sqrt(2 * math.pi * (c2**2))
        sf.append(sf1 + sf2)

    # Generate the frequency axis for plotting
    f = np.linspace(0, 1, Nrr)

    #Copy mayer and RSA to variabel that can be plot
    just_mayer = sf.copy()

    #Mirroring Data
    j = 0
    sf_mirror = np.zeros(len(sf))
    for i in range(Nrr // 2, Nrr):
        sf[i] = sf[Nrr - i - 1]
        j += 1

    #Sqrt data 
    for i in range(Nrr):
        sf[i] = math.sqrt(sf[i])

    for i in range (Nrr):
        sf_mirror[i]=(sf[i])


    plotly_mayer_rsa_mirror = go.Figure()
    plotly_mayer_rsa_mirror.add_trace(go.Scatter(x=f, y=just_mayer, mode='lines', name="Mayer and RSA",line=dict(color='red')))
    plotly_mayer_rsa_mirror.add_trace(go.Scatter(x=f, y=sf, mode='lines', name="Mirroring and Sqrt",line=dict(color='blue')))
    # Customize the layout
    plotly_mayer_rsa_mirror.update_layout(
        xaxis_title="Frequency [Hz]",
        yaxis_title="Power",
        title="Mayer and RSA",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )
    st.plotly_chart(plotly_mayer_rsa_mirror)

    # IDFT
    sf_real_idft = np.zeros(Nrr)
    sf_imaj_idft = np.zeros(Nrr)
    MagIDFT = np.zeros(Nrr) 

    #Random signal 
    sfre= np.zeros(Nrr)
    sftm= np.zeros(Nrr)

    for n in range (Nrr):
        sfre[n] = sf[n]*math.cos(2*math.pi*(random.uniform(0.0, 2*math.pi)))
        sftm[n] = sf[n]*math.sin(2*math.pi*(random.uniform(0.0, 2*math.pi)))

    #IDFT
    for n in range(Nrr):
        for k in range(Nrr):
            sf_real_idft[n] += sfre[k]*math.cos(2*math.pi*k*n/Nrr)
            sf_imaj_idft[n] += sftm[k]*math.sin(2*math.pi*k*n/Nrr)
        MagIDFT[n] = sf_real_idft[n]/Nrr + sf_imaj_idft[n]/Nrr

    #Looping 
    k =  np.arange (0, Nrr, 1, dtype=int)
    n = np.arange (0, Nrr, 1, dtype=int)


    plotly_random_signal = go.Figure()

    plotly_random_signal.add_trace(go.Scatter(x=f, y=sfre, mode='lines', name= "Real Component",line=dict(color='green')))
    plotly_random_signal.add_trace(go.Scatter(x=f, y=sftm, mode='lines', name= "Imajiner Component",line=dict(color='red')))
    # Customize the layout
    plotly_random_signal.update_layout(
        xaxis_title="Frequency [Hz]",
        yaxis_title="Power",
        title="Mirroring and sqrt",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )
    st.plotly_chart(plotly_random_signal)
    ################################3
    #output RR

    rr_tachogram = []

    for i in range (Nrr):
        rr_tachogram.append((MagIDFT[i]*scalling+offset))
    
    plotly_IDFT_scalling = go.Figure()
    plotly_IDFT_scalling.add_trace(go.Scatter(x=k, y=MagIDFT, mode='lines', name="Before",line=dict(color='yellow')))
    plotly_IDFT_scalling.add_trace(go.Scatter(x=k, y=rr_tachogram, mode='lines', name="After",line=dict(color='red')))
    # Customize the layout
    plotly_IDFT_scalling.update_layout(
        xaxis_title="Frequency [Hz]",
        yaxis_title="Power",
        title="RR Tachogram",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )
    st.plotly_chart(plotly_IDFT_scalling)

    def angfreq(in1):
        i = int(math.floor(in1 / fecg))
        if rr_tachogram[i]==0:
            rr_tachogram[i] = 0.000000000000001
        return ((2*np.pi)/rr_tachogram[int(np.floor(in1/fecg))])
    # ---- fungsi modulus untuk dimasukkan ke rumus z dot ---- #
    def modulus(in1, in2):
        while in1 >= in2:
            in1 = in1 - in2
        return in1
    def ddt(t0, x0, y0, z0, trig, indikator_pvc):
        a0 = 1- math.sqrt((x0*x0)+(y0*y0)) #rumus untuk alpha di papaer
        if trig == 1:
            return a0*x0 - (angfreq(t0)*y0)
        elif trig == 2:
            return a0*y0 + (angfreq(t0)*x0)
        elif trig == 3:
            temp = 0
            zbase = 0.005*math.sin(2*math.pi*f2*t0) #baseline wander
            te = np.arctan2(y0,x0) 
            for i in range (5):
                if indikator_pvc == 0:
                    det = ((te - ti[i])%(2 * math.pi))-math.pi # det = delta teta
                    det2 = det * det
                    temp = temp - (ai[i] * det * math.exp((-0.5 * det2) / (bi[i] * bi[i])))
                else:
                    det = ((te - ti_modif[i])%(2 * math.pi))-math.pi # det = delta teta
                    det2 = det * det
                    temp = temp - (ai_modif[i] * det * math.exp((-0.5 * det2) / (bi_modif[i] * bi_modif[i])))
            temp -= (z0 - zbase)
            return temp
    #Set All the morphological Parameter 
    # Inisialisasi kondisi awal
    x = np.zeros(Nrr)
    y = np.zeros(Nrr)
    z = np.zeros(Nrr)
    x[0] = 0.1
    y[0] = 0
    z[0] = 0.001
    timev = 0.0
    
    yt = np.zeros(Nrr)
    xt = np.zeros(Nrr)
    zt = np.zeros(Nrr)
    output = []
    if mode_ecg == "PVC":
        pvc_cycle = random.randint(0,cycle-1)
    else:
        pvc_cycle = cycle+2 #untuk skip parameter 
    # Inisialisasi series untuk plotting
    for j in range (cycle):
        for i in range(Nrr):
            if j==pvc_cycle: 
                xt[i] = x[0]
                yt[i] = y[0]
                zt[i] = z[0]
                # Konstanta 1
                k1x = ddt(timev, x[0], y[0], z[0], 1, 1)
                k1y = ddt(timev, x[0], y[0], z[0], 2, 1)
                k1z = ddt(timev, x[0], y[0], z[0], 3, 1)
                # Konstanta 2
                k2x = ddt(timev + (dt * 0.5), x[0] + (dt * 0.5 * k1x), y[0] + (dt * 0.5 * k1y), z[0] + (dt * 0.5 * k1z), 1, 1)
                k2y = ddt(timev + (dt * 0.5), x[0] + (dt * 0.5 * k1x), y[0] + (dt * 0.5 * k1y), z[0] + (dt * 0.5 * k1z), 2, 1)
                k2z = ddt(timev + (dt * 0.5), x[0] + (dt * 0.5 * k1x), y[0] + (dt * 0.5 * k1y), z[0] + (dt * 0.5 * k1z), 3, 1)
                # Konstanta 3
                k3x = ddt(timev + (dt * 0.5), x[0] + (dt * 0.5 * k2x), y[0] + (dt * 0.5 * k2y), z[0] + (dt * 0.5 * k1z), 1, 1)
                k3y = ddt(timev + (dt * 0.5), x[0] + (dt * 0.5 * k2x), y[0] + (dt * 0.5 * k2y), z[0] + (dt * 0.5 * k1z), 2, 1)
                k3z = ddt(timev + (dt * 0.5), x[0] + (dt * 0.5 * k2x), y[0] + (dt * 0.5 * k2y), z[0] + (dt * 0.5 * k1z), 3, 1)
                # Konstanta 4
                k4x = ddt(timev + dt, x[0] + (k3x * dt), y[0] + (k3y * dt), z[0] + (k3z * dt), 1, 1)
                k4y = ddt(timev + dt, x[0] + (k3x * dt), y[0] + (k3y * dt), z[0] + (k3z * dt), 2, 1)
                k4z = ddt(timev + dt, x[0] + (k3x * dt), y[0] + (k3y * dt), z[0] + (k3z * dt), 3, 1)
                # Hasil Runge-Kutta orde 4
                x[0] += (dt / 6) * (k1x + (2 * k2x) + (2 * k3x) + k4x)
                y[0] += (dt / 6) * (k1y + (2 * k2y) + (2 * k3y) + k4y)
                z[0] += (dt / 6) * (k1z + (2 * k2z) + (2 * k3z) + k4z)
                timev = timev + dt
                output.append(zt[i])
            else: 
                xt[i] = x[0]
                yt[i] = y[0]
                zt[i] = z[0]
                # Konstanta 1
                k1x = ddt(timev, x[0], y[0], z[0], 1, 0)
                k1y = ddt(timev, x[0], y[0], z[0], 2, 0)
                k1z = ddt(timev, x[0], y[0], z[0], 3, 0)
                # Konstanta 2
                k2x = ddt(timev + (dt * 0.5), x[0] + (dt * 0.5 * k1x), y[0] + (dt * 0.5 * k1y), z[0] + (dt * 0.5 * k1z), 1, 0)
                k2y = ddt(timev + (dt * 0.5), x[0] + (dt * 0.5 * k1x), y[0] + (dt * 0.5 * k1y), z[0] + (dt * 0.5 * k1z), 2, 0)
                k2z = ddt(timev + (dt * 0.5), x[0] + (dt * 0.5 * k1x), y[0] + (dt * 0.5 * k1y), z[0] + (dt * 0.5 * k1z), 3, 0)
                # Konstanta 3
                k3x = ddt(timev + (dt * 0.5), x[0] + (dt * 0.5 * k2x), y[0] + (dt * 0.5 * k2y), z[0] + (dt * 0.5 * k1z), 1, 0)
                k3y = ddt(timev + (dt * 0.5), x[0] + (dt * 0.5 * k2x), y[0] + (dt * 0.5 * k2y), z[0] + (dt * 0.5 * k1z), 2, 0)
                k3z = ddt(timev + (dt * 0.5), x[0] + (dt * 0.5 * k2x), y[0] + (dt * 0.5 * k2y), z[0] + (dt * 0.5 * k1z), 3, 0)
                # Konstanta 4
                k4x = ddt(timev + dt, x[0] + (k3x * dt), y[0] + (k3y * dt), z[0] + (k3z * dt), 1, 0)
                k4y = ddt(timev + dt, x[0] + (k3x * dt), y[0] + (k3y * dt), z[0] + (k3z * dt), 2, 0)
                k4z = ddt(timev + dt, x[0] + (k3x * dt), y[0] + (k3y * dt), z[0] + (k3z * dt), 3, 0)
                # Hasil Runge-Kutta orde 4
                x[0] += (dt / 6) * (k1x + (2 * k2x) + (2 * k3x) + k4x)
                y[0] += (dt / 6) * (k1y + (2 * k2y) + (2 * k3y) + k4y)
                z[0] += (dt / 6) * (k1z + (2 * k2z) + (2 * k3z) + k4z)
                timev = timev + dt
                output.append(zt[i])

    plotly_output = go.Figure()
    i = np.arange (0, len(output), 1)
    plotly_output.add_trace(go.Scatter(x=i/fecg, y=output, mode='lines', line=dict(color='green')))
    # Customize the layout
    plotly_output.update_layout(
        xaxis_title="Time [Second]",
        yaxis_title="Voltage [Volts]",
        title="ECG Synthetic",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )
    st.plotly_chart(plotly_output)
