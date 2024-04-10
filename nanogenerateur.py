import time

import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

sigma = -9.1673 * 10 ** (-5)  # C/m²
d = 100 * 10 ** (-6)  # m
epsr = 2.2
epszero = 8.85 * 10 ** (-11)
eps = epszero * epsr
Rload = 10 ** 6  # ohm
r0 = 0.0015  # m
L = 0.003  # m
Selec = 0.000003  # m²
Rm = 10 ** 6  # ohm
v = 1  # m/s

dt = 10 ** (-4)  # S
duree = 0.25  # s

T = [dt * i for i in range(int(duree / dt))]
n = len(T)

# Loi du rayon de la goutte
t1 = 1.5 * 10 ** -3
t2 = 3.4 * 10 ** -3
t3 = 6 * 10 ** -3

st.title("Etude d'un générateur tribo-électrique")


def calculRayon(t):
    if t <= t1:
        return r0 * t / t1
    elif t1 < t <= t2:
        return r0 + r0 * (t - t1) / (t2 - t1)
    elif t2 < t <= t3:
        return 2 * r0 - r0 * (t - t2) / (t3 - t2)
    return r0


def Sgoutte(t):
    if tcontact <= t <= tseparation:
        return np.pi * calculRayon(t) ** 2
    return Selec


tab1, tab2 = st.tabs(["Résolution euler", "Animation goutte"])
with tab1:
    tcontact, tseparation = st.slider(value=(0.03, 0.2), min_value=0.0, max_value=0.25, step=0.01, label="Temps contact entre la goutte et l'électrode")


    def Sgoutte2(t, tcontact, tseparation):
        if tcontact <= t <= tseparation:
            return (261.53 * (t ** 3) - 169.02 * (t ** 2) + 26.187 * t + 0.2658) * 10 ** -4
        return Selec


    @st.cache_data
    def euler(tcontact, tseparation):
        I = [0] * n
        U = [0] * n
        P = [0] * n
        C = [Sgoutte2(T[i], tcontact, tseparation) * eps / d for i in range(n)]
        Velec = (sigma * d) / eps
        Q = [-Velec * C[0]] * n

        for i in range(n - 1):

            Q[i + 1] = (Q[i] - (dt / Rm) * Velec) / (1 + dt / (Rm * C[i + 1]))
            if tcontact+0.001> T[i] > tcontact-0.001:
                print(T[i], Q[i])
            I[i] = (Q[i + 1] - Q[i]) / dt
            U[i] = Velec + Q[i] / C[i]
            P[i] = U[i] * I[i]

        plot = sns.lineplot(x=T, y=U, linewidth=2.5)
        col1, col2 = st.columns(2)
        col1.subheader("Tension")
        col1.pyplot(plot.get_figure())
        plt.clf()
        plot = sns.lineplot(x=T, y=[Sgoutte2(T[i], tcontact, tseparation) for i in range(n)], linewidth=2.5)
        col2.subheader("Surface active")
        col2.pyplot(plot.get_figure())
        plt.clf()

        col1, col2 = st.columns(2)
        col1.subheader("Intensité")
        plot = sns.lineplot(x=T, y=I, linewidth=2.5)
        col1.pyplot(plot.get_figure())
        plt.clf()

        col2.subheader("Q")
        plot = sns.lineplot(x=T, y=Q, linewidth=2.5)
        col2.pyplot(plot.get_figure())
        plt.clf()

        print("Max", max(U), min(U))



    euler(tcontact, tseparation)

with tab2:
    st.subheader("Evolution de la goutte en fonction du temps")
    play = st.button("Play animation")

    placeholder = st.empty()
    # t_surface = st.slider(value=0.0, min_value=0.0, max_value=T[-1], step=0.01, label="Temps")
    t_surface = 0
    # Boucle pour mettre à jour le contenu toutes les 2 secondes
    if play:
        for i in range(20):  # Modifier 10 par n pour un nombre n d'itérations
            with placeholder.container():
                st.write(f"{i * 0.01/2} secondes")
                t_surface = i * 0.01/2

                r_t = np.sqrt((261.53 * (t_surface ** 3) - 169.02 * (t_surface ** 2) + 26.187 * t_surface + 0.2658) * 10 ** -4 / np.pi)
                plot = sns.scatterplot(x=[r_t * np.cos(2 * np.pi * i / 100) for i in range(100)], y=[r_t * np.sin(2 * np.pi * i / 100) for i in range(100)], linewidth=2.5)

                fig = go.Figure()

                # Create scatter trace of text labels
                fig.add_trace(go.Scatter(
                    x=[0],
                    y=[0],
                    text=["Goutte", ],
                    mode="text",
                ))
                # Set axes properties
                fig.update_xaxes(range=[-0.01, 0.01], zeroline=False)
                fig.update_yaxes(range=[-0.01, 0.01])
                fig.add_shape(type="circle",
                              xref="x", yref="y",
                              x0=-r_t, y0=-r_t, x1=r_t, y1=r_t,
                              line_color="LightSeaGreen",
                              )
                fig.update_layout(
                    autosize=False,
                    width=600,
                    height=600,
                )

                st.plotly_chart(fig)
                plt.clf()

                time.sleep(1/2)  # Attend 2 secondes avant la prochaine itératio
