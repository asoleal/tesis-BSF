#_____________________________________________________________________________________________________________________
#
# Modelo para estimar la producción de CO_{2} del proceso de bioconversión de la Mosca Soldado Negra Para el estimado J.Leal con red neuronal multicapa 
# para la estimación del B_max y solución numérica de las ecuaciones diferenciales con métodos de Runge-Kutta y Euler
#
# Autores: Luis Fernando Mejia Rodriguez y J. Leal_
# ___________________________________________________________________________________________________________________


# --------------------------
#Se importan las librerías necesarias para redes neuronales, graficas y computación cientifica
# --------------------------

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --------------------------
# 1. Red neuronal para B_max
# --------------------------
# Esta función define el comportamiento de B_max en función del tiempo t.
# Si t es menor que t_p_min, devuelve B_max_0; de lo contrario, calcula B_max_0 menos un decremento lineal basado en rho.
# Se asegura de que t sea un array para manejar tanto entradas escalares como vectoriales.
# Esta función es utilizada para entrenar una red neuronal que estima B_max en función del tiempo.
# La red neuronal se entrena con datos ruidosos para simular condiciones reales.
# --------------------------
def B_max(t, B_max_0=90, t_p_min=13, rho=1):
    if isinstance(t, (int, float)):
        if t < t_p_min:
            return B_max_0
        else:
            return B_max_0 - rho * (t - t_p_min)
    else:
        t = np.asarray(t)
        return np.where(t < t_p_min, B_max_0, B_max_0 - rho * (t - t_p_min))
# --------------------------
t_vals = np.linspace(0, 30, 1000).reshape(-1, 1)
B_vals = B_max(t_vals)
noise = np.random.normal(0, 1, size=B_vals.shape)
B_vals_noisy = B_vals + noise
# --------------------------
# una vez tenemos los datos de entrenamiento para la red neuronal procedemos a definir el modelo, vamos a usar tensorflow para crear la red, 
# keras.Sequential es una clase en Keras (parte de TensorFlow) que permite crear modelos de redes neuronales capa por capa, de forma lineal. 
# Es ideal para arquitecturas simples donde los datos fluyen secuencialmente de una capa a la siguiente, sin ramificaciones.
# Se define una red neuronal con 3 capas densas, las dos primeras con activación ReLU y la última sin activación.
# Esta red se entrenará para predecir B_max a partir de t.
# Se compila el modelo con el optimizador Adam y la función de pérdida MSE (error cuadrático medio).
# Se entrena el modelo con los datos ruidosos durante 100 épocas.
# Se usa verbose=0 para suprimir la salida del entrenamiento.
# Se entrena el modelo con los datos ruidosos durante 100 épocas.
# --------------------------
model_nn = keras.Sequential([
    layers.Input(shape=(1,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])
model_nn.compile(optimizer='adam', loss='mse')
model_nn.fit(t_vals, B_vals_noisy, epochs=100, verbose=0)
# --------------------------
# Comparación visual de los resultados de la red neuronal con la función real B_max.
# Se crea un conjunto de datos de prueba t_test y se calcula B_true usando la función B_max.
# Luego, se predice B_pred usando el modelo entrenado.
# Finalmente, se grafican ambos resultados para comparar la estimación de la red neuronal con la función real.
# --------------------------

t_test = np.linspace(0, 30, 1000).reshape(-1, 1)
B_true = B_max(t_test)
B_pred = model_nn.predict(t_test, verbose=0)

plt.figure(figsize=(10, 6))
plt.plot(t_test, B_true, label='Función real B_max(t)', linewidth=2)
plt.plot(t_test, B_pred, label='Red neuronal', linestyle='--')
plt.xlabel('Tiempo (días)')
plt.ylabel('B_max')
plt.title('Comparación entre B_max real y estimada por red neuronal')
plt.legend()
plt.grid(True)
plt.show()

# --------------------------
# 2. Se dedinen las Funciones auxiliares necesarias para el modelo
# --------------------------
# La función instar clasifica el valor de B en diferentes categorías basadas en umbrales predefinidos.
# Esta clasificación se utiliza para determinar la tasa de asimilación en función del estado de crecimiento de la biomasa.
# Los valores de B se comparan con los límites establecidos para asignar una categoría.
# Si B es menor que 1, retorna 1; si está entre 1 y 5, retorna 2; y así sucesivamente.
# Si B es mayor que delta_I * B_max_0, retorna 7.   
# --------------------------
def instar(B, B_max_0=90, delta_I=5):
    if B < 1:
        return 1
    elif B < 5:
        return 2
    elif B < 10:
        return 3
    elif B < 20:
        return 4
    elif B < 35:
        return 5
    elif B < delta_I * B_max_0:
        return 6
    else:
        return 7
# --------------------------
# Se define la función tasa_asimilacion que calcula la tasa de asimilación en función del tiempo t, el valor de B, y el modelo de red neuronal.
# Esta función utiliza el modelo entrenado para predecir B_max en el tiempo t.
# Dependiendo del estado de crecimiento (I) determinado por la función instar, se calcula la tasa de asimilación.
# Si I es menor que 6, retorna a_max; si I es igual a 6, calcula una fracción basada en B y B_max_t; si I es mayor que 6, retorna 0.
# La tasa de asimilación se ajusta según los parámetros a_max, alpha, B_max_0 y delta_I.
# --------------------------
def tasa_asimilacion(t, B, model_nn, a_max=1.2, alpha=1, B_max_0=90, delta_I=5):
    B_max_t = float(model_nn.predict(np.array([[t]]), verbose=0))
    I = instar(B, B_max_0, delta_I)

    if I < 6:
        return a_max
    elif I == 6:
        fraccion = max(0, 1 - B / B_max_t)
        return a_max * (fraccion ** alpha)
    else:
        return 0
# --------------------------
# Se define la función dB_dt que representa la ecuación diferencial dB/dt.
# Esta función calcula la tasa de cambio de B en función del tiempo t, el valor de B, y el modelo de red neuronal.
# Utiliza la tasa de asimilación calculada por la función tasa_asimilacion y el valor de B_max_t predicho por el modelo.
# La ecuación incluye un término de crecimiento que depende del parámetro beta y ajusta la tasa de cambio según los parámetros m y Y_B.
# --------------------------
def dB_dt(t, B, model_nn, m=0.08, Y_B=0.44, beta=2.0):
    a = tasa_asimilacion(t, B, model_nn)
    B_max_t = float(model_nn.predict(np.array([[t]]), verbose=0))
    if B_max_t == 0:
        return 0
    growth_term = 1 - (B / B_max_t) ** beta
    return ((a - m) / (1 + Y_B)) * growth_term * B

# --------------------------
# 3. Método de Runge-Kutta de orden 4 para calcular la solución de la ecuación diferencial dB_dt 
# --------------------------
def runge_kutta_4(f, B0, t0, tf, dt, model_nn):
    t_vals = [t0]
    B_vals = [B0]
    t = t0
    B = B0
    while t < tf:
        k1 = dt * f(t, B, model_nn)
        k2 = dt * f(t + dt / 2, B + k1 / 2, model_nn)
        k3 = dt * f(t + dt / 2, B + k2 / 2, model_nn)
        k4 = dt * f(t + dt, B + k3, model_nn)
        B += (k1 + 2*k2 + 2*k3 + k4) / 6
        t += dt
        t_vals.append(t)
        B_vals.append(B)
    return np.array(t_vals), np.array(B_vals)

# --------------------------
# 4. Método de Euler para calcular la solución de la ecuación diferencial dB_dt
# --------------------------
def euler(f, B0, t0, tf, dt, model_nn):
    t_vals = [t0]
    B_vals = [B0]
    t = t0
    B = B0
    while t < tf:
        B += dt * f(t, B, model_nn)
        t += dt
        t_vals.append(t)
        B_vals.append(B)
    return np.array(t_vals), np.array(B_vals)

# --------------------------
# 5. Ejecutar métodos
# --------------------------
B0 = 0.5
t0 = 0
tf = 30
dt = 0.1

# RK4
t_rk, B_rk = runge_kutta_4(dB_dt, B0, t0, tf, dt, model_nn)

# Euler
t_euler, B_euler = euler(dB_dt, B0, t0, tf, dt, model_nn)

# --------------------------
# 6. Calcular derivadas con numpy.gradient
# --------------------------
dB_rk = np.gradient(B_rk, dt)
dB_euler = np.gradient(B_euler, dt)

# --------------------------
# 7. Graficar comparativo B(t)
# --------------------------
plt.figure(figsize=(10, 6))
plt.plot(t_rk, B_rk, label='Runge-Kutta 4°', linewidth=2)
plt.plot(t_euler, B_euler, '--', label='Euler', linewidth=2)
plt.xlabel('Tiempo (días)')
plt.ylabel('Biomasa estructural B(t)')
plt.title('Comparación de métodos numéricos (RK4 vs Euler)')
plt.legend()
plt.grid(True)
plt.show()

# --------------------------
# 8. Graficar dB/dt
# --------------------------
plt.figure(figsize=(10, 6))
plt.plot(t_rk, dB_rk, label='dB/dt (RK4)', linewidth=2)
plt.plot(t_euler, dB_euler, '--', label='dB/dt (Euler)', linewidth=2)
plt.xlabel('Tiempo (días)')
plt.ylabel('dB/dt')
plt.title('Derivada numérica de B(t) usando numpy.gradient')
plt.legend()
plt.grid(True)
plt.show()

# --------------------------
# 9. Graficar a(t)
# --------------------------
a_vals = [tasa_asimilacion(t, B, model_nn) for t, B in zip(t_rk, B_rk)]

plt.figure(figsize=(10, 6))
plt.plot(t_rk, a_vals, label='a(t)', color='orange', linewidth=2)
plt.xlabel('Tiempo (días)')
plt.ylabel('Tasa específica de asimilación a(t)')
plt.title('Evolución de a(t) en el tiempo')
plt.grid(True)
plt.legend()
plt.show()

# --------------------------
# 10. Calcular nuevas variables para determinar r_a, r_CO2_m, r_CO2_B, r_L
# --------------------------
# Definimos los parámetros m, Y_B y Y_L que se utilizarán en los cálculos posteriores.
# Estos parámetros son constantes que afectan la dinámica del modelo.
# Calculamos r_a, r_CO2_m, r_CO2_B y r_L para ambos métodos (RK4 y Euler).
# Estas variables representan diferentes flujos de CO₂ en el proceso de bioconversión.
# r_a es la tasa de asimilación total, r_CO2_m es el flujo de CO₂ por metabolismo, r_CO2_B es el flujo de CO₂ por biomasa, y r_L es el flujo de CO₂ por respiración.
# Se ajusta r_L según Y_L, que representa la proporción de CO₂ liberado en la respiración.
# --------------------------
m = 0.08
Y_B = 0.44
Y_L = 0.42

# Para RK4
a_vals_rk = np.array([tasa_asimilacion(t, B, model_nn) for t, B in zip(t_rk, B_rk)])
r_a_rk = a_vals_rk * B_rk
r_CO2_m_rk = m * B_rk
r_CO2_B_rk = Y_B * dB_rk
r_L_rk = r_a_rk - (dB_rk + r_CO2_m_rk + r_CO2_B_rk)
r_L_rk = np.where(r_L_rk >= 0, r_L_rk / (1 + Y_L), r_L_rk)

# Para Euler
a_vals_euler = np.array([tasa_asimilacion(t, B, model_nn) for t, B in zip(t_euler, B_euler)])
r_a_euler = a_vals_euler * B_euler
r_CO2_m_euler = m * B_euler
r_CO2_B_euler = Y_B * dB_euler
r_L_euler = r_a_euler - (dB_euler + r_CO2_m_euler + r_CO2_B_euler)
r_L_euler = np.where(r_L_euler >= 0, r_L_euler / (1 + Y_L), r_L_euler)

# --------------------------
# 11. Graficar r_a, r_CO2_m, r_CO2_B, r_L
# --------------------------
plt.figure(figsize=(12, 8))
plt.plot(t_rk, r_a_rk, label='r_a (RK4)', linewidth=2)
plt.plot(t_euler, r_a_euler, '--', label='r_a (Euler)', linewidth=2)
plt.xlabel('Tiempo (días)')
plt.ylabel('r_a')
plt.title('Tasa de asimilación total r_a(t)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(t_rk, r_CO2_m_rk, label='r_CO2_m (RK4)')
plt.plot(t_euler, r_CO2_m_euler, '--', label='r_CO2_m (Euler)')
plt.plot(t_rk, r_CO2_B_rk, label='r_CO2_B (RK4)')
plt.plot(t_euler, r_CO2_B_euler, '--', label='r_CO2_B (Euler)')
plt.xlabel('Tiempo (días)')
plt.ylabel('Flujos de CO2')
plt.title('Producción de CO₂')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(t_rk, r_L_rk, label='r_L (RK4)', linewidth=2)
plt.plot(t_euler, r_L_euler, '--', label='r_L (Euler)', linewidth=2)
plt.xlabel('Tiempo (días)')
plt.ylabel('r_L')
plt.title('Flujo r_L(t)')
plt.legend()
plt.grid(True)
plt.show()

# --------------------------
# 12. Calcular r_CO2_L y r_CO2 totales
# --------------------------
r_CO2_L_rk = np.where(r_L_rk >= 0, Y_L * r_L_rk, 0)
r_CO2_L_euler = np.where(r_L_euler >= 0, Y_L * r_L_euler, 0)

r_CO2_total_rk = r_CO2_m_rk + r_CO2_B_rk + r_CO2_L_rk
r_CO2_total_euler = r_CO2_m_euler + r_CO2_B_euler + r_CO2_L_euler

# --------------------------
# 13. Ecuación dCO2/dt = r_CO2
# --------------------------
def dCO2_dt(t, CO2, r_CO2_array, t_array):
# Interpolación para encontrar r_CO2 en el tiempo t
    return np.interp(t, t_array, r_CO2_array)

# Integración con RK4 para CO2
def integrar_CO2_rk4(r_CO2_array, t_array, CO2_0=0):
    CO2_vals = [CO2_0]
    CO2 = CO2_0
    for i in range(len(t_array)-1):
        dt_local = t_array[i+1] - t_array[i]
        k1 = dt_local * dCO2_dt(t_array[i], CO2, r_CO2_array, t_array)
        k2 = dt_local * dCO2_dt(t_array[i] + dt_local/2, CO2 + k1/2, r_CO2_array, t_array)
        k3 = dt_local * dCO2_dt(t_array[i] + dt_local/2, CO2 + k2/2, r_CO2_array, t_array)
        k4 = dt_local * dCO2_dt(t_array[i+1], CO2 + k3, r_CO2_array, t_array)
        CO2 += (k1 + 2*k2 + 2*k3 + k4) / 6
        CO2_vals.append(CO2)
    return np.array(CO2_vals)

# Integración con Euler para CO2
def integrar_CO2_euler(r_CO2_array, t_array, CO2_0=0):
    CO2_vals = [CO2_0]
    CO2 = CO2_0
    for i in range(len(t_array)-1):
        dt_local = t_array[i+1] - t_array[i]
        CO2 += dt_local * dCO2_dt(t_array[i], CO2, r_CO2_array, t_array)
        CO2_vals.append(CO2)
    return np.array(CO2_vals)

# --------------------------
# 14. Resolver CO2(t)
# --------------------------
CO2_rk = integrar_CO2_rk4(r_CO2_total_rk, t_rk)
CO2_euler = integrar_CO2_euler(r_CO2_total_euler, t_euler)

# --------------------------
# 15. Graficar CO2 acumulado
# --------------------------
plt.figure(figsize=(10, 6))
plt.plot(t_rk, CO2_rk, label='CO2 (RK4)', linewidth=2)
plt.plot(t_euler, CO2_euler, '--', label='CO2 (Euler)', linewidth=2)
plt.xlabel('Tiempo (días)')
plt.ylabel('CO₂ acumulado')
plt.title('Evolución de CO₂ acumulado')
plt.legend()
plt.grid(True)
plt.show()

# --------------------------
# 16. Graficar r_CO2 componentes 
# --------------------------
plt.figure(figsize=(12, 8))
plt.plot(t_rk, r_CO2_m_rk, label=r'$r_{CO2,m}$ (RK4)')
plt.plot(t_rk, r_CO2_B_rk, label=r'$r_{CO2,B}$ (RK4)')
plt.plot(t_rk, r_CO2_L_rk, label=r'$r_{CO2,L}$ (RK4)')
plt.plot(t_rk, r_CO2_total_rk, 'k--', label=r'$r_{CO2}$ total (RK4)')
plt.xlabel('Tiempo (días)')
plt.ylabel('Tasas de CO₂')
plt.title('Componentes de producción de CO₂')
plt.legend()
plt.grid(True)
plt.show()