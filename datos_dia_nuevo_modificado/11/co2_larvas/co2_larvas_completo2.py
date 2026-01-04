import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

# ======= CONFIGURACIÓN - CAMBIA AQUÍ LOS ARCHIVOS =======
archivo_larvas_alimento = 'experimento_co2.csv'         # Experimento con larvas + alimento
archivo_solo_alimento = 'experimento_co2_.csv' # Experimento solo alimento
# ========================================================

def leer_archivo_csv(archivo):
    """Lee un archivo CSV con manejo de diferentes separadores"""
    try:
        df = pd.read_csv(archivo, sep='\t', header=None)
        print(f"Archivo {archivo} leído correctamente (separador: tabulación)")
    except:
        try:
            df = pd.read_csv(archivo, sep=',', header=None)
            print(f"Archivo {archivo} leído correctamente (separador: coma)")
        except:
            try:
                df = pd.read_csv(archivo, header=None)
                print(f"Archivo {archivo} leído correctamente (separador automático)")
            except Exception as e:
                print(f"Error leyendo {archivo}: {e}")
                return None
    return df

def procesar_archivo(archivo, nombre_exp):
    """Procesa un archivo y devuelve tiempos, valores CO2 e incrementos"""
    df = leer_archivo_csv(archivo)

    if df is None:
        return None, None, None, None

    print(f"\n{nombre_exp}:")
    print(f"  - Forma del archivo: {df.shape}")
    print(f"  - Primeras 3 filas:")
    print(df.head(3))

    if df.shape[1] < 2:
        print(f"Error: {archivo} no tiene suficientes columnas")
        return None, None, None, None

    # Extraer columnas
    time_col = df.iloc[:, 0].astype(str)
    co2_col = df.iloc[:, 1].astype(float)

    tiempos = []
    co2_valores = []
    co2_incrementos = []

    base_time = None
    co2_inicial = co2_col.iloc[0]  # Primer valor como referencia

    for i, time_str in enumerate(time_col):
        try:
            time_obj = datetime.strptime(time_str, '%H:%M:%S')

            if base_time is None:
                base_time = time_obj

            # Calcular minutos transcurridos
            minutos = ((time_obj.hour - base_time.hour) * 60 +
                      (time_obj.minute - base_time.minute) +
                      (time_obj.second - base_time.second) / 60)

            tiempos.append(minutos)
            co2_valores.append(co2_col.iloc[i])
            co2_incrementos.append(co2_col.iloc[i] - co2_inicial)

        except Exception as e:
            print(f"Error procesando línea {i}: {e}")
            continue

    print(f"  - Puntos procesados: {len(tiempos)}")
    print(f"  - Duración: {tiempos[-1]:.1f} minutos")
    print(f"  - CO2 inicial: {co2_inicial} ppm")
    print(f"  - CO2 final: {co2_valores[-1]} ppm")
    print(f"  - Incremento total: {co2_incrementos[-1]:.1f} ppm")

    return tiempos, co2_valores, co2_incrementos, co2_inicial

# Procesar ambos archivos
print("=== CARGANDO Y PROCESANDO ARCHIVOS ===")
t1, co2_1, inc_1, inicial_1 = procesar_archivo(archivo_larvas_alimento, "LARVAS + ALIMENTO")
t2, co2_2, inc_2, inicial_2 = procesar_archivo(archivo_solo_alimento, "SOLO ALIMENTO")

# Verificar que ambos archivos se cargaron correctamente
if t1 is None or t2 is None:
    print("Error: No se pudieron cargar ambos archivos. Script terminado.")
    exit()

# Análisis de longitudes y tiempos
print(f"\n=== COMPARACIÓN DE EXPERIMENTOS ===")
print(f"Larvas + Alimento: {len(t1)} puntos, duración {t1[-1]:.1f} min")
print(f"Solo Alimento: {len(t2)} puntos, duración {t2[-1]:.1f} min")

# Determinar el tiempo común para el análisis
tiempo_comun = min(t1[-1], t2[-1])
print(f"Tiempo común para análisis: {tiempo_comun:.1f} minutos")

# Filtrar datos hasta el tiempo común
t1_comun = [t for t in t1 if t <= tiempo_comun]
co2_1_comun = [co2_1[i] for i, t in enumerate(t1) if t <= tiempo_comun]
inc_1_comun = [inc_1[i] for i, t in enumerate(t1) if t <= tiempo_comun]

t2_comun = [t for t in t2 if t <= tiempo_comun]
co2_2_comun = [co2_2[i] for i, t in enumerate(t2) if t <= tiempo_comun]
inc_2_comun = [inc_2[i] for i, t in enumerate(t2) if t <= tiempo_comun]

# Interpolar datos del solo alimento para los mismos tiempos que larvas+alimento
if len(t2_comun) > 1 and len(t1_comun) > 1:
    inc_alimento_interpolado = np.interp(t1_comun, t2_comun, inc_2_comun)

    # Calcular contribución de las larvas
    contribucion_larvas = np.array(inc_1_comun) - inc_alimento_interpolado

    print(f"\n=== RESULTADOS DEL ANÁLISIS ===")
    print(f"En {tiempo_comun:.1f} minutos:")
    print(f"  - Incremento Larvas + Alimento: {inc_1_comun[-1]:.1f} ppm")
    print(f"  - Incremento Solo Alimento (interpolado): {inc_alimento_interpolado[-1]:.1f} ppm")
    print(f"  - Contribución estimada de las LARVAS: {contribucion_larvas[-1]:.1f} ppm")

    # Calcular tasa de producción
    if tiempo_comun > 0:
        tasa_larvas = contribucion_larvas[-1] / tiempo_comun
        print(f"  - Tasa de producción de CO2 por larvas: {tasa_larvas:.2f} ppm/min")

    # Crear SOLO 2 gráficas
    plt.figure(figsize=(12, 8))

    # Gráfica 1: Incrementos
    plt.subplot(2, 1, 1)
    plt.plot(t1_comun, inc_1_comun, 'b-o', label='Incremento: Larvas + Alimento', linewidth=2, markersize=4)
    plt.plot(t1_comun, inc_alimento_interpolado, 'r--s', label='Incremento: Solo Alimento (interpolado)', linewidth=2, markersize=4)
    plt.xlabel('t (min)', fontsize=12)
    plt.ylabel('Incremento CO2 (ppm)', fontsize=12)
    plt.title('Incrementos de CO2 desde Valores Iniciales', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Gráfica 2: Contribución de las larvas
    plt.subplot(2, 1, 2)
    plt.plot(t1_comun, contribucion_larvas, 'g-^', label='Contribución de las LARVAS', linewidth=3, markersize=5)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('t (min)', fontsize=12)
    plt.ylabel('CO2 generado por larvas (ppm)', fontsize=12)
    plt.title('Estimación de CO2 Generado ÚNICAMENTE por las Larvas', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Configurar 4 marcas en cada eje X
    for i in range(1, 3):  # Solo 2 gráficas
        ax = plt.subplot(2, 1, i)
        x_ticks = np.linspace(0, tiempo_comun, 4)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'{tick:.1f}' for tick in x_ticks])

    plt.tight_layout()
    plt.savefig('analisis_co2_comparacion.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nGráfica guardada como: analisis_co2_comparacion.png")

else:
    print("Error: No hay suficientes datos para realizar la interpolación.")
