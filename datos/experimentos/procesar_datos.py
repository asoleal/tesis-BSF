import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

# ==========================================
# CONFIGURACIÓN
# ==========================================
BASE_DIR = "."
LIMITE_SATURACION = 5850
GASES = ["co2", "ch4"]
DIAS_DISPONIBLES = ["9", "11", "13", "17"]

# ==========================================
# LÓGICA DE CLASIFICACIÓN
# ==========================================
def clasificar_carpeta(nombre_carpeta):
    nombre = nombre_carpeta.lower()
    if "alimento" in nombre and "d1" in nombre: return "Control_Alimento_D1"
    if "alimento" in nombre and "d4" in nombre: return "Control_Alimento_D4"
    if "d1" in nombre: return "D1_Tratamiento"
    if "d4" in nombre: return "D4_Tratamiento"
    if "larvas" in nombre and "solas" in nombre: return "Larvas_Ayuno"
    if "generacion" in nombre: return "Larvas_Generacion_Base"
    return "Desconocido"

# ==========================================
# PROCESAMIENTO (CORREGIDO PARA TABULADORES)
# ==========================================
def procesar_archivo_tsv(ruta_completa, dia, etiqueta, gas):
    try:
        # CORRECCIÓN CLAVE:
        # 1. sep='\t' (Tabulador)
        # 2. header=None (Porque la primera linea ya es dato '15:21..')
        df = pd.read_csv(ruta_completa, sep='\t', header=None)
    except:
        return []

    # Asumimos que la columna 0 es la Hora (texto) y las siguientes son datos
    # Eliminamos la columna 0 de hora para quedarnos solo con los números
    df_numerico = df.iloc[:, 1:].copy()

    # Tomar solo los primeros 30 minutos (filas)
    df_numerico = df_numerico.head(30)

    resultados = []

    # Iterar sobre las columnas de datos (ahora pueden llamarse 1, 2, 3...)
    for col_idx in df_numerico.columns:
        # Convertir a número forzando errores a NaN
        y_raw = pd.to_numeric(df_numerico[col_idx], errors='coerce').dropna()

        if len(y_raw) < 5: continue

        x_raw = np.arange(len(y_raw)) # Minutos (0, 1, 2...)

        # --- FILTRO SATURACIÓN ---
        mask_validos = y_raw < LIMITE_SATURACION
        y_final = y_raw[mask_validos]
        x_final = x_raw[mask_validos]

        saturado = False
        if len(y_final) < 3:
            # Si se saturó al instante, tomamos los primeros 4 puntos crudos
            # para tener al menos una pendiente empinada
            y_final = y_raw.iloc[:4]
            x_final = x_raw[:4]
            saturado = True
        elif len(y_final) < len(y_raw):
            saturado = True

        # --- CÁLCULO PENDIENTE ---
        slope, _, r2, _, _ = linregress(x_final, y_final)

        resultados.append({
            "Dia_Experimento": int(dia),
            "Etiqueta": etiqueta,
            "Gas": gas,
            "Muestra_ID": f"{os.path.basename(os.path.dirname(ruta_completa))}_col{col_idx}",
            "Tasa_Produccion": slope,
            "Saturado": saturado,
            "R2": r2**2
        })
    return resultados

# ==========================================
# EJECUCIÓN
# ==========================================
print("--- 🚀 Iniciando Procesamiento (Modo TSV/Tabulador) ---")

datos_acumulados = []

for dia in DIAS_DISPONIBLES:
    ruta_dia = os.path.join(BASE_DIR, dia)
    if not os.path.exists(ruta_dia): continue

    subcarpetas = [f for f in os.listdir(ruta_dia) if os.path.isdir(os.path.join(ruta_dia, f))]
    print(f"📂 Día {dia}: Analizando {len(subcarpetas)} carpetas...")

    for sub in subcarpetas:
        etiqueta = clasificar_carpeta(sub)
        ruta_sub = os.path.join(ruta_dia, sub)

        for gas in GASES:
            archivo = f"experimento_{gas}.csv" # Aunque es TSV, se llama .csv
            ruta_csv = os.path.join(ruta_sub, archivo)

            if os.path.exists(ruta_csv):
                datos = procesar_archivo_tsv(ruta_csv, dia, etiqueta, gas)
                datos_acumulados.extend(datos)

# Convertir y Guardar
if len(datos_acumulados) > 0:
    df_final = pd.DataFrame(datos_acumulados)

    # CSV Final para la PINN
    csv_salida = "datos_finales_PINN_corregidos.csv"
    df_final.to_csv(csv_salida, index=False)

    print(f"\n✅ ¡ÉXITO! Se generaron {len(df_final)} puntos de datos.")
    print(f"📁 Archivo guardado: {csv_salida}")

    # --- VISUALIZACIÓN RÁPIDA ---
    print("📊 Generando gráfica resumen...")
    try:
        plt.figure(figsize=(10, 6))
        # Filtramos solo CO2 y Tratamientos clave para ver si funciona
        subset = df_final[
            (df_final["Gas"] == "co2") &
            (df_final["Etiqueta"].str.contains("Tratamiento|Control"))
        ]

        sns.lineplot(
            data=subset,
            x="Dia_Experimento",
            y="Tasa_Produccion",
            hue="Etiqueta",
            marker="o"
        )
        plt.title("Evolución de Actividad Larvaria (CO2)")
        plt.ylabel("Tasa (ppm/min)")
        plt.grid(True, alpha=0.3)
        plt.show()
    except Exception as e:
        print(f"No se pudo graficar (falta entorno gráfico?): {e}")

else:
    print("❌ Aún no se recuperan datos. Revisa si los archivos tienen contenido.")
