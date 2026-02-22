import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
import subprocess
import sys

# --- CONFIGURACIÓN DE ARCHIVOS ---
FILE_CO2 = 'experimento_co2.csv'
FILE_CH4 = 'experimento_ch4.csv'
FIG_NAME = 'figura_dinamica_gases'
TEX_NAME = 'reporte_experimento_1'

def paso_1_analisis_y_figuras():
    print(">>> PASO 1: Analizando datos y generando figuras...")

    # 1. Cargar Datos
    try:
        df_co2 = pd.read_csv(FILE_CO2, sep='\t', header=None, names=['Hora', 'CO2_ppm'])
        df_ch4 = pd.read_csv(FILE_CH4, sep='\t', header=None, names=['Hora', 'CH4_ppm'])
    except FileNotFoundError:
        print(f"ERROR: No encuentro los archivos {FILE_CO2} o {FILE_CH4}")
        sys.exit(1)

    # 2. Procesar Tiempos
    base_date = '2025-09-09'
    df_co2['Timestamp'] = pd.to_datetime(base_date + ' ' + df_co2['Hora'])
    df_ch4['Timestamp'] = pd.to_datetime(base_date + ' ' + df_ch4['Hora'])

    # Merge y Normalización
    df = pd.merge(df_co2, df_ch4, on='Timestamp', suffixes=('_co2', '_ch4'))
    t0 = df['Timestamp'].iloc[0]
    df['Minutos'] = (df['Timestamp'] - t0).dt.total_seconds() / 60.0

    # 3. Cálculos Biológicos
    # Detectar inicio de anaerobiosis (CH4 > 0)
    anaerobio_data = df[df['CH4_ppm'] > 0]
    if not anaerobio_data.empty:
        t_corte = anaerobio_data.iloc[0]['Minutos']
        co2_limite = anaerobio_data.iloc[0]['CO2_ppm']
    else:
        t_corte = df['Minutos'].max()
        co2_limite = df['CO2_ppm'].max()

    # Calcular pendiente (Fase Aerobia Valida: 2 min < t < t_corte)
    fase_aerobia = df[(df['Minutos'] > 2) & (df['Minutos'] < t_corte)]
    slope, intercept = np.polyfit(fase_aerobia['Minutos'], fase_aerobia['CO2_ppm'], 1)

    # 4. Generar la Gráfica
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_co2 = '#1f77b4'
    color_ch4 = '#ff7f0e'

    # Eje CO2
    ax1.set_xlabel('Tiempo (minutos)', fontsize=12)
    ax1.set_ylabel('Respiración $CO_2$ (ppm)', color=color_co2, fontsize=12, fontweight='bold')
    ax1.plot(df['Minutos'], df['CO2_ppm'], color=color_co2, linewidth=2, label='$CO_2$ Aerobio')
    ax1.tick_params(axis='y', labelcolor=color_co2)
    ax1.grid(True, alpha=0.3)

    # Eje CH4
    ax2 = ax1.twinx()
    ax2.set_ylabel('Fermentación $CH_4$ (ppm)', color=color_ch4, fontsize=12, fontweight='bold')
    ax2.plot(df['Minutos'], df['CH4_ppm'], color=color_ch4, linewidth=2, linestyle='--', label='$CH_4$ Anaerobio')
    ax2.tick_params(axis='y', labelcolor=color_ch4)

    # Linea de corte
    plt.axvline(x=t_corte, color='red', linestyle=':', linewidth=1.5, label='Límite PINN')

    # Anotaciones
    plt.title(f'Dinámica Metabólica BSF (Tasa: {slope:.1f} ppm/min)', fontsize=14)
    fig.tight_layout()

    # GUARDAR FIGURAS
    print(f"    ...Guardando {FIG_NAME}.pdf (Vectorial)")
    plt.savefig(f"{FIG_NAME}.pdf", format='pdf', dpi=300)
    print(f"    ...Guardando {FIG_NAME}.png (Imagen)")
    plt.savefig(f"{FIG_NAME}.png", format='png', dpi=300)
    plt.close()

    return slope, t_corte, co2_limite

def paso_2_generar_latex(slope, t_corte, co2_limite):
    print(">>> PASO 2: Generando archivo LaTeX...")

    # NOTA: En f-strings, las llaves de LaTeX {{ }} deben ir dobles.
    # Las llaves de variables de Python { } van simples.

    latex_content = fr"""
\documentclass[12pt, a4paper]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[spanish]{{babel}}
\usepackage{{graphicx}}
\usepackage{{geometry}}
\usepackage{{amsmath}}
\geometry{{margin=2.5cm}}

\title{{\textbf{{Reporte Automático: Experimento BSF post-ayuno}}}}
\author{{Generado por Script de Tesis}}
\date{{\today}}

\begin{{document}}

\maketitle

\section{{Resumen de Parámetros Calculados}}
El script de análisis procesó los datos crudos y determinó los siguientes valores críticos para el modelo:

\begin{{itemize}}
    \item \textbf{{Tasa Metabólica Aerobia ($V_{{max}}$):}} {slope:.2f} ppm/min
    \item \textbf{{Tiempo de Quiebre Anaerobio:}} {t_corte:.2f} minutos
    \item \textbf{{Límite de $CO_2$ Válido:}} {co2_limite:.0f} ppm
\end{{itemize}}

\section{{Visualización de Resultados}}
La Figura \ref{{fig:gases}} muestra la transición metabólica detectada. Se observa claramente el punto donde inicia la producción de metano.

\begin{{figure}}[h]
    \centering
    % Inyectamos el nombre del archivo de la figura
    \includegraphics[width=0.9\textwidth]{{{FIG_NAME}.pdf}}
    \caption{{Dinámica de gases y zona de corte para el entrenamiento de la PINN.}}
    \label{{fig:gases}}
\end{{figure}}

\section{{Conclusión}}
Para el entrenamiento de la red neuronal, se utilizará el intervalo temporal $t \in [2, {t_corte:.1f}]$. Los datos posteriores se descartan por violar la estequiometría aerobia.

\end{{document}}
    """

    with open(f"{TEX_NAME}.tex", "w", encoding='utf-8') as f:
        f.write(latex_content)
    print(f"    ...Archivo {TEX_NAME}.tex creado exitosamente.")

def paso_3_compilar_pdf():
    print(">>> PASO 3: Compilando PDF final con pdflatex...")

    # -interaction=nonstopmode evita que se detenga si hay errores pequeños
    comando = f"pdflatex -interaction=nonstopmode {TEX_NAME}.tex"

    result = subprocess.run(comando, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode == 0:
        print(f"\n¡ÉXITO! El documento '{TEX_NAME}.pdf' ha sido generado.")
    else:
        print("\nERROR EN LA COMPILACIÓN LATEX:")
        print("Asegúrate de tener instalado 'pdflatex' (TeX Live o MikTeX)")
        # Decodificamos el error para verlo en consola
        print(result.stdout.decode('latin-1', errors='replace'))

# --- EJECUCIÓN MAESTRA ---
if __name__ == "__main__":
    dato_slope, dato_t, dato_co2 = paso_1_analisis_y_figuras()
    paso_2_generar_latex(dato_slope, dato_t, dato_co2)
    paso_3_compilar_pdf()
