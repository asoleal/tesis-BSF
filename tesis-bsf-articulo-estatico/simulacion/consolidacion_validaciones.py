#!/usr/bin/env python3
"""Consolidacion Fase B (B0-B6): tabla unica predicho vs observado.

Los numeros aqui son los APROBADOS en la Fase B (calibracion aceptada con
nucleo de instares de Eriksen + validaciones externas). Este script NO
recalcula nada: solo los reune en un CSV y una tabla LaTeX para el articulo.
"""
import csv
import os

AQUI = os.path.dirname(os.path.abspath(__file__))

# (bloque, fuente, metrica, modelo, observado, criterio, resultado)
FILAS = [
    ("B0", "Propia (pesadas d0)", "Masa inicial $B_0 + L_0$ (mg MS/larva)",
     "1.43", "1.3 / 0.9 (D1/D4)", "orden de magnitud", "CUMPLE"),
    ("B1", "Propia (camara estatica)", "RMSE flujo CO$_2$ D1 (ppm/min)",
     "25.1", "---", "---", "Ajuste"),
    ("B1", "Propia (camara estatica)", "RMSE flujo CO$_2$ D4 (ppm/min)",
     "39.3 ($R^2 = -0.08$)", "---", "---", "Limitacion declarada"),
    ("B2", "Propia", "RMSE biomasa D1 / D4 (mg PH)",
     "25.1 / 15.2", "---", "---", "Ajuste"),
    ("B2", "Propia", "Masa larval d14 D1 / D4 (mg PH)",
     "169 / 144", "173 / 135", "$\\pm$10\\,\\%", "CUMPLE"),
    ("B3", "Propia (bromatologia)", "Lipidos d14 D1 / D4 (\\% MS)",
     "34 / 28", "33.4 / 26.6 (EE)", "punto a punto", "CUMPLE"),
    ("B4", "Propia (ec. 22 Eriksen)", "Fraccion MS $\\delta_{DW}$",
     "0.39", "0.33--0.45", "dentro del rango", "CUMPLE"),
    ("B5", "Ermolaev et al.", "Tasas CO$_2$ (3 pts) + acumulado d14 (g kg$^{-1}$)",
     "90.8", "$92.4 \\pm 14.5$", "residuos $< 1\\sigma$", "CUMPLE"),
    ("B5", "Ermolaev et al.", "Masa larval final (mg)",
     "111.1", "101.9", "factor 2", "CUMPLE ($+9$\\,\\%)"),
    ("B5", "Ermolaev et al.", "$t_{pm}$ (d) / $B_{max}$ (mg) recuperados",
     "13.0 / 94.8", "13 / 90 (Eriksen)", "consistencia", "CUMPLE"),
    ("B6", "Sch\\o{}n et al.", "$f$ por dieta (REF, 1:3, 1:5, 1:9)",
     "0.899, 0.874, 0.870, 0.725", "orden P:C", "coherente", "CUMPLE"),
    ("B6", "Sch\\o{}n et al.", "VCO$_2$ total (L)",
     "15.0, 14.7, 14.7, 12.6", "19.6, 16.9, 16.2, 10.8", "$\\pm$25\\,\\%", "CUMPLE"),
    ("B6", "Sch\\o{}n et al.", "Masa final (mg MS)",
     "51.5, 50.0, 49.8, 40.0", "43.2, 37.7, 33.5, 21.0", "factor 2", "CUMPLE"),
]

ENCABEZADO = ["bloque", "fuente", "metrica", "modelo", "observado",
              "criterio", "resultado"]

# --- CSV ---
ruta_csv = os.path.join(AQUI, "resultados_consolidacion_validaciones.csv")
with open(ruta_csv, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(ENCABEZADO)
    for fila in FILAS:
        w.writerow(fila)

# --- Tabla LaTeX (sin booktabs, solo \\hline; ajustada al ancho de pagina) ---
lin = []
lin.append("% Generada por simulacion/consolidacion_validaciones.py (Fase B)")
lin.append("\\begin{table}[htbp]")
lin.append("  \\centering")
lin.append("  \\caption{Consolidaci\\'on de las validaciones del modelo "
           "(bloques B0--B6). B0--B4 corresponden al ajuste y a los chequeos "
           "con datos propios; B5--B6 son validaciones externas por "
           "transferencia de par\\'ametros. Los RMSE de B1 est\\'an en "
           "ppm/min y los de B2 en mg de peso h\\'umedo por larva.}")
lin.append("  \\label{tab:consolidacion_validaciones}")
lin.append("  \\small")
lin.append("  \\resizebox{\\textwidth}{!}{%")
lin.append("  \\begin{tabular}{l l l c c c l}")
lin.append("    \\hline")
lin.append("    Bloque & Fuente & M\\'etrica & Modelo & Observado "
           "& Criterio & Resultado \\\\")
lin.append("    \\hline")
for fila in FILAS:
    lin.append("    " + " & ".join(fila) + " \\\\")
lin.append("    \\hline")
lin.append("  \\end{tabular}}%")
lin.append("\\end{table}")
tabla = "\n".join(lin)

ruta_tex = os.path.join(AQUI, "tabla_consolidacion_validaciones.tex")
with open(ruta_tex, "w", encoding="utf-8") as fh:
    fh.write(tabla + "\n")

print(tabla)
print()
print("Escritos:")
print(" -", ruta_csv)
print(" -", ruta_tex)
