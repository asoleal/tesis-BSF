# patch_menores.py — menores del review: unificar Cuadro, (P2), ref Cuadro 2,
# §3.4 sin redundancia, titulo PINN corto, conclusiones 2-5 fusionadas
from pathlib import Path
f = Path("contenido_art4.tex")
s = f.read_text()

# 1) Tabla 1 -> Cuadro 1
old = "(Tabla~\\ref{tab:cfd})"
assert s.count(old) == 1
s = s.replace(old, "(Cuadro~\\ref{tab:cfd})")

# 2) etiquetas (P2) en 3.2 y 3.3
old = "\\subsection{Escenarios de puertos en fase ventilada}"
assert s.count(old) == 1
s = s.replace(old, "\\subsection{Escenarios de puertos en fase ventilada (P2)}")
old = "\\subsection{Mezcla forzada con ventilador en tapa (C5)}"
assert s.count(old) == 1
s = s.replace(old, "\\subsection{Mezcla forzada con ventilador en tapa (C5, P2)}")

# 3) citar Cuadro 2 en el texto
old = ("La Figura~\\ref{fig:validacion} contrasta el modelo con las tasas\n"
       "observadas.")
assert s.count(old) == 1
s = s.replace(old,
              "La Figura~\\ref{fig:validacion} contrasta el modelo con las "
              "tasas\nobservadas; el Cuadro~\\ref{tab:validacion} las resume "
              "por dia del ciclo.")

# 4) §3.4: quitar repeticion de §2.1
a0 = "Como primera validacion se uso el experimento"
a1 = "datos\\_finales\\_PINN\\_corregidos.csv}."
assert s.count(a0) == 1 and s.count(a1) >= 1
i0 = s.index(a0); i1 = s.index(a1, i0) + len(a1)
nuevo_p34 = (
    "Como primera validacion se uso el experimento de septiembre de 2025 "
    "descrito en la Seccion 2.1 (panera sellada de 12.4 L, $N = 700$\n"
    "larvas, dietas D1/D4, cierres de acumulacion con bandeja control del "
    "mismo dia, supuesto S7). Las tasas extraidas, con su R$^2$ de ajuste, "
    "estan en\n\\texttt{datos/experimentos/datos\\_finales\\_PINN\\_corregidos.csv}.")
s = s[:i0] + nuevo_p34 + s[i1:]

# 5) titulo PINN corto (evita corte feo en el PDF)
old = "\\subsection{Trabajo en curso: correccion con redes informadas por la fisica}"
assert s.count(old) == 1
s = s.replace(old, "\\subsection{Trabajo en curso: correccion con PINN}")

# 6) fusionar conclusiones 2-5 en dos items
b0 = "  \\item En fase ventilada, ninguna configuracion"
b1 = "correccion de volumen muerto del gemelo digital."
assert s.count(b0) == 1 and s.count(b1) == 1
j0 = s.index(b0); j1 = s.index(b1, j0) + len(b1)
nuevos = (
    "  \\item En fase ventilada, ninguna configuracion de puertos "
    "homogeneiza la\n  camara en regimen laminar ($\\tau_{mix} > 720$ s "
    "frente a $\\tau_{aire} \\approx 175$ s): el\n  supuesto A1 no es "
    "sostenible en fase abierta sin mezcla forzada. Y al ser cotas\n  "
    "conservadoras (laminar, isotermo, difusividad molecular), los "
    "resultados\n  subestiman la mezcla real; la discrepancia se cerrara "
    "con la calibracion\n  experimental con dos sensores a distinta altura "
    "y con la correccion de volumen\n  muerto del gemelo digital.\n"
    "  \\item Un ventilador pequeno en la tapa homogeneiza la camara "
    "sellada en\n  menos de 60 s ($\\tau_{mix} = 4$--$24$ s para un rango "
    "de 100$\\times$ en la\n  fuerza inyectada), validando el supuesto A1 "
    "para los cierres de medicion con\n  margen de 10--30$\\times$; y el "
    "escalamiento lineal de Stokes, verificado de\n  forma exacta, permite "
    "dimensionar el ventilador sin corridas adicionales.")
s = s[:j0] + nuevos + s[j1:]

f.write_text(s)
print("OK: 6 menores aplicados")
