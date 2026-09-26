# patch_1.py - trazabilidad de ecuaciones en metodos.tex
from pathlib import Path
p = Path("secciones/metodos.tex")
s = p.read_text()
reps = [
 (r"con $Y_L$ el costo de deposito de lipidos; todas las tasas en mg/larva/d.",
  r"con $Y_L$ el costo de deposito de lipidos; todas las tasas en mg/larva/d. Las ecs. \ref{eq:asimilacion}--\ref{eq:co2lar} reproducen las ecs. 4, 5, 10 y 12 de \textcite{eriksenDynamicModellingFeed2022} en su forma original; los unicos terminos introducidos en este trabajo son el factor $(1 - S(t))$ en la ec. \ref{eq:asimilacion} y el suelo $m_{\mathrm{suelo}}$ en la ec. \ref{eq:mantenimiento}."),
 (r"en g/d. El termino de CH$_4$ se usa solo como indicador cualitativo",
  r"en g/d. La forma funcional (primer orden, correccion $Q_{10}$ y respuesta tipo Monod a la humedad) es la estandar en cinetica de descomposicion de residuos \parencite{roels1980,shulerbioprocess2017}; la funcion $\xi(\theta_s)$ se introduce en este trabajo. El termino de CH$_4$ se usa solo como indicador cualitativo"),
 (r"y $\gamma_w$ el agua metabolica por unidad de CO$_2$;",
  r"y $\gamma_w = 0.41$ el agua metabolica por unidad de CO$_2$ (estequiometria respiratoria de carbohidratos);"),
 (r"con $R_g = 8.314$ J/(mol$\cdot$K) y $T$, $P$ la temperatura y presion de la camara.",
  r"con $R_g = 8.314$ J/(mol$\cdot$K) (gas ideal, \parencite{atkinsPhysicalChemistry2014}) y $T$, $P$ la temperatura y presion de la camara."),
]
for old, new in reps:
    assert s.count(old) == 1, f"no unico: {old[:60]}"
    s = s.replace(old, new)
p.write_text(s)
print("patch_1 aplicado:", len(reps), "reemplazos")
