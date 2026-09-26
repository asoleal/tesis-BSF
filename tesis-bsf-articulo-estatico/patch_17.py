# patch_17.py - formalizacion de la propagacion de errores en sec. 2.4
from pathlib import Path
p = Path("secciones/metodos.tex"); s = p.read_text()
old = ("El tiempo de respuesta del sensor ($T_{90} < 30$ s) es despreciable frente a la "
"duracion de los cierres.")
new = r'''Formalmente, para la tasa molar $R = f(m, V_{\mathrm{aire}}, T)$, con $m$ la pendiente medida, la ley de propagacion para errores independientes da
\begin{equation}
\left(\frac{u_R}{R}\right)^2 = \left(\frac{u_m}{m}\right)^2 + \left(\frac{u_V}{V_{\mathrm{aire}}}\right)^2 + \left(\frac{u_T}{T}\right)^2,
\label{eq:propagacion}
\end{equation}
con $u_m/m \approx 0.06$ a $0.12$ (instrumental, sistematica: afecta todos los cierres en el mismo sentido y no se reduce al promediar), $u_V/V_{\mathrm{aire}} < 0.02$ y $u_T/T < 0.005$, lo que da una incertidumbre combinada de 6.5 a 12 \% por tasa. La componente aleatoria (dispersion del ajuste lineal y entre repeticiones) se suma en cuadratura y domina la dispersion punto a punto observada entre el modelo y los datos. El tiempo de respuesta del sensor ($T_{90} < 30$ s) es despreciable frente a la duracion de los cierres.'''
assert s.count(old) == 1
p.write_text(s.replace(old, new))
print("patch_17 aplicado: ley de propagacion formal")
