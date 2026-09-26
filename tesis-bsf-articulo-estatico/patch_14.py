# patch_14.py - nueva subseccion: solucion numerica y ajuste de parametros
from pathlib import Path
p = Path("secciones/metodos.tex"); s = p.read_text()
old = ("El sistema se integra con \\texttt{solve\\_ivp} (LSODA, tolerancias $10^{-6}$ y "
"$10^{-9}$) por segmentos entre eventos de reposicion. El codigo fuente y los datos quedan "
"disponibles en el repositorio del articulo.")
new = r'''\subsection{Solucion numerica y ajuste de parametros}
\label{sec:numerica}

El estado que se integra es $(B, L, D_M, W)$: el compartimento $A$ queda en cuasi-equilibrio por construccion (ec. \ref{eq:co2lar}) y las concentraciones de gas no se integran en el tiempo, pues la camara se abre entre cierres; la comparacion con el sensor se hace sobre la tasa aparente instantanea (ec. \ref{eq:ppm}). El sistema es hibrido: la dinamica continua se interrumpe cada dos dias en los eventos de reposicion, donde $D_M$ y $W$ se restablecen a 100 y 150 g como saltos discretos. Cada segmento entre eventos se integro con \texttt{solve\_ivp} (metodo LSODA, tolerancias relativa $10^{-6}$ y absoluta $10^{-9}$), implementado en Python 3 con NumPy y SciPy.

El ajuste de parametros se hizo en dos etapas desacopladas, siguiendo la descomposicion de fuentes. En la primera, los parametros microbianos ($k_{\mathrm{ref}}$, $Y_{CO_2}$) se ajustaron por minimos cuadrados sobre las tasas de los controles de alimento solo, simulando el lecho sin larvas y sin reposicion, en escala logaritmica y con cotas $k_{\mathrm{ref}} \in [10^{-4}, 0.2]$ d$^{-1}$ e $Y_{CO_2} \in [0.05, 2]$. En la segunda, el ajuste conjunto del nucleo larvario se hizo sobre las tasas netas con cuatro grados de libertad: $B_0$ compartido entre dietas (misma postura de huevos), el factor de calidad $f$ de D4, $t_p$ por dieta y $w_p$ compartido; los cierres saturados entraron como restricciones de desigualdad (residuo nulo cuando el modelo supera la cota inferior) y el ajuste se repitio desde tres puntos de arranque para evitar minimos locales. Los parametros cineticos del nucleo DEB no se reajustaron. El codigo fuente y los datos quedan disponibles en el repositorio del articulo.'''
assert s.count(old) == 1
p.write_text(s.replace(old, new))
print("patch_14 aplicado: subseccion solucion numerica")
