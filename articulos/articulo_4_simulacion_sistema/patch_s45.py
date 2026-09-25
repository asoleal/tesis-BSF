# patch_s45.py — §4.5 Metricas (expandida) + §4.6 Verificacion numerica (nueva)
from pathlib import Path

f = Path("contenido_art4.tex")
s = f.read_text()

ini = "\\subsection{Metricas de mezcla}"
fin = "\\subsection{Escenarios de puertos en fase ventilada}"
assert s.count(ini) == 1 and s.count(fin) == 1, \
    f"ini={s.count(ini)} fin={s.count(fin)}"
i0, i1 = s.index(ini), s.index(fin)
assert i0 < i1

nuevo = r'''\subsection{Metricas de mezcla}

Con la media espacial $\bar{c}(t) = \frac{1}{V}\int_\Omega c\,dx$ y la
desviacion estandar $\sigma_c(t) = \left( \frac{1}{V}\int_\Omega
(c - \bar{c})^2\,dx \right)^{1/2}$, el indice de heterogeneidad es

\begin{equation}
  \eta(t) = \sigma_c(t)/\bar{c}_0,
\end{equation}

normalizado con la concentracion media inicial $\bar{c}_0$ para que arranque
cerca de $\eta(0) \approx 1$ independientemente del volumen. El tiempo de
mezcla $\tau_{mix}$ es el primer instante con $\eta < 0.05$ (mezcla al
95\%).

En fase ventilada el referente es $\tau_{aire} = V_{aire}/Q_{eff}$. El
caudal efectivo, medido por integrales de frontera, resulta 15--18\% inferior
al nominal: la banda de elementos del borde del disco no lleva la velocidad
impuesta (el campo P2 decae antes del contorno), lo que equivale a un area
de puerto ligeramente menor. El fraccionamiento de la camara se evalua
comparando la masa remanente $m(t)/m_0$ contra el decaimiento del reactor
ideal $e^{-t/\tau_{aire}}$: la diferencia cuantifica el bypass, la fraccion
de trazador atrapada en zonas muertas que el flujo no barre.

\subsection{Verificacion numerica}

Tres verificaciones respaldan los resultados. (i)~Balance de masa exacto en
cada frontera: en C2 la entrada y la salida coinciden a 1.034 L/min, y en
camara sellada la masa del trazador deriva menos del 1\% en 720 s.
(ii)~Linealidad de Stokes verificada de forma exacta: al dividir la fuerza
del ventilador por 100 la velocidad maxima cae exactamente un factor 100
(1101.8 a 11.02 m/s), lo que permite extrapolar cualquier fuerza sin nuevas
simulaciones. (iii)~Control negativo C3: una entrada tangente pura inyecta
masa nula ($\int \mathbf{u}\cdot\mathbf{n}\,ds = 0$) y la bomba de salida
solo succiona, de modo que $\eta$ cayo por artefacto sin mezcla real; este
caso quedo descartado por el balance de masa, no por la plausibilidad del
campo. La conclusion conjunta es que, bajo la hipotesis de cota
conservadora del regimen de Stokes, los tiempos calculados son cotas
superiores de la mezcla real.

'''

s = s[:i0] + nuevo + s[i1:]
f.write_text(s)
print("OK: S4.5 y S4.6 escritas,", len(nuevo.splitlines()), "lineas nuevas")
