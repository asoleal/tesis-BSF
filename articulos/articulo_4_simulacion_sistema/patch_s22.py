# patch_s22.py — reescribe §2.2 (Entradas conmutadas y cierres)
from pathlib import Path

f = Path("contenido_art4.tex")
s = f.read_text()

ini = "\\subsection{Entradas conmutadas y cierres}"
fin = "\\section{Simulacion del sistema dinamico}"
assert s.count(ini) == 1, f"ini count={s.count(ini)}"
assert s.count(fin) == 1, f"fin count={s.count(fin)}"
i0 = s.index(ini)
i1 = s.index(fin)
assert i0 < i1

nuevo = r'''\subsection{Entradas conmutadas y cierres}

El protocolo de medicion alterna fases ventiladas y cierres. Con la
ventilacion activa, el caudal $Q(t)$ lo fija un controlador PI que mantiene
el O$_2$ en banda operativa, con $Q = 4$ L/min entre cierres (supuesto S11);
durante un cierre, $Q = 0$ y la concentracion evoluciona por acumulacion
pura, $\mathrm{d}c_i/\mathrm{d}t = \sigma_i R_i/V_{aire}$, de donde se estima
la tasa por ajuste de pendiente.

El calendario de cierres es adaptativo y de dos pasadas. La primera integra
el ciclo con cierres nominales de $\Delta = 15$ min y registra la
concentracion al inicio de cada cierre, $c(t_{c,k})$. La segunda fija la
duracion de cada cierre para que la acumulacion proyectada no supere el tope
del NDIR, $c_{max} = 5000$ ppm (supuesto S9), con la tasa evaluada en el
punto medio del cierre, $t_{m,k} = t_{c,k} + \Delta_k/2$:

\begin{equation}
  \Delta_k = \mathrm{clip}\left(
    \frac{c_{max} - c(t_{c,k})}{\dot{c}(t_{m,k})},\
    120\ \mathrm{s},\ 1800\ \mathrm{s} \right)
\end{equation}

El piso de 120 s protege la resolucion temporal del cierre y el techo de
1800 s preserva la linealidad de la pendiente; la evaluacion en el punto
medio corrige la sobreestimacion propia de los cierres largos, cuando la
tasa cae a lo largo del dia.

Un resultado de diseno relevante es que el factor limitante no es $\Delta$
sino la ventilacion entre cierres. La concentracion de base entre cierres es
$c_{in} + R/Q$; con $Q = 1$ L/min alcanza unos 4300 ppm en el pico del
ciclo y la camara satura aun con la tapa abierta, de modo que ningun
$\Delta$ admisible evita el tope. Con $Q = 4$ L/min la base se mantiene
entre 500 y 1600 ppm y ningun cierre supera los 5000 ppm en los escenarios
simulados (E2: pico 4978 ppm; E5: pico 4604 ppm).

'''

s = s[:i0] + nuevo + s[i1:]
f.write_text(s)
print("OK: S2.2 reescrita,", len(nuevo.splitlines()), "lineas nuevas")
