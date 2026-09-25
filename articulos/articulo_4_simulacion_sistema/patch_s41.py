# patch_s41.py — §4.1 objetivo/criterio (nueva) + §4.2 geometria/malla (expandida)
from pathlib import Path

f = Path("contenido_art4.tex")
s = f.read_text()

ini = "\\section{Verificacion CFD de la mezcla}"
fin = "\\subsection{Modelo de flujo}"
assert s.count(ini) == 1 and s.count(fin) == 1, \
    f"ini={s.count(ini)} fin={s.count(fin)}"
i0, i1 = s.index(ini), s.index(fin)
assert i0 < i1

nuevo = r'''\section{Verificacion CFD de la mezcla}
\subsection{Objetivo y criterio de aceptacion}

El modelo 0D descrito en la Seccion~\ref{sec:dinamica0d} asume que el aire
de la camara esta perfectamente mezclado en todo instante (supuesto A1):
existe una unica concentracion de cada gas y la lectura del sensor la
representa. El objetivo de esta seccion es verificar ese supuesto para el
diseno de la camara y del protocolo. Para ello se resuelve el flujo y el
transporte de un trazador pasivo de CO$_2$ que no modela la biologia del
proceso: el CFD aporta el campo de mezcla con el que el gemelo digital
interpreta, o corrige, las lecturas del sensor.

El criterio de aceptacion es temporal. Con el indice de heterogeneidad
$\eta(t) = \sigma_c(t)/\bar{c}_0$ del trazador y el tiempo de mezcla
$\tau_{mix}$ (primer instante con $\eta < 0.05$), A1 se considera valido si

\begin{equation}
  \tau_{mix}/\tau_{aire} \leq 1.2, \qquad
  \tau_{aire} = V_{aire}/Q_{eff},
\end{equation}

es decir, si la camara se homogeneiza en una constante de tiempo del aire.
En fase ventilada la renovacion y la mezcla son comparables; en fase
cerrada ($\tau_{aire} \rightarrow \infty$) la homogeneidad debe lograrla un
mecanismo interno antes de que la pendiente de acumulacion se contamine por
heterogeneidad espacial.

Como hipotesis de trabajo se adopta el regimen de Stokes, sin inercia ni
turbulencia. Esta eleccion es una cota conservadora: el jet real del puerto
es transicional ($\mathrm{Re} \approx 220$) y la turbulencia real mezcla mas
que el modelo, de modo que los tiempos de mezcla calculados son cotas
superiores de los tiempos reales.

\subsection{Geometria y malla}

La camara es un contenedor de $26 \times 19 \times 18$ cm
($\approx 8.9$ L geometricos) empleado para conservar la temperatura del
lote; el volumen de aire efectivo sobre el lecho de cria es
$V_{aire} = 2.5$ L. Los puertos de entrada y salida son discos conformales
de radio 3 mm que sobresalen 1 mm de la cara para garantizar su
reconocimiento en el mallado; a ese radio, el caudal nominal de 1 L/min
exige una velocidad de puerto $U_{in} = 0.589$ m/s ($\mathrm{Re} \approx 220$,
transicional).

La malla se genero con gmsh: fragmentacion con el volumen mas puntos de
refinamiento local (tamano de elemento 1--1.5 mm) que resuelven el chorro
de entrada; el resultado son $\sim 32\,000$ tetraedros, usados con
elementos P2 para la velocidad y P1 para el trazador. Las fronteras se
marcaron por centroide geometrico (la lectura del mallado aborta si este
lleva grupos fisicos en superficies), y los puertos quedaron con 29 facetas
cada uno, lo que verifica la resolucion local del jet.

'''

s = s[:i0] + nuevo + s[i1:]
f.write_text(s)
print("OK: S4.1 y S4.2 escritas,", len(nuevo.splitlines()), "lineas nuevas")
