# patch_paso2.py — §2.1 Sistema experimental (nueva)
from pathlib import Path
f = Path("contenido_art4.tex")
s = f.read_text()

ini = "\\subsection{Dinamica 0D: doce estados}"
assert s.count(ini) == 1, s.count(ini)

nuevo = r'''\subsection{Sistema experimental}
\label{sec:sistema}

El experimento preliminar se realizo en septiembre de 2025 en una panera
de plastico sellada (base $30 \times 19$ cm, tapa $36 \times 25$ cm,
altura 17 cm), cuyo volumen interno de tronco de cono es
$V \approx 12.4$ L; descontando el lecho, el volumen de aire efectivo es
$V_{aire} = 12.1$ L (supuesto S1). En la panera se criaron $N = 700$
larvas de \emph{Hermetia illucens} desde huevo, con el ciclo completo
hasta la prepupa (18 d), a temperatura ambiente cercana a 27~$^{\circ}$C
(S4).

Se evaluaron dos dietas: D1, dieta estandar de sustrato de pollo
reportada en la literatura, y D4, dieta de residuos agroindustriales
(frutas, cascaras de naranja y residuos similares) procesada con
\emph{Bacillus}. En ambos casos el alimento se suministro a libre
disposicion en raciones de 250 g, renovadas en cada jornada de medicion
(S3).

Las emisiones se midieron con un sensor NDIR de CO$_2$ (rango operativo
acotado a 5000 ppm, supuesto S9) y un sensor TGS2611 de CH$_4$ cuya
senal se uso solo como indicador de anoxia (S10). El protocolo consistio
en cierres de acumulacion en los dias 9, 11, 13 y 17 del ciclo: con la
panera sellada, la concentracion de CO$_2$ crece de forma casi lineal
y la tasa de produccion se estima por ajuste de la pendiente, aceptando
solo ajustes con $R^2 \geq 0.5$. Para cada dia y dieta se incluyo una
bandeja control con 250 g de alimento sin larvas; la tasa neta de
emision del tratamiento se obtiene restando la tasa del control del
mismo dia (supuesto S7). Las tasas extraidas, con su $R^2$, se
consolidan en
\texttt{datos/experimentos/datos\_finales\_PINN\_corregidos.csv}.
La biomasa larval no se peso en esta campana, lo que se declara como
limitacion y se corrige en los experimentos finales.

'''

i = s.index(ini)
s = s[:i] + nuevo + s[i:]
f.write_text(s)
print("OK: paso 2 — S2.1 Sistema experimental agregada")
