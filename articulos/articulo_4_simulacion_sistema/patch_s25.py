# patch_s25.py — reescribe §5 Resultados y discusion con los resultados nuevos
from pathlib import Path

f = Path("contenido_art4.tex")
s = f.read_text()

ini = "\\section{Resultados y discusion}"
fin = "\\section{Conclusiones}"
assert s.count(ini) == 1 and s.count(fin) == 1
i0, i1 = s.index(ini), s.index(fin)
assert i0 < i1

nuevo = r'''\section{Resultados y discusion}

En la dinamica del ciclo, el calendario adaptativo de dos pasadas cumple el
tope del NDIR en los dos escenarios extremos: picos de CO$_2$ de 4978 ppm
(E2) y 4604 ppm (E5), con cierres que se acortan desde 30 min al inicio del
ciclo hasta 2.5--7 min en el pico de emision y vuelven a alargarse en la
fase de ayuno. La ventilacion de 4 L/min entre cierres mantiene la base
entre 500 y 1600 ppm, condicion necesaria: con 1 L/min la base supera los
4300 ppm en el pico y ningun cierre es viable. Las emisiones acumuladas del
ciclo son 0.59 mol de CO$_2$ (E2, $N_0 = 400$) y 1.04 mol (E5, $N_0 = 700$).

El switch de prepupa (S5) tiene un efecto cuantitativo decisivo: sin el, el
modelo acumula lipidos de forma ficticia hasta 209 mg/larva y las emisiones
de E2 alcanzan 1.37 mol; con el switch, la biomasa lipidica final es de
54 mg/larva y las emisiones caen a 0.59 mol. El efecto es directamente
relevante para el inventario de GEI del sistema, pues fija la fraccion de
carbono que se emite como CO$_2$ frente a la que se retiene en biomasa de
prepupas.

La combinacion de resultados fija las condiciones de validez del protocolo de
medicion. En fase ventilada, el gemelo digital 0D proporciona tasas
instantaneas de emision, pero el CFD muestra que el supuesto A1 no se
sostiene con ventilacion por puertos en el regimen analizado: las lecturas
del sensor promedian una camara con zonas muertas, con sesgo hacia las
concentraciones mas altas del entorno del lecho (en C1v2 el 36.5\% del
trazador permanece en la camara a 720 s). Por ello las tasas de fase
ventilada deben tratarse como estimaciones acotadas, no como mediciones de
precision. En fase cerrada, en cambio, el ventilador en tapa garantiza la
homogeneidad en menos de 60 s frente a cierres de 10--30 min: el supuesto A1
queda validado con un margen de 10--30$\times$, y la pendiente de
acumulacion durante el cierre es una medicion de tasa directa y defendible.
El diseno final que emerge es: (i) ventilador fijo (no controlado en
velocidad; la mezcla saturada hace innecesario el control) de material de
baja emision en la tapa centrado sobre las larvas; (ii) medicion del caudal
efectivo de los puertos y de la presion interna de la camara (deteccion de
fugas y correccion); (iii) cierres estaticos para CH$_4$ y validacion cruzada
con la fase ventilada; y (iv) correccion del volumen muerto del gemelo
digital a partir de la fraccion de mezcla observada. La linealidad de Stokes,
verificada en C5, da ademas una ley de diseno sin nuevas simulaciones:
$\tau_{mix}$ escala inversamente con el momentum inyectado.

La validacion preliminar de la Seccion~\ref{sec:validacion} situa el modelo
en el orden de magnitud correcto y reproduce la caida de emisiones hacia el
ayuno de prepupa, aunque sobreestima el pico del ciclo medio; las causas
estan identificadas y delimitan el trabajo pendiente antes de los
experimentos finales: verificacion de los parametros DEB con biomasa medida,
recalibracion del modulo microbiano frente a las curvas de control,
calibracion de $V_{aire}$ por trazador y calibracion cuantitativa de CH$_4$.

'''

s = s[:i0] + nuevo + s[i1:]
f.write_text(s)
print("OK: S5 reescrita,", len(nuevo.splitlines()), "lineas nuevas")
