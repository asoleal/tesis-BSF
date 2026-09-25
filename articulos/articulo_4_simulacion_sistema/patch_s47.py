# patch_s47.py — §4.7 Escenarios + §4.8 C5 alineadas con 4.1-4.6
from pathlib import Path

f = Path("contenido_art4.tex")
s = f.read_text()

# --- R1: parrafo de Escenarios (hasta la tabla tab:cfd) ---
a1 = "\\subsection{Escenarios de puertos en fase ventilada}"
b1 = "\\begin{table}[htbp]"
assert s.count(a1) == 1 and s.count(b1) >= 1, (s.count(a1), s.count(b1))
i0 = s.index(a1); i1 = s.index(b1, i0)
nuevo1 = r'''\subsection{Escenarios de puertos en fase ventilada}

Como palanca de diseno se barrio la posicion de los puertos en fase
ventilada: C1v2 (entrada lateral baja, salida lateral alta), C2v2 (entrada
en tapa con chorro vertical) y C4v2 (entrada inclinada 45$^\circ$ con
componente tangencial); la configuracion C3 (entrada tangencial pura) quedo
descartada por el artefacto de succion documentado en la verificacion
numerica. La Figura~\ref{fig:tau_mix} (izquierda) muestra que en ningun
caso $\eta$ cruza el umbral: $\tau_{mix} > 720$ s en las tres
configuraciones, con $\eta(720\,\mathrm{s}) = 0.246/0.114/0.110$ y masa
remanente $36.5/17.2/18.5\%$ para C1v2/C2v2/C4v2 (Tabla~\ref{tab:cfd}),
es decir $\tau_{mix}/\tau_{aire} > 4$. La interpretacion es estructural:
sin inercia, el flujo va por hilos directos de la entrada a la salida y el
intercambio con las zonas muertas es difusion lenta; la geometria de
puertos, por si sola, no sostiene el supuesto A1 en fase ventilada. La
palanca real pasa a ser el regimen transicional del jet (fuera del alcance
del modelo) o la mezcla forzada interna.

'''
s = s[:i0] + nuevo1 + s[i1:]

# --- R2: parrafo de C5 (hasta su figura) ---
a2 = "\\subsection{Mezcla forzada con ventilador en tapa (C5)}"
assert s.count(a2) == 1, s.count(a2)
i0 = s.index(a2)
b2 = "\\begin{figure}[htbp]"
i1 = s.index(b2, i0)
nuevo2 = r'''\subsection{Mezcla forzada con ventilador en tapa (C5)}

Se modelo un ventilador axial pequeno (40 mm) en la tapa, centrado sobre las
larvas, como la fuente de momentum $-\chi F_0\,\hat{\mathbf{z}}$ del modelo
de flujo: la representacion inyecta el momentum del ventilador sin mallar
las aspas. El escenario de interes es la camara sellada ($Q = 0$ en ambos
puertos), que reproduce exactamente la fase de cierre del protocolo. La
Figura~\ref{fig:tau_mix} (derecha) muestra el decaimiento de $\eta$:
$\tau_{mix} = 4$ s con $F_0 = 400$ N/m$^3$ y $\tau_{mix} = 24$ s con
$F_0 = 4$ N/m$^3$. La mezcla satura: para cualquier fuerza significativa la
camara se homogeniza en menos de 60 s, uno o dos ordenes de magnitud por
debajo de los cierres de 10--30 min del protocolo. Con la linealidad exacta
verificada en la verificacion numerica, estos tiempos se extrapolan a
cualquier fuerza sin nuevas simulaciones; y como el regimen real del
ventilador es transicional-turbulento, son cotas conservadoras de la mezcla
real.

'''
s = s[:i0] + nuevo2 + s[i1:]

f.write_text(s)
print("OK: S4.7 y S4.8 reescritas")
