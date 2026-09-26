# patch_mayor1.py — mayor #1: reencuadre CFD (diseno) + conveccion natural
# panera + sensores MH-410/MH-440D + sin menciones de experimentos futuros
from pathlib import Path

f = Path("contenido_art4.tex")
s = f.read_text()

# 1) §2.1 sensores: TGS2611 -> MH-440D, CO2 -> MH-410
old = ("Las emisiones se midieron con un sensor NDIR de CO$_2$ (rango operativo\n"
       "acotado a 5000 ppm, supuesto S9) y un sensor TGS2611 de CH$_4$ cuya\n"
       "senal se uso solo como indicador de anoxia (S10).")
new = ("Las emisiones se midieron con un sensor NDIR MH-410 de CO$_2$ (rango\n"
       "operativo 0--5000 ppm, supuesto S9) y un sensor NDIR MH-440D de\n"
       "CH$_4$, cuya senal se uso como indicador de anoxia por no haberse\n"
       "calibrado contra una referencia (S10).")
assert s.count(old) == 1, "sensores"
s = s.replace(old, new)

# 1b) §2.1 biomasa: quitar "se corrige en los experimentos finales"
old = ("La biomasa larval no se peso en esta campana, lo que se declara como\n"
       "limitacion y se corrige en los experimentos finales.")
new = ("La biomasa larval no se peso en esta campana, lo que se declara como\n"
       "limitacion.")
assert s.count(old) == 1, "biomasa futuro"
s = s.replace(old, new)

# 2) §1: Figura 1 como camara instrumentada de estudio CFD
old = "Figura~\\ref{fig:esquema} muestra el sistema."
new = ("Figura~\\ref{fig:esquema} muestra la camara instrumentada cuyo\n"
       "diseno se verifica en la Seccion 2.4.")
assert s.count(old) == 1, "fig1 intro"
s = s.replace(old, new)

# 3) §2.4.1: doble objeto (camara de estudio vs panera)
old = ("El objetivo de esta seccion es verificar ese supuesto para el\n"
       "diseno de la camara y del protocolo.")
new = ("El objetivo de esta seccion es verificar ese supuesto para la\n"
       "camara instrumentada representativa a escala de contenedor de la\n"
       "Figura~\\ref{fig:esquema}; el experimento preliminar de la\n"
       "Seccion 2.1 (panera sellada, sin ventilador) se examina aparte\n"
       "en la Seccion 4.")
assert s.count(old) == 1, "objetivo 2.4.1"
s = s.replace(old, new)

# 4) §2.4.2: geometria representativa
old = ("La camara es un contenedor de $26 \\times 19 \\times 18$ cm\n"
       "($\\approx 8.9$ L geometricos) empleado para conservar la temperatura del\n"
       "lote; el volumen de aire efectivo sobre el lecho de cria es\n"
       "$V_{aire} = 2.5$ L.")
new = ("La camara de estudio es una caja de $26 \\times 19 \\times 18$ cm\n"
       "($\\approx 8.9$ L geometricos), geometria representativa de una\n"
       "camara instrumentada a escala de contenedor\n"
       "(Figura~\\ref{fig:esquema}); el volumen de aire efectivo sobre el\n"
       "lecho de cria es $V_{aire} = 2.5$ L.")
assert s.count(old) == 1, "geometria 2.4.2"
s = s.replace(old, new)

# 5) §3.3 C5: tiempos son del diseno estudiado
old = ("que reproduce exactamente la fase de cierre del protocolo. La\n"
       "Figura~\\ref{fig:tau_mix} (derecha) muestra")
new = ("que reproduce exactamente la fase de cierre del protocolo; los\n"
       "tiempos que se reportan son propiedades de este diseno. La\n"
       "Figura~\\ref{fig:tau_mix} (derecha) muestra")
assert s.count(old) == 1, "C5 diseno"
s = s.replace(old, new)

# 6) §3.4: cierres estaticos, A1 por conveccion natural
old = ("cierres de acumulacion con bandeja control del mismo dia, supuesto S7). Las tasas extraidas, con su R$^2$ de ajuste, estan en")
new = ("cierres de acumulacion con bandeja control del mismo dia, supuesto S7).\n"
       "Los cierres fueron estaticos, sin ventilador: el supuesto A1 se apoyo\n"
       "en la conveccion natural del lecho caliente y en la linealidad de la\n"
       "acumulacion (Seccion 4). Las tasas extraidas, con su R$^2$ de ajuste, estan en")
assert s.count(old) == 1, "3.4 estaticos"
s = s.replace(old, new)

# 7) §3.4: "experimentos finales" -> biomasa pesada
old = "queda sujeta a la calibracion con los experimentos finales."
new = "queda sujeta a calibracion experimental con biomasa pesada."
assert s.count(old) == 1, "3.4 calibracion"
s = s.replace(old, new)

# 8) §4: parrafo de conveccion natural tras el parrafo 1
old = ("correccion del volumen muerto del gemelo digital a partir de la fraccion\n"
       "de mezcla observada.\n")
new = ("correccion del volumen muerto del gemelo digital a partir de la fraccion\n"
       "de mezcla observada.\n\n"
       "El experimento preliminar, en cambio, se realizo en una panera sellada\n"
       "y estatica, sin ventilador: alli la mezcla descansa en la conveccion\n"
       "natural. El lecho metabolico se mantiene 1--2~$^{\\circ}$C por encima\n"
       "del aire (Figura~\\ref{fig:escenarios}c), lo que con una altura de\n"
       "0.17 m da un numero de Rayleigh Ra~$= g\\beta\\Delta T L^{3}/(\\nu\\alpha)\n"
       "\\approx 10^{6}$, mas de dos ordenes de magnitud por encima del umbral\n"
       "convectivo: la celda de conveccion recircula el volumen de aire en\n"
       "decenas de segundos, muy por debajo de la duracion de los cierres\n"
       "(2--30 min). La linealidad de la acumulacion observada en los cierres\n"
       "es evidencia empirica consistente con esa mezcla. En la panera, por\n"
       "tanto, A1 se sostiene por conveccion natural; la camara instrumentada\n"
       "de la Figura~\\ref{fig:esquema} anade el ventilador como garantia\n"
       "cuando el gradiente termico no basta o la escala crece.\n")
assert s.count(old) == 1, "conveccion natural"
s = s.replace(old, new)

# 9) §4: prepupa, quitar "trabajo de factores de emision ... habilita"
old = ("magnitud central para\n"
       "el trabajo de factores de emision que esta instrumentacion habilita.")
new = "magnitud central para la contabilidad de carbono del sistema."
assert s.count(old) == 1, "habilita"
s = s.replace(old, new)

# 10) §4 limitaciones: TGS2611 -> MH-440D; quitar campana final
old = ("y\n"
       "el CH$_4$ del TGS2611 es indicativo, con cuantificacion pendiente contra\n"
       "una referencia. Del lado del modelo: los parametros DEB no han podido\n"
       "verificarse sin biomasa pesada y el modulo microbiano esta sin\n"
       "recalibrar (S8), lo que explica la sobreestimacion del pico medio de la\n"
       "Seccion~\\ref{sec:validacion}; ambas quedan resueltas con la campana\n"
       "experimental final, que ademas pesara biomasa cada dos dias. Con esa\n"
       "campana, el sistema validado aqui pasa de demostracion a produccion de\n"
       "factores de emision por dieta.")
new = ("y\n"
       "el CH$_4$ del MH-440D es indicativo, con cuantificacion pendiente contra\n"
       "una referencia. Del lado del modelo: los parametros DEB no han podido\n"
       "verificarse sin biomasa pesada y el modulo microbiano esta sin\n"
       "recalibrar (S8), lo que explica la sobreestimacion del pico medio de la\n"
       "Seccion~\\ref{sec:validacion}; ambas limitaciones delimitan la\n"
       "calibracion experimental pendiente.")
assert s.count(old) == 1, "limitaciones"
s = s.replace(old, new)

# 11) §4 PINN: TGS2611 -> MH-440D; quitar experimentos finales
old = ("cuantificacion de CH$_4$ a\n"
       "partir de la senal no calibrada del TGS2611.")
new = ("cuantificacion de CH$_4$ a\n"
       "partir de la senal no calibrada del MH-440D.")
assert s.count(old) == 1, "pinn tgs"
s = s.replace(old, new)

old = ("El entrenamiento definitivo, con datos de\n"
       "literatura digitalizados y los experimentos finales, se reportara por\n"
       "separado.")
new = "El entrenamiento definitivo se reportara por\nseparado."
assert s.count(old) == 1, "pinn futuro"
s = s.replace(old, new)

# 12) Resumen: (ii) camara instrumentada + cierre sin campana
old = ("recupera con un ventilador pequeno en la tapa de la camara sellada\n"
       "($\\tau_{mix} = 4$--$24$ s, margen de 10--30$\\times$ sobre los cierres); y")
new = ("recupera en el diseno de una camara instrumentada con un ventilador\n"
       "pequeno en la tapa ($\\tau_{mix} = 4$--$24$ s, margen de 10--30$\\times$\n"
       "sobre los cierres); y")
assert s.count(old) == 1, "resumen ii"
s = s.replace(old, new)

old = ("preliminar de cria (panera de 12.4 L, 700 larvas). El sistema queda listo\n"
       "para la campana de factores de emision.")
new = ("preliminar de cria (panera de 12.4 L, 700 larvas). El resultado es un\n"
       "instrumento de medicion de tasas de emision con condiciones de validez\n"
       "explicitas para la cria intensiva de BSF en contenedor.")
assert s.count(old) == 1, "resumen cierre"
s = s.replace(old, new)

# 13) Conclusiones: item 3 camara instrumentada; item 5 sin promesa
old = ("  \\item Un ventilador pequeno en la tapa homogeneiza la camara sellada en\n"
       "  menos de 60 s")
new = ("  \\item Un ventilador pequeno en la tapa homogeneiza la camara\n"
       "  instrumentada sellada del diseno estudiado en menos de 60 s")
assert s.count(old) == 1, "concl 3"
s = s.replace(old, new)

old = ("la discrepancia se cerrara con la calibracion\n"
       "  experimental con dos sensores a distinta altura y con la correccion de volumen\n"
       "  muerto del gemelo digital.")
new = ("su cuantificacion requiere calibracion\n"
       "  experimental con dos sensores a distinta altura y la correccion de volumen\n"
       "  muerto del gemelo digital.")
assert s.count(old) == 1, "concl 5"
s = s.replace(old, new)

# 14) Conclusion item 8: experimentos finales -> trabajo futuro
old = ("observados; su entrenamiento definitivo queda para los\n"
       "  experimentos finales.")
new = ("observados; su entrenamiento definitivo queda como trabajo futuro.")
assert s.count(old) == 1, "concl 8"
s = s.replace(old, new)

f.write_text(s)

# 15) supuestos.md: S10 sensor real
f2 = Path("supuestos.md")
t = f2.read_text()
old = "- S10 CH4: TGS2611 no cuantitativo -> indicador solamente; NO entra al ajuste ni al PINN"
new = "- S10 CH4: MH-440D (NDIR Winsen) no calibrado -> indicador solamente; NO entra al ajuste ni al PINN"
assert t.count(old) == 1, "S10"
t = t.replace(old, new)
f2.write_text(t)

print("OK: mayor #1 — 15 cambios (reencuadre CFD + conveccion + sensores + sin futuro)")
