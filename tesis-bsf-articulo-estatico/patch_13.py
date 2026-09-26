# patch_13.py - protocolo de cierres en sec. 2.1 + tabla de parametros sincronizada
from pathlib import Path
p = Path("secciones/metodos.tex"); s = p.read_text()

ancla = "\\subsection{Modelo dinamico"
i = s.index(ancla)
parrafo = ("En cada dia de medicion los cierres se realizaron de forma secuencial por dieta: "
"primero un recipiente hasta la saturacion del sensor de CO$_2$ (6000 ppm) y luego el otro, "
"con el mismo procedimiento. La tasa de cada cierre se estimo por ajuste lineal del tramo "
"previo a la saturacion; cuando el ajuste fue de baja calidad o la curva alcanzo el techo "
"del sensor (cierres marcados como saturados), la tasa estimada es conservadora y se trato "
"como cota inferior en la validacion (sec. \\ref{sec:resultados}).\n\n")
s = s[:i] + parrafo + s[i:]

old1 = "Dia central de prepupa & $t_p$ & 12.0 & d \\\\"
new1 = "Dia central de prepupa & $t_p$ & 12.3 (D1), 11.7 (D4) & d \\\\"
assert s.count(old1) == 1; s = s.replace(old1, new1)

old2 = "Ancho del interruptor & $w_p$ & 1.0 & d \\\\"
new2 = "Ancho del interruptor & $w_p$ & 0.48 & d \\\\"
assert s.count(old2) == 1; s = s.replace(old2, new2)

old3 = "Suelo de mantenimiento & $m_{\\mathrm{suelo}}$ & 0.13 & -- \\\\"
new3 = ("Suelo de mantenimiento & $m_{\\mathrm{suelo}}$ & 0.13 & -- \\\\\n"
        "Biomasa inicial por larva & $B_0$ & 0.0175 & mg \\\\\n"
        "Factor de calidad de dieta & $f$ & 1 (D1), 0.817 (D4) & -- \\\\")
assert s.count(old3) == 1; s = s.replace(old3, new3)

old4 = ("\\caption{Parametros del modelo. Nucleo DEB: literatura (ver texto); interruptor de "
"prepupa y camara: este trabajo; modulo microbiano: valores iniciales, recalibrados con las "
"curvas de control.}")
new4 = ("\\caption{Parametros del modelo. Nucleo DEB: literatura (ver texto). Interruptor de "
"prepupa, $B_0$ y $f$: calibrados en este trabajo con las tasas netas (sec. "
"\\ref{sec:resultados}). Modulo microbiano: $k_{\\mathrm{ref}}$ e $Y_{CO_2}$ calibrados con "
"los controles (sec. \\ref{sec:resultados}); demas parametros fijos. Camara: mediciones del "
"experimento.}")
assert s.count(old4) == 1; s = s.replace(old4, new4)

p.write_text(s)
print("patch_13 aplicado: protocolo en 2.1 + tabla sincronizada")
