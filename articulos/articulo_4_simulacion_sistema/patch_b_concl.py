from pathlib import Path
f = Path("contenido_art4.tex")
s = f.read_text()

# R1: item 1 (version vieja "dia a dia")
old1 = (r"  \item El gemelo digital 0D integra la dinamica larvaria, el lecho y el" "\n"
        r"  aire interno, y reproduce los escenarios E2--E5 del protocolo con control" "\n"
        r"  de O$_2$ por ventilacion y cierres adaptativos de duracion dia a dia.")
new1 = (r"  \item El gemelo digital 0D integra la dinamica larvaria, el lecho y el" "\n"
        r"  aire interno, y reproduce los escenarios E2--E5 del protocolo con control" "\n"
        r"  de O$_2$ por ventilacion y un calendario de cierres adaptativo de dos" "\n"
        r"  pasadas que respeta el tope del NDIR (picos de 4978 y 4604 ppm en E2 y" "\n"
        r"  E5).")
assert s.count(old1) == 1, s.count(old1)
s = s.replace(old1, new1)

# R2: items 6 y 7 antes del \end{enumerate}
assert s.count("\\end{enumerate}") == 1, s.count("\\end{enumerate}")
items67 = (r"  \item El switch de prepupa, motivado por la medicion de ayuno del dia 17" "\n"
           r"  (14.3 ppm/min), elimina la acumulacion ficticia de lipidos (209 a" "\n"
           r"  54 mg/larva) y reduce las emisiones acumuladas de CO$_2$ de 1.37 a" "\n"
           r"  0.59 mol en E2, con efecto directo sobre el inventario de GEI del" "\n"
           r"  sistema." "\n"
           r"  \item La validacion preliminar contra el experimento de septiembre de" "\n"
           r"  2025 (panera de 12.4 L, $N = 700$) situa el modelo en el orden de" "\n"
           r"  magnitud correcto y reproduce la caida de emisiones del ayuno de" "\n"
           r"  prepupa; la sobreestimacion del pico medio queda atribuida a los" "\n"
           r"  parametros DEB sin verificar y delimita la calibracion pendiente." "\n"
           r"\end{enumerate}")
s = s.replace("\\end{enumerate}", items67)

f.write_text(s)
print("OK: Conclusiones — item 1 corregido, items 6 y 7 agregados")
