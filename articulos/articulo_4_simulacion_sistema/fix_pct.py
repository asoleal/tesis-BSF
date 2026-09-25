from pathlib import Path
p = Path("contenido_art4.tex")
s = p.read_text()
n = s.count("\\,\\%")
p.write_text(s.replace("\\,\\%", "\\%"))
print("reemplazos:", n)
