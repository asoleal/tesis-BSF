# Usamos la imagen oficial completa de TeX Live (aprox. 4GB)
# Garantiza que tienes todos los paquetes: fontspec, biber, xelatex, etc.
FROM texlive/texlive:latest

# Metadatos
LABEL maintainer="jjlg"
LABEL description="Entorno para compilar Tesis BSF con XeLaTeX"

# Instalamos 'make' por si quisieras correr el make DENTRO del contenedor
# (aunque tu estrategia actual es correrlo desde fuera, esto da flexibilidad)
RUN apt-get update && apt-get install -y make && rm -rf /var/lib/apt/lists/*

# Pre-generamos la caché de fuentes para LuaLaTeX/XeLaTeX
# Esto evita que la primera compilación sea lenta o falle buscando fuentes
RUN luaotfload-tool --update

# Configuramos el directorio de trabajo
WORKDIR /src

# Comando por defecto (útil para debug si corres 'docker run -it tesis-bsf')
CMD ["/bin/bash"]
