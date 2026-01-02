FROM texlive/texlive:latest

# Instalar paquetes necesarios
RUN tlmgr install natbib babel-english babel-spanish beamer polyglossia xetex

# Listo para compilar
