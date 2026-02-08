DOCKER_IMG = tesis-bsf
CURRENT_DIR = $(shell pwd)
# Capturamos tu usuario local para que Docker no cree archivos como 'root'
UID = $(shell id -u)
GID = $(shell id -g)

# --user $(UID):$(GID) es la clave para evitar problemas de permisos
DOCKER_CMD = docker run --rm --user $(UID):$(GID) -v "$(CURRENT_DIR):/src" -w /src $(DOCKER_IMG)

.PHONY: all tesis slides clean help

help:
	@echo "Opciones: make tesis, make slides, make clean"

all: tesis slides

tesis:
	# Usamos latexmk con la opción -xelatex.
	# -interaction=nonstopmode evita que se congele si hay un error.
	$(DOCKER_CMD) latexmk -xelatex -synctex=1 -interaction=nonstopmode -file-line-error -outdir=. tesis.tex

slides:
	$(DOCKER_CMD) latexmk -xelatex -f -synctex=1 -interaction=nonstopmode -file-line-error -outdir=. presentacion.tex

clean:
	# latexmk tiene su propio comando de limpieza (-C limpia todo, -c solo temporales)
	# Pero mantenemos tu limpieza manual para asegurar
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.bbl *.blg *.fls *.fdb_latexmk *.synctex.gz *.nav *.snm *.vrb *.bcf *.run.xml *.xdv
	rm -f contenido/*.aux
	# Borrar PDFs también si quieres un clean total
	# rm -f tesis.pdf presentacion.pdf
