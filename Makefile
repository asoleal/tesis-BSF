DOCKER_IMG = tesis-bsf
CURRENT_DIR = $(CURDIR)
DOCKER_CMD = docker run --rm -v "$(CURRENT_DIR):/src" -w /src $(DOCKER_IMG)

.PHONY: all tesis slides clean help

help:
	@echo "Opciones: make tesis, make slides, make clean"

all: tesis slides

tesis:
	$(DOCKER_CMD) sh -c "pdflatex tesis.tex && biber tesis && pdflatex tesis.tex && pdflatex tesis.tex"

slides:
	$(DOCKER_CMD) xelatex presentacion.tex

clean:
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.bbl *.blg *.fls *.fdb_latexmk *.synctex.gz *.nav *.snm *.vrb *.bcf *.run.xml
	rm -f contenido/*.aux
