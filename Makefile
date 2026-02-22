# Variables de configuración
DOCKER_IMG  = tesis-bsf
CURRENT_DIR = $(shell pwd)
UID         = $(shell id -u)
GID         = $(shell id -g)

# Comando base de Docker
# Se usa -it para interactividad y ver errores en tiempo real
DOCKER_CMD  = docker run --rm -it --user $(UID):$(GID) -v "$(CURRENT_DIR):/src" -w /src

# Colores para mensajes (opcional, para mejor legibilidad)
YELLOW = \033[0;33m
NC     = \033[0m

.PHONY: all tesis slides articulos clean help watch-tesis watch-art

help:
	@echo "Uso del Makefile:"
	@echo "  make tesis         - Compila la tesis principal"
	@echo "  make slides        - Compila las presentaciones"
	@echo "  make articulos     - Compila todos los artículos en la carpeta articulos/"
	@echo "  make watch-tesis   - Monitoreo continuo de la tesis (auto-compilar al guardar)"
	@echo "  make watch-art DIR=nombre-carpeta - Monitoreo continuo de un artículo"
	@echo "  make clean         - Elimina todos los archivos temporales de LaTeX"

all: tesis slides articulos

# --- Reglas de Compilación Única ---

tesis:
	@echo "$(YELLOW)Compilando tesis...$(NC)"
	$(DOCKER_CMD)/tesis $(DOCKER_IMG) latexmk -xelatex -synctex=1 -interaction=nonstopmode -file-line-error -outdir=. tesis.tex

slides:
	@echo "$(YELLOW)Compilando presentación...$(NC)"
	$(DOCKER_CMD)/presentacion $(DOCKER_IMG) latexmk -xelatex -f -synctex=1 -interaction=nonstopmode -file-line-error -outdir=. presentacion.tex

articulos:
	@for art in $$(find articulos -name "main.tex"); do \
		dir=$$(dirname $$art); \
		echo "$(YELLOW)Compilando $$art en /src/$$dir$(NC)"; \
		$(DOCKER_CMD)/$$dir $(DOCKER_IMG) latexmk -C; \
		$(DOCKER_CMD)/$$dir $(DOCKER_IMG) latexmk -xelatex -synctex=1 -interaction=nonstopmode -file-line-error -outdir=. main.tex; \
	done

# --- Reglas de Monitoreo Continuo (Watch Mode) ---

watch-tesis:
	@echo "$(YELLOW)Modo watch activado para la Tesis. Presiona Ctrl+C para detener.$(NC)"
	$(DOCKER_CMD)/tesis $(DOCKER_IMG) latexmk -xelatex -pvc -interaction=nonstopmode -outdir=. tesis.tex

watch-art:
	@if [ -z "$(DIR)" ]; then \
		echo "Error: Debes especificar el directorio. Ejemplo: make watch-art DIR=04-hardware-dashboard"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Modo watch activado para $(DIR). Presiona Ctrl+C para detener.$(NC)"
	$(DOCKER_CMD)/articulos/$(DIR) $(DOCKER_IMG) latexmk -xelatex -pvc -interaction=nonstopmode -outdir=. main.tex

# --- Limpieza ---

clean:
	@echo "Limpiando archivos temporales..."
	find . -type f \( -name "*.aux" -o -name "*.log" -o -name "*.out" -o -name "*.toc" -o \
	                  -name "*.fls" -o -name "*.fdb_latexmk" -o -name "*.synctex.gz" -o \
	                  -name "*.xdv" -o -name "*.bcf" -o -name "*.run.xml" -o -name "*.bbl" -o \
	                  -name "*.blg" -o -name "*.snm" -o -name "*.nav" -o -name "*.vrb" \) -delete
	@echo "Limpieza completada."
