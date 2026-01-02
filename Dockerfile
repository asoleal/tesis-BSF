FROM texlive/texlive:latest

WORKDIR /src

COPY . .

RUN pdflatex -interaction=nonstopmode -shell-escape 0000.tex && \
    bibtex 0000 && \
    pdflatex -interaction=nonstopmode -shell-escape 0000.tex && \
    pdflatex -interaction=nonstopmode -shell-escape 0000.tex && \
    xelatex -interaction=nonstopmode presentacion.tex
