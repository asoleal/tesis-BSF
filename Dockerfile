FROM texlive/texlive:latest

LABEL maintainer="jjlg"
LABEL org.opencontainers.image.title="tesis-bsf-base"
LABEL org.opencontainers.image.description="Entorno reproducible para compilar la tesis BSF con XeLaTeX, Biber y latexmk"
LABEL org.opencontainers.image.source="https://github.com/asoleal/tesis-BSF"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      make \
      fontconfig \
      locales \
      fonts-texgyre \
      fonts-lmodern && \
    rm -rf /var/lib/apt/lists/*

RUN sed -i '/es_CO.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen

ENV LANG=es_CO.UTF-8
ENV LANGUAGE=es_CO:es
ENV LC_ALL=es_CO.UTF-8

RUN fc-cache -fv

WORKDIR /src

CMD ["/bin/bash"]
