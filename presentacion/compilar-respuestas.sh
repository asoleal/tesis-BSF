#!/usr/bin/env bash
set -euo pipefail

IMAGE="ghcr.io/asoleal/tesis-bsf-base:sha-7c75eab"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

run_in_container() {
  docker run --rm -it \
    --network none \
    --user "$(id -u):$(id -g)" \
    --mount type=bind,src="${ROOT_DIR}",target=/work \
    -w /work \
    "${IMAGE}" \
    bash -lc "$1"
}

usage() {
  cat <<'UsoEOF'
Uso:
  scripts/compilar-respuestas.sh shell
  scripts/compilar-respuestas.sh respuestas
  scripts/compilar-respuestas.sh clean

Ejemplos:
  scripts/compilar-respuestas.sh respuestas
  scripts/compilar-respuestas.sh clean
UsoEOF
}

cmd="${1:-}"

case "${cmd}" in
  shell)
    docker run --rm -it \
      --network none \
      --user "$(id -u):$(id -g)" \
      --mount type=bind,src="${ROOT_DIR}",target=/work \
      -w /work \
      "${IMAGE}" \
      bash
    ;;
  respuestas)
    run_in_container "cd presentacion && latexmk -xelatex -synctex=1 -interaction=nonstopmode presentacion_respuestas_completas.tex"
    ;;
  clean)
    run_in_container "cd presentacion && latexmk -C presentacion_respuestas_completas.tex"
    ;;
  *)
    usage
    exit 1
    ;;
esac
