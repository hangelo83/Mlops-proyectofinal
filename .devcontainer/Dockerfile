FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm

# La imagen base trae el repositorio apt de Yarn con una llave GPG vencida:
# apt-get update falla y con él la instalación de docker-in-docker.
RUN rm -f /etc/apt/sources.list.d/yarn.list
