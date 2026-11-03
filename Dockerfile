# syntax=docker/dockerfile:1.7
# BCDC API image: serves `bcdc.api` (FastAPI) on port 8000.
#
#   docker build -t bcdc-api .
#   docker run --rm -p 8000:8000 -v "$PWD/deploy:/bundle:ro" bcdc-api
#
# The deployment bundle (made with `bcdc calibrate`) is mounted at /bundle at run time,
# so one image serves any trained model.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# 1) Dependencies only (cached until pyproject.toml changes). A stub package lets pip resolve
#    the project's dependencies before the real code is copied.
#    Behind a TLS-intercepting proxy, pass its CA with: --secret id=pip_cert,src=/path/to/ca.pem
#    Versions are pinned by constraints-docker.txt for reproducible images.
COPY pyproject.toml Readme.md LICENSE constraints-docker.txt ./
RUN --mount=type=secret,id=pip_cert,required=false \
    if [ -f /run/secrets/pip_cert ]; then export PIP_CERT=/run/secrets/pip_cert; fi && \
    mkdir -p bcdc && touch bcdc/__init__.py && \
    pip install --constraint constraints-docker.txt ".[api,dicom,edge]" && \
    pip uninstall -y bcdc && rm -rf bcdc build

# 2) The package itself.
COPY bcdc ./bcdc
RUN pip install --no-deps . && rm -rf build

RUN useradd --create-home --uid 10001 bcdc
USER 10001

ENV BCDC_BUNDLE=/bundle
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD ["python", "-c", "import urllib.request, sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status != 200)"]

CMD ["uvicorn", "bcdc.api:app_from_env", "--factory", "--host", "0.0.0.0", "--port", "8000"]
