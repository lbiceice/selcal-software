FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e AS builder

WORKDIR /build
COPY pyproject.toml README.md LICENSE.txt Licence.txt ./
COPY src/ ./src/
COPY docs/ ./docs/
RUN python -m pip install --no-cache-dir setuptools==80.9.0 wheel==0.45.1 \
    && python -m pip wheel --no-deps --no-build-isolation --wheel-dir /wheels . \
    && python -m pip download --only-binary=:all: --no-deps --dest /wheels numpy==2.4.6

FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir --no-index --no-deps /wheels/*.whl \
    && groupadd --gid 1000 selcal \
    && useradd --uid 1000 --gid 1000 --create-home selcal \
    && mkdir /input /output \
    && chown 1000:1000 /output
COPY examples/workflow/ /opt/selcal/examples/
USER 1000:1000
WORKDIR /output
STOPSIGNAL SIGINT
ENTRYPOINT ["python", "-I", "-m", "selcal"]
CMD ["doctor"]
