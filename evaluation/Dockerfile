FROM python:3.10-slim

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --progress-bar off -r /tmp/requirements.txt

COPY --chown=user:user ground-truth /opt/app/ground-truth
COPY --chown=user:user evaluation.py /opt/app/
COPY --chown=user:user misc.py /opt/app/

ENTRYPOINT [ "python", "-m", "evaluation" ]


