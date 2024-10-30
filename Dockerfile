FROM python:3.12.7-slim-bookworm

ENV LANG="C.UTF-8" \
    TZ="Asia/Shanghai" \
    REPO_URL="https://github.com/TonyLiooo/OCR-Server.git" \
    WORKDIR="/app"

COPY . ${WORKDIR}

WORKDIR ${WORKDIR}

RUN apt-get update && apt-get install -y python3-distutils && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /tmp/* \
           /root/.cache \
           /var/tmp/*

WORKDIR /app

EXPOSE 8899
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8899"]