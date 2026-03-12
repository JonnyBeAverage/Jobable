FROM python:3.12.9

RUN apt-get update && \
    apt-get install -y git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /Jobable-1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY jobable jobable

EXPOSE 8000

CMD ["sh", "-c", "uvicorn jobable.api:app --host 0.0.0.0 --port $PORT"]
