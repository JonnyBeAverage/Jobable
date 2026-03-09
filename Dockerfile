FROM python:3.12.9



RUN apt-get update && \
    apt-get install -y zsh git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /Jobable-1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY jobable/ml_logic jobable/ml_logic
COPY jobable/data jobable/data


EXPOSE 8000

CMD ["zsh"]
