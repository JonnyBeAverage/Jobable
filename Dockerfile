FROM python:3.12.9



RUN apt-get update && \
    apt-get install -y zsh git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /Jobable-1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["zsh"]
