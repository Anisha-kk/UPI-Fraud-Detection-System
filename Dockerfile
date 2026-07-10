FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
EXPOSE 8501

COPY scripts/start.sh /start.sh

RUN sed -i 's/\r$//' /start.sh && chmod +x /start.sh

ENTRYPOINT ["/start.sh"]