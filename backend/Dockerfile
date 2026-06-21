FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads weights

EXPOSE 7860

ENV PORT=7860
ENV FLASK_DEBUG=false
ENV MODEL_PATH=weights/pneumonia_model.tflite

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:7860", "--timeout", "120", "app:app"]
