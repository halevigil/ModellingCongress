# Dockerfile for building flask app
FROM python:3.11-slim

WORKDIR .

# Copy requirements and install dependencies
COPY flask-app .
COPY modellingcongress modellingcongress
COPY outputs/preprocess0/inference outputs/preprocess6/inference inference
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
# COPY . .

# Expose the port
EXPOSE 8080

# Use Gunicorn to serve the app
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]