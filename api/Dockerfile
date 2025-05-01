# Use the official Python image
FROM python:3.10

# Set working directory
WORKDIR /api

# Fix the COPY path here
COPY api/app/requirements.txt /api/app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /api/app/requirements.txt

# Copy entire project
COPY . /app

# Set correct directory for training
WORKDIR /app

# Run training script
RUN python train.py

# Copy trained model to the app folder
# RUN cp -r /app/xgboost-model.pkl /app/api/app/

# Set working directory to API app
WORKDIR /app/api/app

EXPOSE 8000
EXPOSE 8001

# Run the FastAPI app
CMD ["python", "app.py"]
