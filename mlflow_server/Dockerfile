# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 for the MLflow server
EXPOSE 5000

# Define environment variable for the tracking URI (optional)
ENV MLFLOW_TRACKING_URI http://0.0.0.0:5000

# Run mlflow server command
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]