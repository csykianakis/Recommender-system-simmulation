# Use the official Python 3.11 slim image as base
FROM python:3.11-slim

# Install gcc and python3-dev
RUN apt-get update && apt-get install -y gcc python3-dev

# Set the working directory in the container
WORKDIR /app

# Copy the entire contents of the "app" directory into the container at /app
COPY ./app .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8002
EXPOSE 8002

# Command to run the application
CMD ["uvicorn", "fastapi_1:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]
