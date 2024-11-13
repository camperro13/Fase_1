# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for DVC and other tools
RUN apt-get update && apt-get install -y \
    git \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up Git user for DVC
RUN git config --global user.name "camperro13" && \
    git config --global user.email "jc.trucha13@gmail.com"

# Copy the requirements file to the container
COPY requirements.txt .

# Install Python dependencies (DVC and any other libraries)
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the entire project directory to the container
COPY . .

# Expose any ports if required (for example, for a web service)
EXPOSE 8080

# Command to run when the container starts, running the DVC experiment
CMD ["dvc", "exp", "run"]