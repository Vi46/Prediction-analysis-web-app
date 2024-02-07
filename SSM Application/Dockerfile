# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Copy the required model and vectorizer files into the container
COPY SaAnalysisCodeUsingRSynthetic.pkl /app
COPY TfidfVectorizerRSynthetic.pkl /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV NAME prediction

# Run app.py when the container launches
CMD ["python", "prediction.py"]