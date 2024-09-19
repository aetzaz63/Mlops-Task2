FROM python:3.12.5

# Set the working directory in the container
WORKDIR /app

# Copy all files from the current directory to /app in the container
COPY . .

# Install the required Python packages
RUN pip3 install -r requirements.txt

# Specify the command to run your app when the container starts
CMD ["python", "app.py"]
