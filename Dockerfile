FROM python:3.11

# Set the working directory
WORKDIR /code

# Copy requirements and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port Flask is running on
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]
