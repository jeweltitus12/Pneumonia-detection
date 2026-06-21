# Use an official lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /code

# Copy requirements first to leverage Docker cache
COPY ./requirements.txt /code/requirements.txt

# Install dependencies (using --no-cache-dir keeps the image smaller)
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Create a non-root user for security (required by Hugging Face)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy all backend files into the container
COPY --chown=user . $HOME/app

# Hugging Face Spaces absolutely requires port 7860
EXPOSE 7860

# Start your Flask app using Gunicorn or python command 
# (Make sure your app runs on host 0.0.0.0 and port 7860)
CMD ["python", "app.py"]
