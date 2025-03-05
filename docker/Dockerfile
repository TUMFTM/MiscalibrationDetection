# Use an official CUDA base image from the NVIDIA repository
FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

# Install Python and other necessary dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    python3-setuptools \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the default Python version to 3.10
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install Python packages
RUN pip install --upgrade pip
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["sh", "-c", "python ./src/main.py --save_name resnet18_all_aug_241106 --classifier"]
