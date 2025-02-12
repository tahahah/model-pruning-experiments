FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install git and other dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install required Python packages
RUN pip install torch_pruning pandas
RUN pip install git+https://github.com/mit-han-lab/efficientvit.git

# Set working directory
WORKDIR /app

# Create output directory
RUN mkdir -p output

# We'll mount the code as a volume instead of copying it
CMD ["python", "vae_pruning_analysis.py"]
