FROM nvcr.io/nvidia/pytorch:24.06-py3

ENV PATH=/opt/conda/bin:$PATH

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o ~/miniconda.sh \
    && sh ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh

# Install package manager
RUN pip install uv

# Install required Python packages
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# Create output directory
RUN mkdir -p output

# We'll mount the code as a volume instead of copying it
# CMD ["python", "app2.py"]
CMD ["python", "dcaecore/prune_vae.py", "--config", "dcaecore/config_tiny.yaml", "--output_dir", "dcaecore/TAESD_outputs", "--gpu", "0"]