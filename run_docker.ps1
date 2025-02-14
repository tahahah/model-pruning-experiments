# Build the Docker image
docker build -t vae-pruning .

# Run the container with GPU support and volume mounting
docker run --gpus all `
    -v ${PWD}:/app `
    vae-pruning
