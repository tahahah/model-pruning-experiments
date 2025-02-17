# Build the Docker image
docker build -t vae-pruning .

# Run the container with GPU support and volume mounting
docker run --gpus all -v ${PWD}:/app -v ${PWD}/.cache/torch:/root/.cache/torch -p 7860:7860 vae-pruning
