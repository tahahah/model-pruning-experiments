# Build the Docker image
docker build -t vae-pruning .

# Run the container with GPU support and volume mounting
docker run --gpus all `
    -v ${PWD}/output:/app/output `
    -v ${PWD}/images:/app/images `
    -v ${PWD}/vae_pruning_analysis.py:/app/vae_pruning_analysis.py `
    vae-pruning
