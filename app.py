import os
import gradio as gr
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from model_manager import ModelManager

class PruningUI:
    def __init__(self, config_path: str = "dcaecore/config.yaml", save_dir: str = "output"):
        """Initialize the Pruning UI with ModelManager"""
        self.model_manager = ModelManager(config_path, save_dir)
        self.save_dir = save_dir
        
    def load_model(self, model_path: str) -> tuple:
        """Load a model and return its metrics"""
        try:
            metrics = self.model_manager.load_initial_model(model_path)
            return (
                f"Model loaded successfully!\n\nMetrics:\n"
                f"Total Parameters: {metrics['total_params']:,}\n"
                f"Non-zero Parameters: {metrics['nonzero_params']:,}\n"
                f"Sparsity Ratio: {metrics['sparsity_ratio']:.2%}",
                os.path.join(self.save_dir, "reconstructions_initial.png"),
                os.path.join(self.save_dir, "weight_dist_initial.png")
            )
        except Exception as e:
            return str(e), None, None
            
    def prune_model(self, sparsity: float) -> tuple:
        """Prune the model to specified sparsity"""
        try:
            # Create experimental model if needed
            if self.model_manager.experimental_model is None:
                self.model_manager.create_experimental_model()
                
            # Prune the model
            metrics = self.model_manager.prune_experimental_model(sparsity)
            
            return (
                f"Model pruned successfully!\n\nMetrics:\n"
                f"Total Parameters: {metrics['total_params']:,}\n"
                f"Non-zero Parameters: {metrics['nonzero_params']:,}\n"
                f"Sparsity Ratio: {metrics['sparsity_ratio']:.2%}",
                os.path.join(self.save_dir, "reconstructions_after_pruning.png"),
                os.path.join(self.save_dir, "weight_dist_after_pruning.png")
            )
        except Exception as e:
            return str(e), None, None
            
    def train_model(self, epochs: int, steps: int) -> tuple:
        """Train the experimental model"""
        try:
            metrics = self.model_manager.train_experimental_model(epochs, steps)
            
            return (
                f"Model trained successfully!\n\nMetrics:\n"
                f"Total Parameters: {metrics['total_params']:,}\n"
                f"Non-zero Parameters: {metrics['nonzero_params']:,}\n"
                f"Sparsity Ratio: {metrics['sparsity_ratio']:.2%}",
                os.path.join(self.save_dir, "reconstructions_after_training.png"),
                os.path.join(self.save_dir, "weight_dist_after_training.png")
            )
        except Exception as e:
            return str(e), None, None
            
    def equip_model(self) -> tuple:
        """Equip the experimental model"""
        try:
            metrics = self.model_manager.equip_experimental_model()
            
            return (
                f"Model equipped successfully!\n\nMetrics:\n"
                f"Total Parameters: {metrics['total_params']:,}\n"
                f"Non-zero Parameters: {metrics['nonzero_params']:,}\n"
                f"Sparsity Ratio: {metrics['sparsity_ratio']:.2%}",
                os.path.join(self.save_dir, "reconstructions_equipped.png"),
                os.path.join(self.save_dir, "weight_dist_equipped.png")
            )
        except Exception as e:
            return str(e), None, None

def create_ui() -> gr.Blocks:
    """Create the Gradio interface"""
    ui = PruningUI()
    
    with gr.Blocks(title="Interactive Model Pruning") as interface:
        gr.Markdown("""
        # Interactive Model Pruning Interface
        
        This interface allows you to:
        1. Load a pretrained model
        2. Iteratively prune and retrain the model
        3. Visualize the results
        
        Start by loading a model, then experiment with different pruning ratios!
        """)
        
        with gr.Row():
            with gr.Column():
                # Model Loading Section
                gr.Markdown("### 1. Load Model")
                model_input = gr.Textbox(
                    label="Model Path or HuggingFace ID",
                    placeholder="mit-han-lab/dcae-c64"
                )
                load_btn = gr.Button("Load Model")
                
                # Pruning Section
                gr.Markdown("### 2. Prune Model")
                sparsity_slider = gr.Slider(
                    minimum=0.0,
                    maximum=0.95,
                    value=0.5,
                    step=0.05,
                    label="Pruning Ratio"
                )
                prune_btn = gr.Button("Prune Model")
                
                # Training Section
                gr.Markdown("### 3. Train Model")
                with gr.Row():
                    epochs = gr.Number(
                        value=1,
                        label="Epochs",
                        minimum=1,
                        maximum=100
                    )
                    steps = gr.Number(
                        value=100,
                        label="Steps per Epoch",
                        minimum=10,
                        maximum=1000
                    )
                train_btn = gr.Button("Train Model")
                
                # Equip Section
                gr.Markdown("### 4. Equip Model")
                equip_btn = gr.Button("Equip Model")
            
            with gr.Column():
                # Output Section
                status_text = gr.Textbox(
                    label="Status",
                    lines=10,
                    interactive=False
                )
                
                with gr.Tab("Reconstructions"):
                    recon_image = gr.Image(
                        label="Sample Reconstructions",
                        interactive=False
                    )
                
                with gr.Tab("Weight Distribution"):
                    weight_dist = gr.Image(
                        label="Weight Distribution",
                        interactive=False
                    )
        
        # Event handlers
        load_btn.click(
            fn=ui.load_model,
            inputs=[model_input],
            outputs=[status_text, recon_image, weight_dist]
        )
        
        prune_btn.click(
            fn=ui.prune_model,
            inputs=[sparsity_slider],
            outputs=[status_text, recon_image, weight_dist]
        )
        
        train_btn.click(
            fn=ui.train_model,
            inputs=[epochs, steps],
            outputs=[status_text, recon_image, weight_dist]
        )
        
        equip_btn.click(
            fn=ui.equip_model,
            inputs=[],
            outputs=[status_text, recon_image, weight_dist]
        )
    
    return interface

if __name__ == "__main__":
    interface = create_ui()
    interface.launch(share=True)
