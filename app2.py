import os
import gradio as gr
import torch
import logging
from model_manager import ModelManager
from typing import Dict, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelPruningUI:
    def __init__(self, config_path: str = "dcaecore/config.yaml", save_dir: str = "outputs"):
        self.model_manager = ModelManager(config_path, save_dir)
        
    def load_model(self, model_path: str) -> Tuple[str, gr.Plot, gr.Image, gr.Image]:
        """Load a model and return its metrics and visualizations"""
        try:
            metrics = self.model_manager.load_initial_model(model_path)
            
            # Format metrics text
            metrics_text = f"""VRAM Usage: {metrics.get('vram_usage', 'N/A')}
Parameters: {metrics.get('total_params', '0')}
Sparsity: {metrics.get('sparsity_ratio', '0%')}
Reconstruction Loss: {metrics.get('reconstruction_loss', '0 MSE')}
Perceptual Loss: {metrics.get('perceptual_loss', '0 LPIPS')}
Inference Latency: {metrics.get('latency', '0 ms')}"""

            # Get visualizations
            weight_plot = os.path.join(self.model_manager.save_dir, "weight_distribution_initial.png")
            sample_image = os.path.join(self.model_manager.save_dir, "sample_image_initial.png")
            reconstructed = os.path.join(self.model_manager.save_dir, "reconstruction_initial.png")
            
            return (
                metrics_text,
                weight_plot if os.path.exists(weight_plot) else None,
                sample_image if os.path.exists(sample_image) else None,
                reconstructed if os.path.exists(reconstructed) else None
            )
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return f"Error loading model: {str(e)}", None, None, None
            
    def validate_model(self) -> str:
        """Run validation on currently loaded model"""
        try:
            metrics = self.model_manager._get_model_metrics(self.model_manager.original_model)
            return f"""Validation Results:
Reconstruction Loss: {metrics.get('reconstruction_loss', '0 MSE')}
Perceptual Loss: {metrics.get('perceptual_loss', '0 LPIPS')}
Inference Latency: {metrics.get('latency', '0 ms')}"""
        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            return f"Error during validation: {str(e)}"

def create_interface() -> gr.Blocks:
    ui = ModelPruningUI()
    
    # I prefer default theme
    with gr.Blocks(title="Model Pruning Studio v2",) as interface:
        with gr.Tabs():
            # Load Model Tab
            with gr.Tab("Load Model"):
                with gr.Column():
                    gr.Markdown("### Load Model")
                    with gr.Row():
                        with gr.Column(scale=4):
                            model_path = gr.Textbox(
                                label="Model path or HF name",
                                placeholder="Enter local path or HuggingFace model name",
                                value="mit-han-lab/dc-ae-f32c32-in-1.0"
                            )
                        with gr.Column(scale=1):
                            load_btn = gr.Button(
                                "Load",
                                variant="primary",
                                scale=1,
                                min_width=100
                            )
                    
                    # Metrics and Validate button
                    with gr.Row():
                        with gr.Column(scale=1):
                            metrics_text = gr.Textbox(
                                label="Model Metrics",
                                lines=6,
                                interactive=False
                            )
                            validate_btn = gr.Button("Validate")
                        
                        # Weight distribution plot
                        with gr.Column(scale=2):
                            weight_plot = gr.Plot(
                                label="Weight distribution plot",
                                show_label=True,
                                container=False,
                            )
                    
                    # Sample and Reconstructed images
                    with gr.Row():
                        sample_image = gr.Image(
                            label="Sample Image",
                            show_label=True,
                            container=True
                        )
                        reconstructed = gr.Image(
                            label="Reconstructed Sample image",
                            show_label=True,
                            container=True
                        )
            
            # Placeholder for Prune and Finetune tab
            with gr.Tab("Prune and Finetune"):
                gr.Markdown("### Coming Soon")
        
        # Event handlers
        load_btn.click(
            fn=ui.load_model,
            inputs=[model_path],
            outputs=[
                metrics_text,
                weight_plot,
                sample_image,
                reconstructed
            ]
        )
        
        validate_btn.click(
            fn=ui.validate_model,
            inputs=[],
            outputs=[metrics_text]
        )
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True)
