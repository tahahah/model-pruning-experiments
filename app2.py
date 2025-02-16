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
        
    def load_model(self, model_path: str) -> Tuple[str, str, str, str]:
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
            weight_plot = os.path.join(self.model_manager.save_dir, "weight_dist_initial.png")
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

    def get_equipped_metrics(self) -> Tuple[str, str, str]:
        """Get metrics and visualizations for equipped model"""
        try:
            metrics = self.model_manager._get_model_metrics(
                self.model_manager.equipped_model,
                save_reconstructions=True,
                step="equipped"
            )
            metrics_text = f"""VRAM: {metrics.get('vram_usage', 'N/A')}
Full param count: {metrics.get('total_params', '0')}
Loss: {metrics.get('reconstruction_loss', '0 MSE')}
Latency: {metrics.get('latency', '0 ms')}"""

            sample_image = os.path.join(self.model_manager.save_dir, "reconstruction_equipped.png")
            
            return (
                metrics_text,
                sample_image if os.path.exists(sample_image) else None,
                sample_image if os.path.exists(sample_image) else None
            )
        except Exception as e:
            logger.error(f"Error getting equipped metrics: {str(e)}")
            return f"Error getting equipped metrics: {str(e)}", None, None

    def get_experimental_metrics(self) -> Tuple[str, str, str]:
        """Get metrics and visualizations for experimental model"""
        try:
            metrics = self.model_manager._get_model_metrics(
                self.model_manager.experimental_model,
                save_reconstructions=True,
                step="experimental"
            )
            metrics_text = f"""VRAM: {metrics.get('vram_usage', 'N/A')}
Full param count: {metrics.get('total_params', '0')}
Loss: {metrics.get('reconstruction_loss', '0 MSE')}
Latency: {metrics.get('latency', '0 ms')}"""

            sample_image = os.path.join(self.model_manager.save_dir, "reconstruction_experimental.png")
            
            return (
                metrics_text,
                sample_image if os.path.exists(sample_image) else None,
                sample_image if os.path.exists(sample_image) else None
            )
        except Exception as e:
            logger.error(f"Error getting experimental metrics: {str(e)}")
            return f"Error getting experimental metrics: {str(e)}", None, None

    def prune_model(self, pruning_method: str, sparsity_ratio: float) -> Tuple[str, str, str]:
        """Prune the model using specified method and ratio"""
        try:
            metrics = self.model_manager.prune_experimental_model(sparsity_ratio)
            metrics_text = f"""VRAM: {metrics.get('vram_usage', 'N/A')}
Full param count: {metrics.get('total_params', '0')}
Loss: {metrics.get('reconstruction_loss', '0 MSE')}
Latency: {metrics.get('latency', '0 ms')}"""

            sample_image = os.path.join(self.model_manager.save_dir, "reconstruction_after_pruning.png")
            
            return (
                metrics_text,
                sample_image if os.path.exists(sample_image) else None,
                sample_image if os.path.exists(sample_image) else None
            )
        except Exception as e:
            logger.error(f"Error pruning model: {str(e)}")
            return f"Error pruning model: {str(e)}", None, None

    def train_model(self, n_epochs: int, steps_per_epoch: int) -> Tuple[str, str, str]:
        """Train the experimental model"""
        try:
            self.model_manager.train_experimental_model(n_epochs, steps_per_epoch)
            return self.get_experimental_metrics()
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return f"Error training model: {str(e)}", None, None

    def promote_to_equipped(self) -> Tuple[str, str, str]:
        """Promote experimental model to equipped model"""
        try:
            self.model_manager.promote_experimental_to_equipped()
            return self.get_equipped_metrics()
        except Exception as e:
            logger.error(f"Error promoting model: {str(e)}")
            return f"Error promoting model: {str(e)}", None, None

    def save_to_huggingface(self) -> str:
        """Save equipped model to HuggingFace Hub"""
        try:
            self.model_manager.upload_to_huggingface()
            return f"Successfully saved model to huggingface"
        except Exception as e:
            logger.error(f"Error saving to HuggingFace: {str(e)}")
            return f"Error saving to HuggingFace: {str(e)}"

def create_interface() -> gr.Blocks:
    ui = ModelPruningUI()
    
    with gr.Blocks(title="Model Pruning Studio v2") as interface:
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
                            weight_plot = gr.Image(
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
            
            # Prune and Finetune tab
            with gr.Tab("Prune and Finetune"):
                with gr.Row():
                    # Equipped Model Column
                    with gr.Column():
                        gr.Markdown("### Equipped Model")
                        equipped_metrics = gr.Textbox(
                            label="Metrics",
                            lines=4,
                            interactive=False
                        )
                        with gr.Row():
                            equipped_validate = gr.Button("Validate")
                            save_hf = gr.Button("Save to HuggingFace", variant="secondary")
                        
                        save_status = gr.Textbox(
                            label="Save Status",
                            visible=True
                        )
                        
                        with gr.Row():
                            equipped_sample = gr.Image(
                                label="Sample Image",
                                show_label=True
                            )
                            equipped_reconstructed = gr.Image(
                                label="Reconstructed Sample",
                                show_label=True
                            )
                        
                        with gr.Group():
                            gr.Markdown("### Prune")
                            pruning_method = gr.Dropdown(
                                choices=["magnitude", "random"],
                                label="Pruning Method",
                                value="magnitude"
                            )
                            sparsity_ratio = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.5,
                                label="Sparsity Ratio"
                            )
                            prune_btn = gr.Button("Prune")
                        
                        with gr.Group():
                            gr.Markdown("### Train")
                            n_epochs = gr.Number(
                                value=1,
                                label="Number of Epochs",
                                precision=0
                            )
                            steps_per_epoch = gr.Number(
                                value=100,
                                label="Steps per Epoch",
                                precision=0
                            )
                            with gr.Accordion("Advanced Settings", open=False):
                                gr.Markdown("Coming soon...")
                            train_btn = gr.Button("Train")
                    
                    # Experimental Model Column
                    with gr.Column():
                        gr.Markdown("### Resultant Model")
                        experimental_metrics = gr.Textbox(
                            label="Metrics",
                            lines=4,
                            interactive=False
                        )
                        experimental_validate = gr.Button("Validate")
                        
                        with gr.Row():
                            experimental_sample = gr.Image(
                                label="Sample Image",
                                show_label=True
                            )
                            experimental_reconstructed = gr.Image(
                                label="Reconstructed Sample",
                                show_label=True
                            )
                        
                        promote_btn = gr.Button(
                            "Promote to Equipped Model",
                            variant="primary"
                        )
        
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
        
        equipped_validate.click(
            fn=ui.get_equipped_metrics,
            inputs=[],
            outputs=[
                equipped_metrics,
                equipped_sample,
                equipped_reconstructed
            ]
        )
        
        experimental_validate.click(
            fn=ui.get_experimental_metrics,
            inputs=[],
            outputs=[
                experimental_metrics,
                experimental_sample,
                experimental_reconstructed
            ]
        )
        
        prune_btn.click(
            fn=ui.prune_model,
            inputs=[
                pruning_method,
                sparsity_ratio
            ],
            outputs=[
                experimental_metrics,
                experimental_sample,
                experimental_reconstructed
            ]
        )
        
        train_btn.click(
            fn=ui.train_model,
            inputs=[
                n_epochs,
                steps_per_epoch
            ],
            outputs=[
                experimental_metrics,
                experimental_sample,
                experimental_reconstructed
            ]
        )
        
        promote_btn.click(
            fn=ui.promote_to_equipped,
            inputs=[],
            outputs=[
                equipped_metrics,
                equipped_sample,
                equipped_reconstructed
            ]
        )
        
        save_hf.click(
            fn=ui.save_to_huggingface,
            inputs=[],
            outputs=[save_status]
        )
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True)
