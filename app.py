import os
import gradio as gr
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any
import asyncio
from model_manager import ModelManager
import logging
import sys

@dataclass
class ModelState:
    """Tracks the state and metrics of a model"""
    metrics: Dict[str, Any] = None
    vram_usage: float = 0.0
    reconstructions: str = None
    weight_dist: str = None
    
class PruningUI:
    def __init__(self, config_path: str = "dcaecore/config.yaml", save_dir: str = "output"):
        self.model_manager = ModelManager(config_path, save_dir)
        self.save_dir = save_dir
        
        # Track states for different model versions
        self.original_state = ModelState()
        self.equipped_state = ModelState()
        self.experimental_state = ModelState()
        
        # Default training config
        self.default_config = {
            "init_lr": 1.0e-4,
            "warmup_epochs": 5,
            "warmup_lr": 1.0e-6,
            "lr_schedule": "cosine",
            "optimizer": "adamw",
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1.0e-8,
            "weight_decay": 0.05,
            "grad_clip": 1.0,
            "batch_size": 1,
            "num_workers": 2,
            "val_steps": 100,
            "save_interval": 5,
            "eval_interval": 1,
            "log_interval": 100,
            "num_save_images": 8
        }
    
    def load_model(self, model_path: str, use_cuda: bool) -> tuple:
        """Load initial model and update original state"""
        try:
            # Load model and get metrics
            metrics = self.model_manager.load_initial_model(model_path)
            
            # Update original model state
            self.original_state.metrics = metrics
            self.original_state.reconstructions = os.path.join(self.save_dir, "reconstructions_initial.png")
            self.original_state.weight_dist = os.path.join(self.save_dir, "weight_dist_initial.png")
            
            # Update VRAM usage
            if torch.cuda.is_available() and use_cuda:
                self.original_state.vram_usage = torch.cuda.memory_allocated() / 1024**2
            
            # Also update equipped model state since they start the same
            self.equipped_state.metrics = metrics.copy()
            self.equipped_state.reconstructions = os.path.join(self.save_dir, "reconstructions_equipped.png")
            self.equipped_state.weight_dist = os.path.join(self.save_dir, "weight_dist_equipped.png")
            self.equipped_state.vram_usage = self.original_state.vram_usage
            
            # Ensure files exist before returning
            if not all(os.path.exists(f) for f in [
                self.original_state.reconstructions,
                self.original_state.weight_dist,
                self.equipped_state.reconstructions,
                self.equipped_state.weight_dist
            ]):
                raise FileNotFoundError("Required visualization files were not generated")
            
            return (
                self._format_metrics(self.original_state.metrics),
                self.original_state.reconstructions,
                self.original_state.weight_dist,
                f"VRAM: {self.original_state.vram_usage:.1f}MB",
                self._format_metrics(self.equipped_state.metrics),
                "",  # No relative metrics for initial load
                self.equipped_state.reconstructions,
                self.equipped_state.weight_dist,
                f"VRAM: {self.equipped_state.vram_usage:.1f}MB"
            )
        except Exception as e:
            print(f"Error loading model: {str(e)}")  # For debugging
            return (
                f"Error: {str(e)}",
                None,
                None, 
                "VRAM: N/A",
                "Error: Model not loaded",
                "",
                None,
                None,
                "VRAM: N/A"
            )
    
    def prune_model(
        self, 
        method: str,
        sparsity: float,
        progress: gr.Progress = gr.Progress()
    ) -> tuple:
        """Prune model and update experimental state"""
        try:
            progress(0, desc="Pruning model...")
            metrics = self.model_manager.prune_experimental_model(sparsity)
            self.experimental_state.metrics = metrics
            self.experimental_state.reconstructions = os.path.join(self.save_dir, "reconstructions_after_pruning.png")
            self.experimental_state.weight_dist = os.path.join(self.save_dir, "weight_dist_after_pruning.png")
            
            if torch.cuda.is_available():
                self.experimental_state.vram_usage = torch.cuda.memory_allocated() / 1024**2
            
            progress(1.0, desc="Pruning complete!")
            
            # Calculate relative changes
            base_metrics = self.equipped_state.metrics or self.original_state.metrics
            param_change = (metrics['total_params'] - base_metrics['total_params']) / base_metrics['total_params']
            sparsity_change = metrics['sparsity_ratio'] - base_metrics['sparsity_ratio']
            
            return (
                self._format_metrics(metrics),                     # metrics textbox
                f"Parameter Change: {param_change:+.2%}\n"        # relative metrics
                f"Sparsity Change: {sparsity_change:+.2%}",
                self.experimental_state.reconstructions,           # recon image
                self.experimental_state.weight_dist,              # weights image
                f"VRAM: {self.experimental_state.vram_usage:.1f}MB"  # vram textbox
            )
            
        except Exception as e:
            return (
                f"Error: {str(e)}",  # metrics textbox
                "",                  # relative metrics
                None,               # recon image
                None,               # weights image
                "VRAM: N/A"        # vram textbox
            )
    
    def train_model(
        self,
        epochs: int,
        steps_per_epoch: int,
        init_lr: float,
        warmup_epochs: int,
        warmup_lr: float,
        optimizer_name: str,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
        grad_clip: float,
        progress: gr.Progress = gr.Progress()
    ) -> tuple:
        """Train the experimental model"""
        try:
            progress(0, desc="Training model...")
            metrics = self.model_manager.train_experimental_model(epochs, steps_per_epoch)
            self.experimental_state.metrics = metrics
            self.experimental_state.reconstructions = os.path.join(self.save_dir, "reconstructions_after_training.png")
            self.experimental_state.weight_dist = os.path.join(self.save_dir, "weight_dist_after_training.png")
            
            if torch.cuda.is_available():
                self.experimental_state.vram_usage = torch.cuda.memory_allocated() / 1024**2
            
            progress(1.0, desc="Training complete!")
            
            # Calculate relative changes
            base_metrics = self.equipped_state.metrics or self.original_state.metrics
            param_change = (metrics['total_params'] - base_metrics['total_params']) / base_metrics['total_params']
            sparsity_change = metrics['sparsity_ratio'] - base_metrics['sparsity_ratio']
            
            return (
                self._format_metrics(metrics),                     # metrics textbox
                f"Parameter Change: {param_change:+.2%}\n"        # relative metrics
                f"Sparsity Change: {sparsity_change:+.2%}",
                self.experimental_state.reconstructions,           # recon image
                self.experimental_state.weight_dist,              # weights image
                f"VRAM: {self.experimental_state.vram_usage:.1f}MB"  # vram textbox
            )
            
        except Exception as e:
            return (
                f"Error: {str(e)}",  # metrics textbox
                "",                  # relative metrics
                None,               # recon image
                None,               # weights image
                "VRAM: N/A"        # vram textbox
            )
    
    def equip_model(self) -> tuple:
        """Promote experimental model to equipped status"""
        try:
            metrics = self.model_manager.equip_experimental_model()
            self.equipped_state.metrics = metrics
            self.equipped_state.reconstructions = os.path.join(self.save_dir, "reconstructions_equipped.png")
            self.equipped_state.weight_dist = os.path.join(self.save_dir, "weight_dist_equipped.png")
            
            if torch.cuda.is_available():
                self.equipped_state.vram_usage = torch.cuda.memory_allocated() / 1024**2
            
            # Calculate relative changes from original
            param_change = (metrics['total_params'] - self.original_state.metrics['total_params']) / self.original_state.metrics['total_params']
            sparsity_change = metrics['sparsity_ratio'] - self.original_state.metrics['sparsity_ratio']
            
            return (
                self._format_metrics(metrics),                     # metrics textbox
                f"Parameter Change: {param_change:+.2%}\n"        # relative metrics
                f"Sparsity Change: {sparsity_change:+.2%}",
                self.equipped_state.reconstructions,              # recon image
                self.equipped_state.weight_dist,                 # weights image
                f"VRAM: {self.equipped_state.vram_usage:.1f}MB"   # vram textbox
            )
            
        except Exception as e:
            return (
                f"Error: {str(e)}",  # metrics textbox
                "",                  # relative metrics
                None,               # recon image
                None,               # weights image
                "VRAM: N/A"        # vram textbox
            )
    
    def upload_to_huggingface(self):
        """Upload equipped model to HuggingFace Hub"""
        success = self.model_manager.upload_to_huggingface()
        if success:
            return "Successfully uploaded model to HuggingFace Hub"
        else:
            return "Failed to upload model. Check logs for details."
    
    def _format_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format metrics dictionary into a readable string"""
        if not metrics:
            return "No metrics available"
        
        formatted = []
        for key, value in metrics.items():
            if isinstance(value, float):
                formatted.append(f"{key}: {value:.4f}")
            else:
                formatted.append(f"{key}: {value}")
        return "\n".join(formatted)

    def _get_comparison_metrics(self, model_type: str) -> tuple:
        """Get comparative metrics between models"""
        if model_type == "experimental":
            state = self.experimental_state
            base_metrics = self.equipped_state.metrics or self.original_state.metrics
        else:  # equipped
            state = self.equipped_state
            base_metrics = self.original_state.metrics
            
        if not state.metrics or not base_metrics:
            return (
                "",  # metrics textbox
                "",  # relative metrics
                None,  # recon image
                None,  # weights image
                ""  # vram textbox
            )
            
        # Calculate relative changes
        param_change = (state.metrics['total_params'] - base_metrics['total_params']) / base_metrics['total_params']
        sparsity_change = state.metrics['sparsity_ratio'] - base_metrics['sparsity_ratio']
        
        return (
            self._format_metrics(state.metrics),                     # metrics textbox
            f"Parameter Change: {param_change:+.2%}\n"        # relative metrics
            f"Sparsity Change: {sparsity_change:+.2%}",
            state.reconstructions,           # recon image
            state.weight_dist,              # weights image
            f"VRAM: {state.vram_usage:.1f}MB"  # vram textbox
        )

def create_ui() -> gr.Blocks:
    """Create the Gradio interface following gameplan specifications"""
    ui = PruningUI()
    
    with gr.Blocks(title="Model Pruning Studio") as interface:
        gr.Markdown("""
        # Interactive Model Pruning Studio
        Iteratively prune and retrain vision models with real-time feedback
        """)
        
        # Model Loading Section
        with gr.Row():
            with gr.Column():
                model_path = gr.Textbox(
                    label="Model Path/ID",
                    placeholder="mit-han-lab/dc-ae-f32c32-in-1.0",
                    value="mit-han-lab/dc-ae-f32c32-in-1.0"  # Default value
                )
                use_cuda = gr.Checkbox(
                    label="Use CUDA",
                    value=torch.cuda.is_available()
                )
                load_btn = gr.Button("Load Model", variant="primary")
        
        # Original Model Status
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Original Model Status")
                original_metrics = gr.Textbox(
                    label="Metrics",
                    interactive=False,
                    lines=4
                )
                original_vram = gr.Textbox(
                    label="VRAM Usage",
                    interactive=False
                )
            with gr.Column():
                with gr.Row():
                    original_recon = gr.Image(
                        label="Original Reconstructions",
                        interactive=False,
                        type="filepath"  # Use filepath instead of numpy array
                    )
                    original_weights = gr.Image(
                        label="Original Weight Distribution",
                        interactive=False,
                        type="filepath"
                    )
        
        # Middle Section: Two Columns
        with gr.Row():
            # Left Column (Current Model)
            with gr.Column():
                gr.Markdown("### Current Model")
                # Current Model Metrics
                equipped_metrics = gr.Textbox(
                    label="Current Model Metrics",
                    interactive=False,
                    lines=4
                )
                equipped_relative = gr.Textbox(
                    label="Relative to Original",
                    interactive=False,
                    lines=2
                )
                equipped_vram = gr.Textbox(
                    label="VRAM Usage",
                    interactive=False
                )
                
                with gr.Accordion("Visualizations", open=False):
                    equipped_recon = gr.Image(
                        label="Current Reconstructions",
                        interactive=False,
                        type="filepath"
                    )
                    equipped_weights = gr.Image(
                        label="Current Weight Distribution",
                        interactive=False,
                        type="filepath"
                    )
            
            # Right Column (Experimental Model)
            with gr.Column():
                gr.Markdown("### Experimental Model")
                with gr.Group():
                    # Pruning Configuration
                    pruning_method = gr.Dropdown(
                        choices=["L1 Unstructured", "L2 Structured", "Global Magnitude"],
                        value="L1 Unstructured",
                        label="Pruning Method"
                    )
                    sparsity = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.5,
                        step=0.05,
                        label="Pruning Rate"
                    )
                    prune_btn = gr.Button("Prune Model", variant="primary")
                
                with gr.Group():
                    # Training Configuration
                    with gr.Row():
                        epochs = gr.Number(
                            value=1,
                            label="Epochs",
                            precision=0
                        )
                        steps = gr.Number(
                            value=100,
                            label="Steps/Epoch",
                            precision=0
                        )
                    
                    with gr.Accordion("Advanced Training Options", open=False):
                        init_lr = gr.Number(
                            value=1.0e-4,
                            label="Initial Learning Rate"
                        )
                        warmup_epochs = gr.Number(
                            value=5,
                            label="Warmup Epochs",
                            precision=0
                        )
                        warmup_lr = gr.Number(
                            value=1.0e-6,
                            label="Warmup Learning Rate"
                        )
                        optimizer = gr.Dropdown(
                            choices=["adamw"],
                            value="adamw",
                            label="Optimizer"
                        )
                        with gr.Row():
                            beta1 = gr.Number(
                                value=0.9,
                                label="Beta1"
                            )
                            beta2 = gr.Number(
                                value=0.999,
                                label="Beta2"
                            )
                        eps = gr.Number(
                            value=1.0e-8,
                            label="Epsilon"
                        )
                        weight_decay = gr.Number(
                            value=0.05,
                            label="Weight Decay"
                        )
                        grad_clip = gr.Number(
                            value=1.0,
                            label="Gradient Clipping"
                        )
                    
                    train_btn = gr.Button("Train Model", variant="primary")
                
                # Experimental Model Metrics
                experimental_metrics = gr.Textbox(
                    label="Experimental Model Metrics",
                    interactive=False,
                    lines=4
                )
                experimental_relative = gr.Textbox(
                    label="Relative to Current",
                    interactive=False,
                    lines=2
                )
                experimental_vram = gr.Textbox(
                    label="VRAM Usage",
                    interactive=False
                )
                
                with gr.Accordion("Visualizations", open=False):
                    experimental_recon = gr.Image(
                        label="Experimental Reconstructions",
                        interactive=False,
                        type="filepath"
                    )
                    experimental_weights = gr.Image(
                        label="Experimental Weight Distribution",
                        interactive=False,
                        type="filepath"
                    )
                
                with gr.Row():
                    equip_btn = gr.Button("Promote to Production")
                    upload_btn = gr.Button("Upload to HuggingFace")
                equip_status = gr.Textbox(label="Status")
                
                equip_btn.click(
                    ui.equip_model,
                    outputs=[equip_status]
                )
                upload_btn.click(
                    ui.upload_to_huggingface,
                    outputs=[equip_status]
                )
        
        # Event Handlers
        load_btn.click(
            fn=ui.load_model,
            inputs=[model_path, use_cuda],
            outputs=[
                original_metrics,
                original_recon,
                original_weights,
                original_vram,
                equipped_metrics,  # Also update equipped model metrics
                equipped_relative,
                equipped_recon,
                equipped_weights,
                equipped_vram
            ]
        )
        
        prune_btn.click(
            fn=ui.prune_model,
            inputs=[
                pruning_method,
                sparsity
            ],
            outputs=[
                experimental_metrics,
                experimental_relative,
                experimental_recon,
                experimental_weights,
                experimental_vram
            ]
        )
        
        train_btn.click(
            fn=ui.train_model,
            inputs=[
                epochs,
                steps,
                init_lr,
                warmup_epochs,
                warmup_lr,
                optimizer,
                beta1,
                beta2,
                eps,
                weight_decay,
                grad_clip
            ],
            outputs=[
                experimental_metrics,
                experimental_relative,
                experimental_recon,
                experimental_weights,
                experimental_vram
            ]
        )
    
    return interface

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create output directory
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the Gradio interface
    interface = create_ui()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
