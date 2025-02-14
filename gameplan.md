# Game Plan for Transitioning from CLI to Gradio-based UI

**IMPORTANT NOTE:** The existing `dcaecore` directory contains perfectly functioning code that MUST NOT be modified. All new functionality will be built around the existing `dcaecore` implementation, treating it as a stable core that we interface with.

## 1. Review and Understand the Current Codebase

- **Objective:** Identify the core functionalities in `iterative_prune_retrain.py` while preserving `dcaecore` functionality
- **Action Items:**
  - Document the current code structure and its interaction with `dcaecore`
  - Identify CLI-specific code that can be migrated to UI without affecting `dcaecore`
  - Map out integration points between our new UI and the existing `dcaecore` implementation

## 2. Define the Backend Components

- **Objective:** Create a new `ModelManager` class that wraps and interfaces with `dcaecore` without modifying it
  - **Create new modules:**
    - `model_manager.py`: Main class containing all model-related functionality
    - `ui_interface.py`: Gradio UI implementation
  - **ModelManager Class Structure:**
    - `initialize_model()`: Load and configure model using existing `dcaecore` functionality
    - `prune_model(model, threshold, ...)`: Interface with existing pruning logic
    - `retrain_model(model, epochs, ...)`: Utilize existing training infrastructure
    - `evaluate_model(model, test_data, ...)`: Leverage current evaluation methods
    - `get_model_metrics()`: Collect performance statistics
  - **Technical details:**
    - Maintain complete compatibility with `dcaecore`
    - Each function should wrap existing functionality rather than reimplementing it
    - For consequent pruning and retraining, use a deepcopy of the equipped model instance
    - Support validation loops for reconstruction visualization
    - Handle dataset streaming from `Tahahah/PacmanDataset_3`

## 3. Design the Gradio UI Interface

- **Objective:** Develop an interactive web dashboard using Gradio to replace CLI inputs.
  - **Layout Structure:**
    - **Top Row (Original Model Status):**
      - Display metrics for the initially loaded model
      - Show key statistics: number of parameters, sparsity ratio, latency
      - Display model performance: loss/accuracy metrics
      - Show reconstructed image output
      - Save/load/validate functionality for this baseline model
      - VRAM Occupied

    - **Middle Section (Training Interface - Two Columns):**
      - Left Column (Current Model):
        - Model configuration display
        - Current model metrics
        - Load model functionality
      - Right Column (Experimental Model):
        - Pruning parameters
        - Training parameters
        - Metrics
        - "Equip Model" button to promote experimental model to current model
      - Metrics for both models includes:
        - Live updates of current equipped model metrics
        - Sparsity ratio and parameter counts
        - Latency measurements
        - Loss/accuracy metrics
        - Drop down that contains Reconstructed image display from val sample
        - Drop down that contains weight distribution
        - VRAM Occupied
        - Alongside the absolute values for these, provide percentages improvments/declines with respect to the original model on the top row.
        

  - **Input Parameters:**
    1. **Model Configuration**
       - `pretrained_model_path`: File browser for model weights (.pth/.pt). (Currently only supporting DCAE)
       - `use_cuda`: Checkbox

    2. **Pruning Parameters**
       - `pruning_method`: Dropdown (L1 Unstructured, L2 Structured, Global Magnitude)
       - `pruning_rate`: Slider (0.1-0.9, step=0.05) (Always relative to the model on left)

    3. **Training Parameters**
       - `n_epochs`: Number input (default: 1)
       - `steps_per_epoch`: Number input (default: 100)
       - `init_lr`: Number input (default: 1.0e-4)
       - `warmup_epochs`: Number input (default: 5)
       - `warmup_lr`: Number input (default: 1.0e-6)
       - `lr_schedule_name`: Dropdown ["cosine"] (default: "cosine")
       - `optimizer_name`: Dropdown ["adamw"] (default: "adamw")
       - `optimizer_params`:
         - `betas`: Two number inputs [0.9, 0.999]
         - `eps`: Number input (default: 1.0e-8)
       - `weight_decay`: Number input (default: 0.05)
       - `grad_clip`: Number input (default: 1.0)

    4. **Data Configuration**
       - `dataset_name`: "Tahahah/PacmanDataset_3" (fixed)
       - `image_size`: Number input (default: 512)
       - `batch_size`: Number input (default: 1)
       - `num_workers`: Number input (default: 2)
       - `val_steps`: Number input (default: 100)

    5. **Validation & Logging**
       - `save_interval`: Number input (default: 5)
       - `eval_interval`: Number input (default: 1)
       - `log_interval`: Number input (default: 100)
       - `num_save_images`: Number input (default: 8)
       - Enable validation button for each model view
       - Real-time reconstruction preview
       - Metrics display (loss, accuracy)

  - **Implementation Details:**
    - Use `gradio.Blocks` for flexible layout
    - Implement state management for model tracking
    - Auto-save functionality with detailed naming convention
    - Real-time metric updates using `gr.update()`
    - Conditional widget visibility based on selections
    - Progress bars for long-running operations
    - Error handling and validation for all inputs
    - Allow for extra training epochs with custom steps/epochs even after initial training. Same goes for pruning.

## 4. Integrate Backend with Gradio UI

- **Objective:** Seamlessly connect the modular backend with the interactive UI.
  - **Technical integration:**
    - Wrap backend functions in lightweight wrapper functions that are called by Gradio callbacks.
    - Utilize asynchronous programming (e.g., Python's `asyncio`) when calling long-running tasks to avoid blocking the UI.
    - Ensure thread safety if sharing model state between functions.
    - Optionally, incorporate background task management to handle iterative processes without freezing the UI.

## 5. Enhance Logging and Error Handling

- **Objective:** Implement comprehensive logging through both Python logger and Weights & Biases
  - **Implementation details:**
    - **Python Logger Integration:**
      - Create a custom handler that appends log messages to an in-memory list or queue
      - Display real-time logs in the Gradio UI via `gr.Textbox`
      - Add detailed try/except blocks with descriptive error messages
    
    - **Weights & Biases Integration:**
      - Initialize W&B runs with descriptive names based on model state:
        - Format: `{model_type}_{pruning_method}_{sparsity_ratio}_{timestamp}`
        - Example: `dcae_l1unstructured_0.5_20240214`
      - Log key metrics:
        - Model metrics (parameters, sparsity, latency)
        - Training metrics (loss, accuracy)
        - Reconstruction quality
      - Create comparison tables between original, equipped, and experimental models
      - Save model artifacts with proper versioning
      - Track experiment lineage (which model was derived from which)
    
    - **Unified Logging Strategy:**
      - Ensure critical information appears in both Python logs and W&B
      - Create custom W&B panels for model comparison visualization
      - Enable easy experiment reproduction through logged hyperparameters
      - Maintain a clear history of model evolution through pruning and retraining

## 6. Refactor the Code

- **Objective:** Clean up the existing script to remove CLI interactions and align with the new modular and UI-driven architecture.
  - **Steps:**
    - Remove `argparse` and any related CLI handling code.
    - Replace inline parameter definitions with dynamic inputs from the Gradio UI.
    - Refactor function calls and integrate new backend modules (e.g., call functions from `model_manager.py`).
    - Update docstrings and add inline comments to explain complex sections and new asynchronous logic.

## 7. Create and Run Unit Tests for Backend Components

- **Objective:** Ensure the integrity of core functionalities after refactoring.
  - **Testing details:**
    - Write unit tests focusing solely on the backend functions (e.g., for `prune_model`, `retrain_model`, etc.).
    - Use testing frameworks such as `pytest` to validate that input parameters yield expected outputs, warnings, or errors.
    - Exclude UI component testing since it will be handled manually.

## 8. Documentation and Code Comments

- **Objective:** Improve code clarity with detailed documentation to support future developers.
  - **Steps:**
    - Update inline comments to clearly explain each function, especially newly added asynchronous code and error handling blocks.
    - Maintain a README or dedicated documentation that describes how to run the Gradio application and use each of its features.
    - Document any new dependencies (e.g., gradio, asyncio) in the dependency file (e.g., requirements.txt).

## 9. Deployment and User Feedback

- **Objective:** Launch the refactored application and gather feedback from end-users.
  - **Technical steps:**
    - Deploy the Gradio app locally or on a web server; ensure that all environment variables and dependencies are correctly configured.
    - Monitor application performance and log user interactions to quickly identify and resolve any issues.
    - Provide clear instructions within the UI so users understand how to interact with the system and where to report bugs or suggestions.

## 10. Future Enhancements

- **Objective:** Plan for continuous improvements beyond the initial refactoring.
  - **Action Items:**
    - Consider integrating more advanced visualization libraries like Plotly for real-time metrics.
    - Optionally, explore multi-user dashboards and remote monitoring for the training process.
    - Review additional features based on user requests and real-world usage.

---

This enhanced game plan provides explicit technical details down to individual functions, error handling, asynchronous processing, and UI widget design to guide the transformation from a CLI-based script to an interactive, web-based application powered by Gradio. The focus remains on strengthening backend modularity and UI integration, with unit tests maintained only for backend components.
