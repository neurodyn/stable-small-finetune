# Fine-tuning Stable Audio Open Small

# Previous notes:

# Fine-tuning Stable Audio Open Small

This guide is a work in progress to describe the workflow for fine-tuning the "stable-audio-open-small" model using your own dataset. 

We will use Conda for managing dependencies and the scripts provided in the `stable-audio-tools` repository.

This is not yet complete, so do not expect things to work if you try it! TBC!

## Prerequisites

*   NVIDIA GPU with CUDA drivers installed on your Windows system.
*   WSL installed with a Linux distribution (e.g., Ubuntu).
*   Conda installed within your WSL environment.

## Project Structure

```
.
├── stable-audio-tools/ # Cloned repository from Stability AI
├── dataset/            # Your audio files and metadata
├── configs/            # Model and dataset configuration files
├── models/             # Saved model checkpoints
├── README.md           # This file
└── (your Conda environment)
```

## Setup Instructions

1.  **Clone the `stable-audio-tools` Repository (if not already done):**
    Open your WSL terminal and navigate to your desired project directory.
    ```bash
    git clone https://github.com/Stability-AI/stable-audio-tools.git
    ```

2.  **Navigate to the Project Directory:**
    ```bash
    cd /path/to/your/project/open-small-fine-tune 
    # (This directory, containing stable-audio-tools, dataset, etc.)
    ```

3.  **Create and Activate Conda Environment:**
    The `stable-audio-tools` repository specifies Python 3.10 and PyTorch 2.5 or later.
    ```bash
    conda create -n stable-audio python=3.10
    conda activate stable-audio
    ```

4.  **Install PyTorch with CUDA:**
    Visit the [official PyTorch website](https://pytorch.org/get-started/locally/) to get the correct Conda installation command for your CUDA version. For example, for CUDA 12.1:
    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    ```
    Ensure your NVIDIA drivers on Windows are compatible with the CUDA version you install in WSL.

5.  **Install `stable-audio-tools` and Dependencies:**
    Navigate into the cloned `stable-audio-tools` directory and install it.
    ```bash
    cd stable-audio-tools
    pip install -e . 
    # The -e flag installs it in editable mode, which is good for development.
    cd .. 
    # Go back to the project root
    ```
    This will also install other necessary dependencies listed in its `pyproject.toml` or `setup.py`. You might also need to install `wandb` if you plan to use Weights & Biases for logging:
    ```bash
    pip install wandb
    wandb login # (You'll need a Weights & Biases account)
    ```

## Workflow

The fine-tuning process involves preparing your data, creating configuration files, running the training script, and finally, using the fine-tuned model for inference.

### 1. Data Collection

*   Gather your audio files. These should be high-quality audio samples relevant to the sounds you want the fine-tuned model to generate.
*   Supported formats are typically `.wav`, `.mp3`, etc., but it's best to convert them to a consistent format and sample rate (e.g., WAV, 44.1kHz or 48kHz, as supported by the base model).
*   Place your audio files in a subdirectory within the `dataset/` folder (e.g., `dataset/my_custom_audio/`).

### 2. Data Formatting and Annotation

The `stable-audio-tools` library expects a specific dataset format, often involving a metadata file (e.g., a JSONL file) that links audio file paths to their descriptions or other conditional information.

*   **Create a Metadata File:**
    You'll likely need a JSONL file where each line is a JSON object containing at least:
    *   `"audio_path"`: The relative or absolute path to an audio file.
    *   `"text"`: A textual description of the audio content (this will be your prompt during training and inference).
    *   Other optional metadata like `duration`, `sample_rate`, `artist`, `genre`, etc., depending on the dataset configuration requirements of `stable-audio-tools`.

    Example `metadata.jsonl` entry (place this file in `dataset/`):
    ```json
    {"audio_path": "my_custom_audio/sound1.wav", "text": "a recording of a cat purring loudly", "duration": 10.5, "sample_rate": 48000}
    {"audio_path": "my_custom_audio/sound2.wav", "text": "the sound of a gentle breeze rustling leaves", "duration": 15.2, "sample_rate": 48000}
    ```
    Refer to the `stable-audio-tools` documentation (particularly around dataset configs) for the exact required structure. Look for examples in the `docs` or `configs` directory of the `stable-audio-tools` repository.

### 3. Dataset Configuration

`stable-audio-tools` uses a JSON configuration file to define how your dataset should be loaded and processed.

*   Create a `dataset_config.json` file in your `configs/` directory.
*   This file will specify:
    *   The type of dataset (e.g., local audio directory).
    *   The path to your metadata file (e.g., `../dataset/metadata.jsonl`).
    *   The path to the base directory of your audio files (e.g., `../dataset/`).
    *   Preprocessing steps, target sample rate, etc.

    Example `configs/dataset_config.json` (this is a hypothetical example, refer to `stable-audio-tools` for actual structure):
    ```json
    {
        "dataset_type": "audio_dir_dataset", // This type might vary
        "datasets": [
            {
                "id": "my_custom_dataset",
                "path": "../dataset/", // Base path to audio files
                "metadata_path": "../dataset/metadata.jsonl" // Path to your metadata
            }
        ],
        "sample_rate": 48000, // Match your audio and model needs
        "channels": 2, // Or 1 for mono
        "random_crop_size": 65536 // Example, adjust as needed
        // ... other parameters as required by stable-audio-tools
    }
    ```
    **Action:** You will need to consult the `stable-audio-tools` documentation or existing example configurations to create a valid dataset config.

### 4. Model Configuration

You'll also need a model configuration file. For fine-tuning, you'll typically start from a pre-trained model configuration and modify it.

*   **Download the Base Model Configuration:**
    1.  Go to the Hugging Face model page for `stabilityai/stable-audio-open-small`: [https://huggingface.co/stabilityai/stable-audio-open-small/tree/main](https://huggingface.co/stabilityai/stable-audio-open-small/tree/main)
    2.  You will likely need to log in and/or agree to the model's terms to access the files.
    3.  Download the `model_config.json` file.
    4.  Save this file into your project's `configs/` directory. It's a good idea to save it as something like `model_config_small_original.json` to keep the original safe.
    5.  Make a copy of this file and name it `model_config_small_finetune.json` (also in the `configs/` directory). This is the file you will modify for your fine-tuning experiment.

*   The `model_config_small_finetune.json` file defines model architecture, sample rate, sample size, and crucially, training parameters (learning rate, batch size, etc.).
*   Key aspects to potentially modify in `model_config_small_finetune.json` for fine-tuning:
    *   Ensure `sample_rate` and `sample_size` are compatible with your dataset and goals. The original values are good starting points.
    *   Adjust training parameters like `learning_rate` (usually lower for fine-tuning), `batch_size` (based on GPU VRAM).

    Example snippet from a hypothetical `configs/model_config_small_finetune.json`:
    ```json
    {
        "model_type": "diffusion_cond", // Or similar, based on stable-audio-open-small
        "sample_size": 262144, // Example value from stable-audio-open-small, typically corresponds to a few seconds of audio
        "sample_rate": 44100, // Default for stable-audio-open-small
        "audio_channels": 2,
        "model": {
            // ... model-specific architecture details from the downloaded config...
        },
        "training": {
            // ... training settings from the downloaded config...
            // Parameters you might want to adjust for fine-tuning:
            "optimizer": "adamw", 
            "lr": 1e-5, // Or lower (e.g., 5e-6, 1e-6) for fine-tuning
            "batch_size": 4, // Adjust based on your GPU VRAM
            "grad_accum_steps": 1,
            // ... other training settings
        }
        // ... other parameters from the downloaded config...
    }
    ```
    **Action:** Download the `model_config.json` from Hugging Face as described above. Inspect it and then adapt your copy (`model_config_small_finetune.json`) for fine-tuning, especially the `training` section (e.g., learning rate, batch size). The example snippet above shows common items but the actual file will be more detailed.

### 5. Training

Once your data is prepared and configurations are set up, you can start the fine-tuning process using the `train.py` script from `stable-audio-tools`.

*   **Download Pre-trained Model Weights & Understanding "Unwrapping" for Fine-tuning:**
    Ensure you have completed Step 1 (logged into Hugging Face and accepted terms for the model).
    1.  **Download Weights:** Download the "stable-audio-open-small" model weights from its Hugging Face page: [https://huggingface.co/stabilityai/stable-audio-open-small/tree/main](https://huggingface.co/stabilityai/stable-audio-open-small/tree/main). 
        *   Prioritize the `.safetensors` file (e.g., `model.safetensors`) as it's safer and often more efficient.
        *   Place the downloaded model checkpoint in your project's `models/` directory.

    2.  **Why Unwrapped for `--pretrained-ckpt-path`?**
        When you start a *new fine-tuning run* with `train.py` and provide existing weights via `--pretrained-ckpt-path`, the script typically expects the *core model weights* (an "unwrapped" model). This is because `train.py` will build a new training wrapper (with your specified optimizer settings, learning rate from your `model_config_small_finetune.json`, etc.) and then load the provided pre-trained weights into this new model structure.
        *   Loading a *fully wrapped* checkpoint (with its original optimizer states, etc.) via `--pretrained-ckpt-path` might conflict if your fine-tuning setup (e.g., learning rate, optimizer type in your `model_config_small_finetune.json`) is different from the original pre-training setup.
        *   For continuing training *from an existing wrapped checkpoint with the exact same training setup*, you would typically use the `--ckpt-path` argument instead (not `--pretrained-ckpt-path`).

    3.  **Are Hugging Face Checkpoints Already Unwrapped?**
        Official model releases on Hugging Face (especially `.safetensors` files) are generally distributed as *unwrapped* (or inference-ready) checkpoints. This means they contain the core model weights suitable for direct use or for starting fine-tuning as described here.
        *   So, for `stable-audio-open-small`, the downloaded `model.safetensors` should work directly with `--pretrained-ckpt-path` without needing a manual unwrap step *before fine-tuning*.
        *   The `unwrap_model.py` script is more commonly used to: 
            a. Convert your *own* wrapped training checkpoints (produced during your fine-tuning) into unwrapped models for inference (as detailed in Section 6).
            b. If you had a checkpoint from a different source that was indeed a fully wrapped PyTorch Lightning checkpoint and you wanted to start fresh fine-tuning.

    4.  **Summary for Pre-trained Model:** Use the downloaded `.safetensors` file directly with `--pretrained-ckpt-path`. If you encounter unexpected errors related to checkpoint loading, then investigating the `unwrap_model.py` script for this pre-trained file might be a fallback, but it's usually not the first step for official Hugging Face model files.

*   **Run the Training Script:**
    From your project root directory (`open-small-fine-tune`), execute:
    ```bash
    conda activate stable-audio # If not already active
    python stable-audio-tools/train.py \
        --dataset-config ./configs/dataset_config.json \
        --model-config ./configs/model_config_small_finetune.json \
        --pretrained-ckpt-path ./models/model.safetensors \
        --name "stable_audio_small_finetuned_custom" \
        --save-dir ./models/ \
        --batch-size 4 \
        --num-gpus 1 \
        --precision 16 \
        # Add other relevant flags from stable-audio-tools/train.py --help
        # (e.g., --checkpoint-every, --num-workers, --learning-rate if not in config)
    ```
    *   Replace `./models/model.safetensors` with the actual path to your downloaded pre-trained model checkpoint.
    *   Adjust `batch-size`, `num-gpus`, `precision`, and other parameters based on your hardware and the guidance in `stable-audio-tools` documentation and your `model_config_small_finetune.json`.
    *   Training progress and checkpoints will typically be saved in the directory specified by `--save-dir` under a subdirectory named after your `--name`.

### 6. Inference

After fine-tuning, you'll have a new model checkpoint (likely a wrapped checkpoint, e.g., `epoch_X_step_Y.ckpt`, located in `./models/stable_audio_small_finetuned_custom/checkpoints/`). You can use this for generating audio.

*   **Using `run_gradio.py` (Example):**
    The `run_gradio.py` script usually requires an **unwrapped** model checkpoint if you are loading a local model with `--ckpt-path` and `--model-config`.
    First, you'll likely need to unwrap your fine-tuned training checkpoint.
    Let's assume your fine-tuned checkpoint is at `./models/stable_audio_small_finetuned_custom/checkpoints/LATEST_CHECKPOINT.ckpt` (replace with actual name) and your model config used for fine-tuning is `configs/model_config_small_finetune.json`.

    Unwrap the fine-tuned checkpoint:
    ```bash
    # (Inside stable-audio-tools directory, with conda env active)
    python unwrap_model.py \
        --model-config ../configs/model_config_small_finetune.json \
        --ckpt-path ../models/stable_audio_small_finetuned_custom/checkpoints/LATEST_CHECKPOINT.ckpt \
        --name my_finetuned_unwrapped_model
    # The unwrapped model will be saved (e.g., my_finetuned_unwrapped_model.ckpt in the stable-audio-tools directory, move it to your project's models/ folder)
    # mv my_finetuned_unwrapped_model.ckpt ../models/
    ```
    Then run Gradio (from your project root `open-small-fine-tune`):
    ```bash
    conda activate stable-audio
    python stable-audio-tools/run_gradio.py \
        --model-config ./configs/model_config_small_finetune.json \
        --ckpt-path ./models/my_finetuned_unwrapped_model.ckpt \
        # Add other flags like --model-half if desired
    ```

*   **Custom Inference Script:**
    For more programmatic control, you would load your **unwrapped** fine-tuned model.
    ```python
    # Hypothetical Python script snippet (details depend on stable-audio-tools API)
    # from stable_audio_tools.inference import load_model # Or a similar function
    # from stable_audio_tools.inference.generation import generate_diffusion_cond

    # model_path = "./models/my_finetuned_unwrapped_model.ckpt"
    # config_path = "./configs/model_config_small_finetune.json" # The same config used for fine-tuning

    # model, model_config_dict = load_model(config_path=config_path, ckpt_path=model_path, verbose=True) # Example loading
    # model = model.to("cuda")


    # prompt = "a fine-tuned sound of a cat meowing like a dog"
    # conditioning = [{"prompt": prompt, "seconds_total": 10}] # Match structure from HF example
    # audio_output = generate_diffusion_cond(
    # model,
    # conditioning=conditioning,
    # sample_size=model_config_dict["sample_size"],
    # # ... other parameters like steps, sampler_type, device from HF example ...
    # )

    # # Save or play audio_output (see full example in README or HF for post-processing)
    # # import torchaudio
    # # from einops import rearrange
    # # output = rearrange(audio_output, "b d n -> d (b n)")
    # # output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    # # torchaudio.save("output_finetuned.wav", output, model_config_dict["sample_rate"])
    ```
    **Action:** Consult `stable-audio-tools` (especially the `run_gradio.py` script and any inference example notebooks) for specific inference APIs. The Hugging Face model card for `stable-audio-open-small` also has a good inference example.

## Important Notes

*   **Configuration is Key:** The JSON configuration files (`dataset_config.json`, `model_config_small_finetune.json`) are critical. Their structure and parameters must match what `stable-audio-tools` expects. Refer to the [official `stable-audio-tools` documentation and examples](https://github.com/Stability-AI/stable-audio-tools) extensively.
*   **GPU Memory:** Fine-tuning can be VRAM intensive. Adjust `batch_size`, `sample_size` (in model config), and consider using gradient accumulation (`--accum-batches` in `train.py`) if you run into memory issues.
*   **WandB:** Weights & Biases is highly recommended for tracking experiments, visualizing outputs, and comparing runs.
*   **Model Unwrapping:** Pay attention to when a "wrapped" (training checkpoint) vs. "unwrapped" (inference-ready model) checkpoint is needed. The `unwrap_model.py` script is important here.

This `README.md` provides a general outline. You will need to adapt paths, filenames, and specific configurations based on the exact requirements of `stable-audio-tools` and the "stable-audio-open-small" model. Good luck! 