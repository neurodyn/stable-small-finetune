# Fine-tuning Stable Audio Open Small

Essentially we will follow the tutorial 'Finetuning Stable Audio Open' by Lyraaaa at https://www.youtube.com/watch?v=ex4OBD_lrds&t=80s

We will start with finetuning the base model and then refine the model using the contrastive and adversial step.

```
(stable-audio) root@computer:/path/to/stable-small-finetune# python
Python 3.10.16 (main, Dec 11 2024, 16:24:50) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
```

Let's install stable audio tools. Inside `/stable-audio-tools/'
```
pip install .
```

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
#Weights and Biases Logging

You might also need to install `wandb` if you plan to use Weights & Biases for logging:
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


### 3. Dataset Configuration

### 4. Model Configuration



### 5. Training

```
python train.py --dataset-config ../config/dataset.json --model-config ../models/stable-audio-open-small-base/ckpt/base_model_config.json --save-dir checkpoints --name stable_small_finetune --checkpoint-every 1000 --batch-size 8 --precision 16-mixed --seed 128 --pretrained-ckpt-path ../models/stable-audio-open-small-base/ckpt/base_model.ckpt
```

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

## Important Notes

*   **Configuration is Key:** The JSON configuration files (`dataset_config.json`, `model_config_small_finetune.json`) are critical. Their structure and parameters must match what `stable-audio-tools` expects. Refer to the [official `stable-audio-tools` documentation and examples](https://github.com/Stability-AI/stable-audio-tools) extensively.
*   **GPU Memory:** Fine-tuning can be VRAM intensive. Adjust `batch_size`, `sample_size` (in model config), and consider using gradient accumulation (`--accum-batches` in `train.py`) if you run into memory issues.
*   **WandB:** Weights & Biases is highly recommended for tracking experiments, visualizing outputs, and comparing runs.
*   **Model Unwrapping:** Pay attention to when a "wrapped" (training checkpoint) vs. "unwrapped" (inference-ready model) checkpoint is needed. The `unwrap_model.py` script is important here.

This `README.md` provides a general outline. You will need to adapt paths, filenames, and specific configurations based on the exact requirements of `stable-audio-tools` and the "stable-audio-open-small" model. Good luck! 