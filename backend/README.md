# Backend image analysis

## Connect the trained ViT checkpoint

The ViT notebook trains a binary `fake` / `real` classifier. After the training and evaluation cell finishes successfully, run the final **backend model export** cell in `medical_deepfake_vitb16_fixed_(3).ipynb`. It writes the model weights, `config.json`, and matching image processor into:

```text
backend/models/medical_deepfake_vitb16/
```

The model directory is git-ignored because it contains large generated weights. Install backend dependencies from this directory, then add this setting to `backend/.env`:

```dotenv
MODEL_PATH=models/medical_deepfake_vitb16
```

Relative model paths are resolved from the backend directory. `server.py` loads `.env` before importing the inference service. Restart the API after exporting the model and setting `MODEL_PATH`.

When `MODEL_PATH` is not set, the backend uses its ELA/noise heuristic rather than a trained model. The ViT classifier evaluates the original RGB upload; ELA is generated separately for the visual result and heuristic features. Its class labels must be exactly `fake` and `real`.

The notebook currently records a `KeyboardInterrupt` during training. That run did not reach its model export step, and the backend will not use ViT until training completes and the export cell has run.
