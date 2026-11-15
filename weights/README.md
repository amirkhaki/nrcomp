# ELIC Checkpoints

This directory houses the ELIC model checkpoints for all 6 operational rate-distortion lambdas.
Model weights are downloaded from the community repository's verified Google Drive mirrors using:
```bash
python scripts/download_weights.py --output-dir weights/
```
Inference-ready checkpoints (with updated entropy model CDF parameters generated via `updata.py`) are placed in `weights/updated/`.

Check all SHA-256 hashes against `weights/CHECKSUMS.txt`.
