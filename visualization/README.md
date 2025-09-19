# MemAE Visualization Tools

This folder contains Python scripts to generate performance charts and analysis for the Memory-Augumented Autoencoder (MemAE) anomaly detection model.

## Quick Setup

## Files Included
- `Performance_Comparison_PED1.py` - PED1 dataset performance visualization
- `Performance_Comparison_PED2.py` - PED2 dataset performance visualization

### 1. Create Virtual Environment
```bash
python3 -m venv memae_env
source memae_env/bin/activate  # On Windows: memae_env\Scripts\activate

### 2. Install required libraries
pip install pandas matplotlib numpy scikit-learn


### 3. Generate Performance Chart
# For PED2 performance comparison
python Performance_Comparison_PED2.py

# For PED1 performance comparison  
python Performance_Comparison_PED1.py



