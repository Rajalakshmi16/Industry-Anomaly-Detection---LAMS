# Multi-Scale Industrial Anomaly Detection

A PyTorch implementation for multi-scale industrial visual anomaly detection and localization under domain shifts (illumination and viewpoint variations).

---

## 1. Project Structure

```text
├── config/             # Default configuration settings
├── datasets/           # Dataset loaders (AeBAD-S, AeBAD-V, MVTec)
├── method_config/      # Model and training hyperparameter YAMLs
├── models/             # Neural network architectures & routing modules
├── paper_figures/      # Pre-computed visualization figures
├── paper_tables/       # Experimental metric summaries (CSV)
├── tools/              # Training, evaluation, and benchmark scripts
├── utils/              # Helper utilities, metrics, and feature extractors
├── main.py             # Main entry point
├── run_lightweight_experiment.py  # Unified experiment runner
└── requirements.txt    # Python dependencies
```

---

## 2. Prerequisites & Setup

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## 3. Dataset Setup

Place the dataset in the `dataset/` directory structured as follows:

```text
dataset/
└── AeBAD_S/
    ├── train/
    │   └── normal/
    ├── test/
    │   ├── same/
    │   ├── view/
    │   ├── illumination/
    │   └── background/
    └── ground_truth/
```

Ensure the dataset path is configured in `method_config/AeBAD_S/LightweightAdaptive.yaml`:
```yaml
dataset_path: './dataset'
```

---

## 4. Execution & Usage

### A. Run Hardware Efficiency Benchmark
Measures latency, FLOPs, active parameter count, and memory footprint:
```bash
python run_lightweight_experiment.py --mode benchmark
```

### B. Train the Model
Trains the multi-scale feature distillation model on normal training samples:
```bash
python run_lightweight_experiment.py --mode train
```

### C. Evaluate Anomaly Detection & Localization
Evaluates image-level AUROC and pixel-level localization across domain shifts:
```bash
python run_lightweight_experiment.py --mode eval
```

 
