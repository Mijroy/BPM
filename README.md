# Breast Position Monitoring (BPM) Classification Models

AI models for automated assessment of mammography positioning quality and anatomical measurements.

## Models Overview

**Models 1-4: Classification Models**
- Model 1: Pectoralis muscle visibility assessment
- Model 2: Inframammary fold (IMF) visibility detection  
- Model 3: Nipple in profile evaluation
- Model 4: Retromammary fat sufficiency analysis

**Models 5-6: Measurement Models**
- Model 5: PNL (Posterior Nipple Line) difference calculation between CC and MLO views
- Model 6: Pectoralis muscle extension distance measurement

## Input/Output

**Input:** DICOM mammography images (LCC, RCC, LMLO, RMLO views)

**Output:**
- Models 1-4: Probability scores (0-1)
- Models 5-6: Distance measurements (millimeters)

## Key Files

- `test_single.py` - Main inference script for integrated models 1-6
- `pnl_utils.py` - Utility functions for PNL calculations and geometric measurements
- `simple_per_image_postprocess.py` - Batch postprocessing for prediction masks
- `models_vit/` - Vision transformer model implementations
- `segmammo_model/` - UNet segmentation model

## Usage

### Single Image Inference
```bash
python test_single.py \
    --rcc_image path/to/rcc.dcm \
    --lcc_image path/to/lcc.dcm \
    --rmlo_image path/to/rmlo.dcm \
    --lmlo_image path/to/lmlo.dcm \
    --device cuda \
    --output_dir ./results
```

## Requirements

- Python 3.8+
- PyTorch 2.1.2+cu118
- CUDA 11.8
- See requirements for full dependency list


## Model Performance

- Models 1-4: Binary classification with probability output 0-1
- Model 5: PNL difference measurements within 10mm accuracy
- Model 6: Edge distance measurements in millimeters

## License
Internal use - Mass General Brigham AI
