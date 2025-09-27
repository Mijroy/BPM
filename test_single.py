#!/usr/bin/env python3
"""
FFDM image inference script with Models 1-6
Following the exact original pipeline for Models 5-6
"""

from __future__ import absolute_import, division, print_function

import os
import time
import argparse
import tempfile
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision import transforms
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import label as ndimage_label
from functools import partial
import logging
import pydicom
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import io
from pathlib import Path
from skimage.draw import line
import shutil
import models_vit
from segmammo_model.unet import UNet
from pnl_utils import find_closest_point_in_mask, find_end_of_line

logger = logging.getLogger(__name__)

# ===================================================================
# MODEL 1-4 FUNCTIONS (UNCHANGED)
# ===================================================================

def load_weights(model, weight_path, args):
    pretrained_weights = torch.load(weight_path, map_location=torch.device('cpu'))
    if args.stage=='train':
        pretrained_weights = pretrained_weights['model']
    model_weights = model.state_dict()

    load_weights = {k: v for k, v in pretrained_weights.items() if k in model_weights}

    print("load weights")
    for k, _ in load_weights.items():
        print(k)

    model_weights.update(load_weights)
    model.load_state_dict(model_weights)
    return model

def setup_model(model_path, device):
    """Setup single model"""
    model = models_vit.vit_base_patch16(
        num_classes=1,
        drop_path_rate=0.1,
        global_pool=True,
    )
    
    class Args:
        stage = 'test'
    
    args = Args()
    model = load_weights(model, model_path, args) 
    
    model.to(device)
    model.eval()
    return model

def get_single_transform(random_crop='True', img_size=224):
    """Get transform for single image based on criteria"""
    if random_crop=='True':
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4978],std=[0.2449])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4978],std=[0.2449])
        ])
    return transform

def preprocess_all_images(image_paths, transforms_dict, normalization_flag="False", device='cuda'):
    """
    Preprocess all images for Models 1-4 ONLY
    """
    preprocessed_images = {}
    
    for view, path in image_paths.items():
        if not os.path.exists(path):
            continue
            
        preprocessed_images[view] = {'base': None}
        
        # Process for each criteria
        for criteria in ['nipple', 'fat', 'ccpec', 'imf']:
            
            if path.endswith('.dcm'):
                # Read DICOM and process
                ds = pydicom.dcmread(path)
                img = np.float32(ds.pixel_array)
                
                # Normalize
                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                
                # Apply cropping based on criteria
                filename = os.path.basename(path)
                h_ori, w_ori = img.shape
                
                if criteria == 'nipple':
                    # No cropping for nipple
                    pass
                    
                elif criteria in ['fat', 'ccpec']:
                    # Middle quadrant cropping
                    if '.lcc.' in filename or '.lmlo' in filename:
                        img = img[h_ori//4:h_ori//4+h_ori//2, 0:w_ori//2]
                    elif '.rcc.' in filename or '.rmlo.' in filename:
                        img = img[h_ori//4:h_ori//4+h_ori//2, -w_ori//2:]
                        
                elif criteria == 'imf':
                    # Bottom quadrant cropping
                    if '.lcc.' in filename or '.lmlo' in filename:
                        img = img[-h_ori//2:, 0:w_ori//2]
                    elif '.rcc.' in filename or '.rmlo.' in filename:
                        img = img[-h_ori//2:, -w_ori//2:]
                
                # Simulate PNG save/load process
                buffer = io.BytesIO()
                plt.imsave(buffer, img, cmap='gray', format='png')
                buffer.seek(0)
                
                # Load it back
                imageData = Image.open(buffer).convert("RGB")
                imageData = np.array(imageData)
                imageData = imageData[:,:,0]  # Get single channel
                
            else:
                # PNG file - read as is
                imageData = Image.open(path).convert("RGB")
                imageData = np.array(imageData)
                imageData = imageData[:,:,0]
                
                # Skip if PNG is from wrong preprocessing folder
                if '/img_crop_ccpec/' in path and criteria not in ['ccpec', 'fat']:
                    continue
                elif '/img_crop/' in path and criteria != 'imf':
                    continue
                elif '/img_proc_wocentralize/' in path and criteria != 'nipple':
                    continue
            
            # Continue with processing
            if normalization_flag == "True":
                imageData = np.float32(imageData) / np.float32(imageData.max())
            
            # Zoom to 256x256
            x, y = imageData.shape[0], imageData.shape[1]
            imageData = zoom(imageData, (256 / x, 256 / y), order=3)
            
            # Convert to PIL Image
            imageData = Image.fromarray(imageData).convert("RGB")
            
            # Apply transform and move to device
            transform = transforms_dict[criteria]
            tensor = transform(imageData).unsqueeze(0).to(device)
            
            preprocessed_images[view][criteria] = tensor
    
    return preprocessed_images

# ===================================================================
# MODEL 5-6 PIPELINE FUNCTIONS (FOLLOWING ORIGINAL)
# ===================================================================

def dicom_to_Image(pixel_array):
    """Convert DICOM pixel array to normalized 8-bit image (from create_pngs.py)"""
    pixel_array = pixel_array - np.min(pixel_array)
    if np.max(pixel_array) != 0:
        pixel_array = pixel_array / np.max(pixel_array)
    pixel_array = (pixel_array * 255.999).astype(np.uint8)
    return Image.fromarray(pixel_array)

def preprocess_for_segmentation(image_paths, tmp_dir, save_size=512, nipple_diameter=30):
    """
    Preprocess images following create_pngs.py pipeline
    Save to tmp directory and return paths and zoom factors
    """
    print("\n[PIPELINE] Step 1: Converting DICOM to PNG (512x512) like create_pngs.py...")
    
    saved_files = {}
    zooms = {}
    
    for view, path in image_paths.items():
        if not os.path.exists(path):
            continue
            
        # Read DICOM
        if path.endswith('.dcm'):
            ds = pydicom.dcmread(path)
            image = dicom_to_Image(ds.pixel_array)
        else:
            # Already PNG
            image = Image.open(path).convert("L")
        
        # Get original dimensions (PIL convention: width x height)
        original_width, original_height = image.size
        
        # Calculate zoom factors (numpy array convention)
        zoom_y = save_size / original_height  # HEIGHT becomes y in numpy
        zoom_x = save_size / original_width   # WIDTH becomes x in numpy
        
        # FLAG: Following create_pngs.py zoom calculation exactly
        zooms[view] = (zoom_y, zoom_x)
        
        # Resize to 512x512
        image_resized = image.resize((save_size, save_size))
        
        # Save the resized images
        image_filename = os.path.join(tmp_dir, f"{view}_image.png")
        image_resized.save(image_filename)
        
        saved_files[view] = {
            'image': image_filename,
            'zoom_y': zoom_y,
            'zoom_x': zoom_x
        }
        
        print(f"  {view}: Original {original_width}x{original_height} -> 512x512, "
              f"zoom_y={zoom_y:.4f}, zoom_x={zoom_x:.4f}")
    
    return saved_files, zooms

def load_seg_model(model_path: str, device=None):
    """Load segmentation model (following main.py)"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    n_channel = 1  # No channelize augmentation for inference
    model = UNet(n_channel, n_classes=1, device=device)
    model = torch.nn.DataParallel(model).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

def run_segmentation_inference(model, saved_files, tmp_dir, img_size=256, device='cuda'):
    """
    Run segmentation inference following inference.py
    Process at 256x256, save predictions as PNG
    """
    print("\n[PIPELINE] Step 1: Running segmentation inference at 256x256...")
    
    pred_files = {}
    
    with torch.no_grad():
        for view, files_dict in saved_files.items():
            # Load 512x512 image
            img_512 = Image.open(files_dict['image']).convert("L")
            
            # Resize to 256x256 for model input
            img_256 = img_512.resize((img_size, img_size))
            
            # Convert to tensor
            img_array = np.array(img_256).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array[np.newaxis, np.newaxis, ...]).to(device)
            
            # Run model
            mask_pred = model(img_tensor)
            
            # Apply sigmoid and threshold
            pred = torch.sigmoid(mask_pred).round()
            
            # Convert to numpy
            numpy_pred = pred.cpu().numpy()[0, 0]
            
            # Save prediction as PNG (0-255)
            pred_image = (numpy_pred * 255).astype('uint8')
            pred_pil = Image.fromarray(pred_image)
            
            # FLAG: Saving at 256x256 as per inference.py
            pred_filename = os.path.join(tmp_dir, f"{view}_pred.png")
            pred_pil.save(pred_filename)
            
            pred_files[view] = pred_filename
            
            print(f"  {view}: Prediction saved to {pred_filename}")
    
    return pred_files

def get_centroid(mask):
    """
    Compute the centroid of a binary mask (from create_line_gt.py)
    """
    mask = mask.astype(np.float64)
    y, x = np.indices(mask.shape)
    
    total_intensity = mask.sum()
    if total_intensity == 0:
        return None, None
    
    cx = (mask * x).sum() / total_intensity
    cy = (mask * y).sum() / total_intensity
    
    return cy, cx

def remove_border(binary_mask):
    """
    Remove components touching the border (from run_outcomes.py)
    """
    from scipy import ndimage
    
    # Label the image
    labeled_array, num_features = ndimage.label(binary_mask)
    
    # Find labels touching the border
    border_labels = set()
    # Check the first and last row
    border_labels.update(np.unique(labeled_array[0, :]))
    border_labels.update(np.unique(labeled_array[-1, :]))
    # Check the first and last column
    border_labels.update(np.unique(labeled_array[:, 0]))
    border_labels.update(np.unique(labeled_array[:, -1]))
    
    # Remove border touching labels
    # Note that label 0 is the background, so we skip it
    for label in border_labels:
        if label != 0:
            labeled_array[labeled_array == label] = 0
    
    # Convert back to binary mask
    cleaned_mask = labeled_array > 0
    return cleaned_mask

def extract_nipple_and_pec(mask_pred):
    """
    Extract nipple and pectoral masks following run_outcomes.py logic
    Returns: nipple_mask, pec_mask
    """
    # FLAG: Following exact run_outcomes.py logic
    # Convert to binary
    mask_binary = mask_pred > 127
    
    # Remove border components to get nipple
    nipple_mask = remove_border(mask_binary)
    
    # Pectoral is XOR of original and nipple (everything except nipple)
    pec_mask = np.logical_xor(mask_binary, nipple_mask)
    
    return nipple_mask, pec_mask

def compute_pnl_metrics(saved_files, pred_files, zooms, mm_per_px=0.10):
    """
    Compute PNL metrics following create_line_gt.py logic
    Process at 256x256 resolution as per original pipeline
    """
    print("\n[PIPELINE] Step 2: Computing PNL metrics at 256x256 resolution (original pipeline)...")
    
    pnl_lengths = {}
    mlo_edge_distances = {}
    per_view_results = {}
    
    for view in pred_files:
        # Load images at 256x256
        pred_mask = np.array(Image.open(pred_files[view]))
        img_path = saved_files[view]['image']
        
        # Resizing to 256x256 for metric computation
        img_512 = np.array(Image.open(img_path))
        img_256 = cv2.resize(img_512, (256, 256))
        
        # Get zoom factors
        zoom_y, zoom_x = zooms[view]
        y_by_x_zoom = zoom_y / zoom_x
        
        nipple_mask, pec_mask = extract_nipple_and_pec(pred_mask)
        
        # Check if we have valid masks
        if nipple_mask.sum() <= 0:
            print(f"  {view}: No nipple detected")
            per_view_results[view] = {"status": "NO_NIPPLE_DETECTED"}
            continue
        
        # Get nipple centroid
        cy, cx = get_centroid(nipple_mask.astype(np.float64))
        
        if cy is None or cx is None:
            print(f"  {view}: Could not compute nipple centroid")
            per_view_results[view] = {"status": "NO_NIPPLE_CENTROID"}
            continue
        
        print(f"  {view}: Nipple at ({cy:.2f}, {cx:.2f})")
        
        # Determine view type and side
        view_lower = view.lower()
        is_mlo = 'mlo' in view_lower
        is_cc = 'cc' in view_lower
        is_left = view_lower.startswith('l')
        
        if is_mlo:
            # MLO: Find closest point in pectoral muscle
            my, mx, dist = find_closest_point_in_mask(
                pec_mask.astype(np.uint8) * 255, cy, cx, y_by_x_zoom, mask_threshold=1
            )
            
            if my is None:
                print(f"  {view}: No pectoral muscle found")
                per_view_results[view] = {"status": "NO_PECTORAL_IN_PRED"}
                continue
            
            # Store PNL length
            pnl_lengths[view] = dist
            
            # FLAG: Model 6 - distance to posterior edge
            W = pec_mask.shape[1]
            if is_left:
                edge_dist = float(mx) 
            else:
                edge_dist = float(W - 1 - mx)
            
            mlo_edge_distances[view] = edge_dist
            
            print(f"  {view}: PNL={dist:.2f}px, EdgeDist={edge_dist:.2f}px")
            
            per_view_results[view] = {
                "status": "OK",
                "pnl_length_px": dist,
                "edge_distance_px": edge_dist
            }
            
        elif is_cc:
            # CC: Use find_end_of_line
            my, mx, dist = find_end_of_line(
                pec_mask.astype(np.uint8) * 255, img_256, cy, cx, y_by_x_zoom
            )
            
            if my is None:
                print(f"  {view}: No target found")
                per_view_results[view] = {"status": "NO_TARGET_FOUND"}
                continue
            
            # Store PNL length
            pnl_lengths[view] = dist
            
            print(f"  {view}: PNL={dist:.2f}px")
            
            per_view_results[view] = {
                "status": "OK",
                "pnl_length_px": dist
            }
    
    # Compute Model 5 differences - ORIGINAL PIPELINE: All metrics at 256x256 resolution
    out = {'left': {}, 'right': {}, 'per_view': per_view_results}
    
    # Left side
    if 'LCC' in pnl_lengths and 'LMLO' in pnl_lengths:
        diff_px = abs(pnl_lengths['LCC'] - pnl_lengths['LMLO'])
        out['left']['model5_diff_px'] = diff_px
        out['left']['model5_diff_mm'] = diff_px * mm_per_px
    
    # Model 6 for left
    if 'LMLO' in mlo_edge_distances:
        edge_px = mlo_edge_distances['LMLO']
        out['left']['model6_edge_px'] = edge_px
        out['left']['model6_edge_mm'] = edge_px * mm_per_px
    
    # Right side
    if 'RCC' in pnl_lengths and 'RMLO' in pnl_lengths:
        diff_px = abs(pnl_lengths['RCC'] - pnl_lengths['RMLO'])
        out['right']['model5_diff_px'] = diff_px
        out['right']['model5_diff_mm'] = diff_px * mm_per_px
    
    # Model 6 for right
    if 'RMLO' in mlo_edge_distances:
        edge_px = mlo_edge_distances['RMLO']
        out['right']['model6_edge_px'] = edge_px
        out['right']['model6_edge_mm'] = edge_px * mm_per_px
    
    return out

# ===================================================================
# MAIN INTEGRATED INFERENCE
# ===================================================================

def load_all_models(model_configs, device):
    """Load all models (EXCLUDED FROM TIMING)"""
    print("Loading Models 1-4...")
    start_time = time.time()
    
    models = {}
    for config in model_configs:
        print(f"  Loading {config['name']}...")
        model = setup_model(config['model_path'], device)
        models[config['name']] = {
            'model': model,
            'cls_criteria': config['cls_criteria'],
            'fold_idx': config['fold_idx'],
            'random_crop': config['random_crop']
        }
    
    load_time = time.time() - start_time
    print(f"Models 1-4 loaded in {load_time:.2f} seconds (EXCLUDED from timing)")
    
    # Warmup models
    print("Warming up models...")
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        for model_info in models.values():
            _ = model_info['model'](dummy_input)
    
    return models

def run_integrated_inference_pipeline(models_14, seg_model, image_paths, preprocessed_14, 
                                     device, output_dir, tmp_dir, saved_files=None, zooms=None):
    """Run integrated inference following exact pipeline"""
    
    results = []
    os.makedirs(output_dir, exist_ok=True)
    
    # Define which views each criteria applies to
    criteria_views = {
        'ccpec': ['RCC', 'LCC'],
        'imf': ['RMLO', 'LMLO'],
        'nipple': ['RCC', 'LCC', 'RMLO', 'LMLO'],
        'fat': ['RCC', 'LCC', 'RMLO', 'LMLO']
    }
    
    # Prepare all inference pairs for Models 1-4
    inference_pairs = []
    for model_name, model_info in models_14.items():
        cls_criteria = model_info['cls_criteria']
        applicable_views = criteria_views[cls_criteria]
        
        for view in applicable_views:
            if view in preprocessed_14 and cls_criteria in preprocessed_14[view]:
                tensor = preprocessed_14[view][cls_criteria]
                if tensor is not None:
                    inference_pairs.append((
                        model_name, model_info, view, tensor
                    ))
    
    # Count operations - seg_operations only includes inference, not preprocessing
    seg_operations = 4 if seg_model is not None and saved_files is not None else 0
    total_operations = len(inference_pairs) + seg_operations
    print(f"\nTotal operations: {total_operations} (Models 1-4: {len(inference_pairs)}, Models 5-6: {seg_operations})")
    
    # Get initial GPU memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    print("\n" + "="*60)
    print("STARTING INTEGRATED INFERENCE (TIMING BEGINS)")
    print("="*60)
    
    # ============== START TIMING HERE ==============
    start_time = time.time()
    
    all_results_data_14 = []
    model_56_output = None
    
    # Run Models 1-4 inferences
    with torch.no_grad():
        for model_name, model_info, view, image_tensor in inference_pairs:
            logits = model_info['model'](image_tensor)
            probs = logits.sigmoid()
            preds = (probs > 0.5) * 1
            all_results_data_14.append((logits, probs, preds, model_name, model_info, view))
    
    # Models 5-6: Only inference and metrics (preprocessing already done)
    if seg_model is not None and saved_files is not None:
        # Step 1: Run segmentation inference (inference.py style)
        pred_files = run_segmentation_inference(seg_model, saved_files, tmp_dir, device=device)
        
        # Step 2: Compute metrics (create_line_gt.py style)
        model_56_output = compute_pnl_metrics(saved_files, pred_files, zooms)
    
    # Ensure all GPU operations are complete
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # ============== END TIMING HERE ==============
    total_time = time.time() - start_time
    
    print("="*60)
    print("INTEGRATED INFERENCE COMPLETED (TIMING ENDS)")
    print("="*60)
    print(f"Total operations: {total_operations}")
    print(f"Total inference time: {total_time:.3f} seconds")
    print(f"Average per operation: {total_time/total_operations:.4f} seconds")
    
    # Check timing requirement
    timing_met = total_time < 5.0
    if timing_met:
        print("✅ SUCCESS: Under 5 seconds requirement")
    else:
        print("⚠️ WARNING: Exceeded 5 seconds requirement")
    
    # Process Models 1-4 results
    print("\n" + "="*80)
    print("MODELS 1-4 RESULTS")
    print("="*80)
    
    for logits, probs, preds, model_name, model_info, view in all_results_data_14:
        result = {
            'model_type': 'classification',
            'view': view,
            'model': model_name,
            'cls_criteria': model_info['cls_criteria'],
            'fold_idx': model_info['fold_idx'],
            'probability': probs.cpu().numpy()[0][0],
            'prediction': int(preds.cpu().numpy()[0][0]),
            'logits': logits.cpu().numpy()[0][0],
            'image_path': image_paths.get(view, '')
        }
        results.append(result)
        
        print(f"  {view}-{model_info['cls_criteria']:6s}: "
              f"prob={result['probability']:.4f}, pred={result['prediction']}")
    
    # Process Models 5-6 results
    if model_56_output is not None:
        print("\n" + "="*80)
        print("MODELS 5-6 RESULTS (ORIGINAL PIPELINE - 256x256 RESOLUTION)")
        print("="*80)
        
        def fmt(x):
            return "NA" if x is None else f"{x:.2f}"
        
        left = model_56_output.get('left', {})
        right = model_56_output.get('right', {})
        
        print("\nModel 5 (PNL difference - 256x256 resolution):")
        print(f" Left  (|LCC-LMLO|): {fmt(left.get('model5_diff_px'))} px = {fmt(left.get('model5_diff_mm'))} mm")
        print(f" Right (|RCC-RMLO|): {fmt(right.get('model5_diff_px'))} px = {fmt(right.get('model5_diff_mm'))} mm")
        
        print("\nModel 6 (Posterior edge distance - MLO only, 256x256 resolution):")
        print(f" LMLO: {fmt(left.get('model6_edge_px'))} px = {fmt(left.get('model6_edge_mm'))} mm")
        print(f" RMLO: {fmt(right.get('model6_edge_px'))} px = {fmt(right.get('model6_edge_mm'))} mm")
        
        # Add to results
        for side, prefix in [('left', 'L'), ('right', 'R')]:
            s = model_56_output.get(side, {})
            
            # Model 5
            results.append({
                'model_type': 'segmentation',
                'view': f'{prefix}CC_vs_{prefix}MLO',
                'model': 'model5',
                'cls_criteria': 'PNL_diff',
                'value_px': s.get('model5_diff_px', None),
                'value_mm': s.get('model5_diff_mm', None),
                'metric': 'PNL_difference'
            })
            
            # Model 6
            results.append({
                'model_type': 'segmentation',
                'view': f'{prefix}MLO',
                'model': 'model6',
                'cls_criteria': 'posterior_edge',
                'value_px': s.get('model6_edge_px', None),
                'value_mm': s.get('model6_edge_mm', None),
                'metric': 'edge_distance'
            })
    
    # Save results
    if image_paths:
        sample_image_path = list(image_paths.values())[0]
        image_identifier = os.path.basename(sample_image_path).split('.')[1] if '.' in os.path.basename(sample_image_path) else 'test'
        
        # Save main results CSV
        results_df = pd.DataFrame(results)
        results_csv_path = os.path.join(output_dir, f"2d_proc.{image_identifier}.csv")
        results_df.to_csv(results_csv_path, index=False)
        print(f"\nResults saved to: {results_csv_path}")
        
        # Save timing info
        timing_info = {
            'total_time_seconds': total_time,
            'total_operations': total_operations,
            'models_14_count': len(inference_pairs),
            'models_56_count': seg_operations,
            'avg_time_per_operation': total_time/total_operations,
            'requirement_met': timing_met,
            'processing_style': 'original_pipeline_256x256',
            'pipeline_steps': 'create_pngs->inference->create_line_gt_at_256x256'
        }
        timing_df = pd.DataFrame([timing_info])
        timing_csv_path = os.path.join(output_dir, f"2d_proc.{image_identifier}.timing.csv")
        timing_df.to_csv(timing_csv_path, index=False)
        print(f"Timing info saved to: {timing_csv_path}")
    
    return results, total_time

def main():
    # --- CLI args ---
    parser = argparse.ArgumentParser()
    # 4 views (DICOM or PNG paths)
    parser.add_argument('--rcc_image', type=str, help='Path to RCC image')
    parser.add_argument('--lcc_image', type=str, help='Path to LCC image')
    parser.add_argument('--rmlo_image', type=str, help='Path to RMLO image')
    parser.add_argument('--lmlo_image', type=str, help='Path to LMLO image')

    # IO / runtime
    parser.add_argument('--output_dir', type=str, default="/workspace/bpm_classification/result/",
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--normalization_flag', default="False", type=str)
    parser.add_argument('--tmp_dir', type=str, default="/tmp/bpm_seg/",
                        help='Temporary directory for intermediate files')

    # Segmentation (Model-5/6)
    parser.add_argument('--seg_model_path', type=str,
                        default="/workspace/bpm_classification/criteria/model_5-6/best_checkpoint_final.pt",
                        help='Path to UNet checkpoint for Model-5/6')
    parser.add_argument('--seg_img_size', type=int, default=256,
                        help='UNet input size; default 256')
    parser.add_argument('--save_size', type=int, default=512,
                        help='Resolution for preprocessing; default 512')

    args = parser.parse_args()

    # --- Device / CUDNN ---
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print("CUDA optimizations enabled")

    # --- Classifier model (1-4) configs ---
    model_configs = [
        {
            'name': 'ccpec_fold0',
            'model_path': '/workspace/bpm_classification/criteria/model_1-4/checkpoint/CCPec/910_bestauc_checkpoint.bin',
            'cls_criteria': 'ccpec',
            'fold_idx': 0,
            'random_crop': 'False'
        },
        {
            'name': 'nipple_fold0',
            'model_path': '/workspace/bpm_classification/criteria/model_1-4/checkpoint/NippleInProfile/1400_bestauc_checkpoint.bin',
            'cls_criteria': 'nipple',
            'fold_idx': 0,
            'random_crop': 'True'
        },
        {
            'name': 'fat_fold0',
            'model_path': '/workspace/bpm_classification/criteria/model_1-4/checkpoint/RetromammaryFatSufficient/1820_bestauc_checkpoint.bin',
            'cls_criteria': 'fat',
            'fold_idx': 0,
            'random_crop': 'False'
        },
        {
            'name': 'imf_fold0',
            'model_path': '/workspace/bpm_classification/criteria/model_1-4/checkpoint/IMFVisible/630_bestauc_checkpoint.bin',
            'cls_criteria': 'imf',
            'fold_idx': 0,
            'random_crop': 'False'
        }
    ]

    # --- Gather valid image paths provided ---
    image_paths = {}
    if args.rcc_image and os.path.exists(args.rcc_image):
        image_paths['RCC'] = args.rcc_image
    if args.lcc_image and os.path.exists(args.lcc_image):
        image_paths['LCC'] = args.lcc_image
    if args.rmlo_image and os.path.exists(args.rmlo_image):
        image_paths['RMLO'] = args.rmlo_image
    if args.lmlo_image and os.path.exists(args.lmlo_image):
        image_paths['LMLO'] = args.lmlo_image

    if not image_paths:
        print("Error: No valid image files provided")
        return
    print(f"Processing images: {list(image_paths.keys())}")

    # --- Verify classifier weights ---
    missing = [cfg['model_path'] for cfg in model_configs if not os.path.exists(cfg['model_path'])]
    if missing:
        print("Error: Missing model files:")
        for m in missing:
            print(f"  {m}")
        return

    # --- Verify seg model ---
    use_seg = True
    if args.seg_model_path is None or not os.path.exists(args.seg_model_path):
        print(f"Warning: seg model not found at {args.seg_model_path}. Skipping Model-5/6.")
        use_seg = False

    # Create temp directory
    os.makedirs(args.tmp_dir, exist_ok=True)
    print(f"Using temp directory: {args.tmp_dir}")

    print("All required files found")
    print("=" * 60)

    # ====================================================
    # PHASE 1: Load and prepare everything (EXCLUDED FROM TIMING)
    # ====================================================
    
    print("\n" + "="*60)
    print("PHASE 1: LOADING AND PREPROCESSING (NOT TIMED)")
    print("="*60)
    
    # Load Models 1-4
    models_14 = load_all_models(model_configs, device)
    
    # Create transforms for Models 1-4
    transforms_dict = {
        'nipple': get_single_transform('True', 224),
        'fat': get_single_transform('False', 224),
        'ccpec': get_single_transform('False', 224),
        'imf': get_single_transform('False', 224)
    }
    
    # Preprocess for Models 1-4
    print("\nPreprocessing images for Models 1-4...")
    preprocess_start = time.time()
    preprocessed_14 = preprocess_all_images(image_paths, transforms_dict, 
                                           args.normalization_flag, device)
    preprocess_time = time.time() - preprocess_start
    print(f"Models 1-4 preprocessing completed in {preprocess_time:.2f} seconds (EXCLUDED from timing)")
    
    # Load Model 5-6 and preprocess
    seg_model = None
    saved_files = None
    zooms = None
    if use_seg:
        try:
            print("\nLoading Model 5-6 (segmentation)...")
            seg_load_start = time.time()
            seg_model = load_seg_model(args.seg_model_path, device=device)
            seg_load_time = time.time() - seg_load_start
            print(f"Model 5-6 loaded in {seg_load_time:.2f} seconds (EXCLUDED from timing)")
            
            # Preprocess for segmentation (EXCLUDED from timing)
            print("\nPreprocessing images for Models 5-6...")
            seg_preprocess_start = time.time()
            saved_files, zooms = preprocess_for_segmentation(image_paths, args.tmp_dir)
            seg_preprocess_time = time.time() - seg_preprocess_start
            print(f"Models 5-6 preprocessing completed in {seg_preprocess_time:.2f} seconds (EXCLUDED from timing)")
            
        except Exception as e:
            print(f"Warning: Model-5/6 setup failed: {e}")
            import traceback
            traceback.print_exc()
            use_seg = False
    
    print("\n" + "="*60)
    print("PHASE 1 COMPLETE - ALL MODELS READY")
    print("="*60)
    
    # ====================================================
    # PHASE 2: Run timed inference
    # ====================================================
    
    print("\n" + "="*60)
    print("PHASE 2: TIMED INFERENCE (ORIGINAL PIPELINE)")
    print("  - Models 1-4: Classification inferences")
    print("  - Models 5-6: Following create_pngs -> inference -> create_line_gt at 256x256")
    print("="*60)
    
    results, total_time = run_integrated_inference_pipeline(
        models_14, seg_model, image_paths, preprocessed_14, 
        device, args.output_dir, args.tmp_dir, saved_files, zooms
    )
    
    # ====================================================
    # PHASE 3: Summary
    # ====================================================
    
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(f"Total inference time: {total_time:.3f} seconds")
    print(f"Target: < 5.0 seconds")
    print(f"Status: {'PASSED ✅' if total_time < 5.0 else 'FAILED ⚠️'}")
    print("="*80)
    

    shutil.rmtree(args.tmp_dir)
    print(f"Cleaned up {args.tmp_dir}")

if __name__ == "__main__":
    main()