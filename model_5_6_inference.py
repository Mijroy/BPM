# model_5_6_inference.py
# ------------------------------------------------------------
# Purpose
#   - Preprocess once (DICOM -> 512 canonical, then 512->img_size tensor)
#   - Later, time inference only (UNet forward) and compute:
#       * Model-6 (MLO): posterior-edge â†’ (PNL âˆ© pectoral) distance [px @512]
#       * Model-5 (per side): |PNL_CC âˆ’ PNL_MLO| using anisotropy-adjusted lengths
#
# Key points
#   - Uses your exact dicom_to_Image normalization
#   - Keeps geometry in 512Ã—512 for all distance calculations
#   - No changes to existing model/util functions; no disk I/O
# ------------------------------------------------------------

import os
import numpy as np
from PIL import Image
import pydicom
import torch
import torch.nn.functional as F
from scipy import ndimage

from segmammo_model.unet import UNet
from pnl_utils import find_closest_point_in_mask, find_end_of_line


# ---------------------------
# DICOM -> 8-bit PIL
# ---------------------------
def dicom_to_Image(dicom_file):
    pixel_array = dicom_file.pixel_array
    pixel_array = pixel_array - np.min(pixel_array)
    if np.max(pixel_array) != 0:
        pixel_array = pixel_array / np.max(pixel_array)
    pixel_array = (pixel_array * 255.999).astype(np.uint8)
    return Image.fromarray(pixel_array)


# ---------------------------
# PREPROCESS
# ---------------------------
def preprocess_four_dcm_to_512(image_paths, save_size_512: int = 512):
    """
    Inputs:
      image_paths: dict {'RCC','LCC','RMLO','LMLO'} -> DICOM paths
    Returns:
      imgs_512: dict view -> uint8 array (512,512)
      zooms:    dict view -> (ZoomY, ZoomX) where ZoomY=512/orig_h, ZoomX=512/orig_w
    """
    imgs_512, zooms = {}, {}
    for view_key, dcm_path in image_paths.items():
        ds = pydicom.dcmread(dcm_path)
        pil = dicom_to_Image(ds)                       
        H0, W0 = pil.size[1], pil.size[0]                
        ZoomY, ZoomX = save_size_512 / H0, save_size_512 / W0
        # Same call as your PNG generator (default resample)
        pil512 = pil.resize((save_size_512, save_size_512))
        imgs_512[view_key] = np.array(pil512, dtype=np.uint8)
        zooms[view_key] = (ZoomY, ZoomX)
    return imgs_512, zooms


def tensors_from_imgs_512(imgs_512, img_size: int = 256, aug_method: str = 'none', device=None):
    """
    Make a batch tensor in [0,1], resizing 512 -> img_size with the SAME method as training.
    Here we lock to BICUBIC explicitly to match Pillow 9.3.0's default in your env.
    Returns:
      views: list of view keys in batch order
      batch: torch.FloatTensor (N,C,img_size,img_size) on device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    views = list(imgs_512.keys())
    tensors = []

    for k in views:
        arr_512 = imgs_512[k]                      # (512,512) uint8
        pil_512 = Image.fromarray(arr_512)
        pil_rs  = pil_512.resize((img_size, img_size), resample=Image.BICUBIC)
        arr_rs  = np.array(pil_rs, dtype=np.uint8)

        if aug_method == 'channelize':
            arr_rs = np.stack([arr_rs, arr_rs, arr_rs], axis=0)  # (3,H,W)
        else:
            arr_rs = arr_rs[np.newaxis, ...]                     # (1,H,W)

        t = torch.from_numpy(arr_rs.astype(np.float32) / 255.0)
        tensors.append(t)

    batch = torch.stack(tensors, dim=0).to(device)  # (N,C,H,W)
    return views, batch



def preprocess_for_seg_models(image_paths, canonical_size: int = 512,
                              img_size: int = 256, aug_method: str = 'none', device=None):
    """
    Convenience wrapper to do *only* preprocessing. Use this once up-front.
    Returns a dict you can cache and reuse for timing inference:
      {
        'imgs_512': dict,     # view -> uint8 (512,512)
        'zooms': dict,        # view -> (ZoomY, ZoomX)
        'views': list,        # order used in the batch tensor
        'batch': torch.Tensor # (N,C,img_size,img_size) on device
      }
    """
    imgs_512, zooms = preprocess_four_dcm_to_512(image_paths, save_size_512=canonical_size)
    views, batch = tensors_from_imgs_512(imgs_512, img_size=img_size, aug_method=aug_method, device=device)
    return {'imgs_512': imgs_512, 'zooms': zooms, 'views': views, 'batch': batch}


# ---------------------------
# INFERENCE (time this only)
# ---------------------------
def load_seg_model(model_path: str, aug_method: str = 'none', device=None):
    """
    Load the segmentation UNet (same code path as your training/inference).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_channel = 3 if aug_method == 'channelize' else 1
    model = UNet(n_channel, n_classes=1, device=device)
    model = torch.nn.DataParallel(model).to(device)
    ckpt = torch.load(model_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


def infer_pred_masks_512(model, views, batch, back_size_512: int = 512):
    """
    Forward -> sigmoid -> threshold 0.5 -> upsample back to 512 (nearest).
    Returns dict view -> uint8 mask {0,255} at (512,512)
    """
    with torch.no_grad():
        logits = model(batch)                        
        probs = torch.sigmoid(logits)
        bins = (probs >= 0.5).float()
        up = F.interpolate(bins, size=(back_size_512, back_size_512), mode='nearest')
        masks = (up.squeeze(1).cpu().numpy().astype(np.uint8)) * 255
    return {k: masks[i] for i, k in enumerate(views)}


# ---------------------------
# METRICS (Model-6 & Model-5) in 512 space
# ---------------------------
def infer_view_side(key: str):
    k = key.upper()
    if k.endswith('RMLO'): return ('mlo', 'R')
    if k.endswith('LMLO'): return ('mlo', 'L')
    if k.endswith('RCC'):  return ('cc',  'R')
    if k.endswith('LCC'):  return ('cc',  'L')
    return ('unknown', '?')


def nipple_centroid_from_pred(mask_512: np.ndarray):
    """
    Simple, human-written logic:
      - label connected components on the predicted union mask
      - choose the smallest component (nipple is small; pec is large)
      - centroid = mean of pixel coordinates
    """
    labeled, n = ndimage.label(mask_512 > 0)
    if n == 0:
        return None, None
    if n == 1:
        coords = np.argwhere(labeled == 1)
        return coords.mean(axis=0)
    areas = [(i, (labeled == i).sum()) for i in range(1, n+1)]
    i_small = sorted(areas, key=lambda x: x[1])[0][0]
    coords = np.argwhere(labeled == i_small)
    return coords.mean(axis=0)


def compute_model5_model6(imgs_512, pred_masks_512, zooms):
    """
    All math in 512Ã—512 space.

    Logic:
      1) Find nipple center (cy,cx) from predicted mask (smallest blob).
      2) MLO:
         - target = closest point in predicted pec mask to nipple
         - PNL length (MLO) = anisotropy-adjusted distance nippleâ†’target
         - Model-6 = horizontal distance from posterior edge to target:
             RMLO: edge = left => distance = x
             LMLO: edge = right => distance = (W-1) - x
      3) CC:
         - target from find_end_of_line (nippleâ†’chest wall guided by intensities)
         - PNL length (CC) = anisotropy-adjusted distance nippleâ†’target
      4) Model-5 (per side) = |PNL_CC âˆ’ PNL_MLO|

    Returns:
      {
        'left':  {'model5_diff_adj_px', 'model6_edge_px'},
        'right': {'model5_diff_adj_px', 'model6_edge_px'},
        'per_view': {view: {'nipple':(cy,cx), 'target':(y,x) or None,
                            'pnl_len_adj_px':float, 'm6_edge_px':float(opt)}}
      }
    """
    pnl_len_adj = {}   # view -> adjusted PNL length (px)
    mlo_edge_px = {}   # view (MLO) -> posterior-edge distance (px)
    per_view = {}

    for view_key, mask in pred_masks_512.items():
        view, side = infer_view_side(view_key)
        if view == 'unknown':
            continue

        img = imgs_512[view_key]                   # (512,512) uint8 image
        y_by_x = zooms[view_key][0] / zooms[view_key][1]  # ZoomY/ZoomX

        cy, cx = nipple_centroid_from_pred(mask)
        if cy is None:
            pnl_len_adj[view_key] = None
            if view == 'mlo': mlo_edge_px[view_key] = None
            per_view[view_key] = {'nipple': None, 'target': None, 'pnl_len_adj_px': None}
            continue

        if view == 'mlo':
            my, mx, dist_adj = find_closest_point_in_mask(
                mask=mask, point_y=cy, point_x=cx, y_by_x_zoom=y_by_x, mask_threshold=128
            )
            if my is None:
                pnl_len_adj[view_key] = None
                mlo_edge_px[view_key] = None
                per_view[view_key] = {'nipple': (float(cy), float(cx)), 'target': None, 'pnl_len_adj_px': None}
            else:
                pnl_len_adj[view_key] = float(dist_adj)
                W = mask.shape[1]
                edge_dist = float(mx if side == 'R' else (W - 1) - mx)  # Model-6
                mlo_edge_px[view_key] = edge_dist
                per_view[view_key] = {'nipple': (float(cy), float(cx)),
                                      'target': (float(my), float(mx)),
                                      'pnl_len_adj_px': float(dist_adj),
                                      'm6_edge_px': edge_dist}

        elif view == 'cc':
            my, mx, dist_adj = find_end_of_line(
                mask=mask, image=img, point_y=cy, point_x=cx, y_by_x_zoom=y_by_x
            )
            pnl_len_adj[view_key] = None if my is None else float(dist_adj)
            per_view[view_key] = {'nipple': (float(cy), float(cx)),
                                  'target': None if my is None else (float(my), float(mx)),
                                  'pnl_len_adj_px': None if my is None else float(dist_adj)}

    out = {'left': {}, 'right': {}, 'per_view': per_view}
    # Left pairing: LCC â†” LMLO
    LCC, LMLO = pnl_len_adj.get('LCC'), pnl_len_adj.get('LMLO')
    out['left']['model5_diff_adj_px'] = (abs(LCC - LMLO) if LCC is not None and LMLO is not None else None)
    out['left']['model6_edge_px']     = mlo_edge_px.get('LMLO', None)
    # Right pairing: RCC â†” RMLO
    RCC, RMLO = pnl_len_adj.get('RCC'), pnl_len_adj.get('RMLO')
    out['right']['model5_diff_adj_px'] = (abs(RCC - RMLO) if RCC is not None and RMLO is not None else None)
    out['right']['model6_edge_px']     = mlo_edge_px.get('RMLO', None)

    return out


# ---------------------------
# INFERENCE-ONLY ENTRYPOINT
# ---------------------------
def inference_only_from_preprocessed(model, preproc_state, back_size_512: int = 512):
    """
    Use this during timing:
      - model: loaded UNet (from load_seg_model)
      - preproc_state: result of preprocess_for_seg_models(...)
    Returns:
      results (dict as in compute_model5_model6)
    """
    views = preproc_state['views']
    batch = preproc_state['batch']
    imgs_512 = preproc_state['imgs_512']
    zooms = preproc_state['zooms']

    pred_masks_512 = infer_pred_masks_512(model, views, batch, back_size_512=back_size_512)
    return compute_model5_model6(imgs_512, pred_masks_512, zooms)