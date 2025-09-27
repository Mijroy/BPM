import json
import glob
import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
import pydicom
import nrrd
import glob
from PIL import Image, ImageDraw
import pandas as pd 
from skimage.measure import label
from skimage.transform import rescale
from skimage.draw import line
from scipy.spatial.distance import cdist
from scipy import ndimage
from scipy.stats import norm
from scipy.optimize import minimize



def objective_function(std, points, equal_variance=False):
    thresholds = np.linspace(-10, 11, num=10000)
    std2 = std[0] if equal_variance else std[1]
    cdf_hypothesis_1 = norm.cdf(thresholds, loc=0, scale=std[0])
    cdf_hypothesis_2 = norm.cdf(thresholds, loc=1, scale=std2)
    # Calculate the True Positive Rate (TPR) and False Positive Rate (FPR)
    tpr_values = 1 - cdf_hypothesis_2
    fpr_values = 1 - cdf_hypothesis_1
    val = sum(np.min((tpr_values - tpr) ** 2 + (fpr_values - fpr) ** 2) for tpr, fpr in points)
    return val


def compute_gaussian_metrics(sensi, speci, equal_variance=False):
    fpr_points = 1 - np.array(speci)
    tpr_points = np.array(sensi)
    points = np.stack((tpr_points, fpr_points), axis=1).tolist()

    # ==========================================================================
    # here starts the fitting of parametric ROC curve to experimental points
    # ==========================================================================
    delta = 1e-6
    upper_const = 10
    # Define the range constraints for x and y
    std_constraint = {
        "type": "ineq",
        "fun": lambda x: np.array([x[0] - delta, upper_const - x[0]]),
    }
    std2_constraint = {
        "type": "ineq",
        "fun": lambda x: np.array([x[1] - delta, upper_const - x[1]]),
    }
    if equal_variance:
        constraints = [std_constraint]
        x0 = np.array([1.0])
    else:
        constraints = [std_constraint, std2_constraint]
        x0 = np.array([0.5, 0.5])

    result = minimize(objective_function, x0, args=(points,), constraints=constraints)

    if equal_variance:
        optimal_x = result.x
    else:
        optimal_x, optimal_y = result.x

    # Parameters for hypothesis 1 (mean and standard deviation)
    mean_hypothesis_1 = 0.0
    std_dev_hypothesis_1 = optimal_x

    # Parameters for hypothesis 2 (mean and standard deviation)
    mean_hypothesis_2 = 1
    std_dev_hypothesis_2 = optimal_x
    if not equal_variance:
        std_dev_hypothesis_2 = optimal_y

    # Calculate the cumulative distribution functions (CDFs) for the two distributions
    thresholds = np.linspace(-5, 10, num=1000)
    cdf_hypothesis_1 = norm.cdf(thresholds, loc=mean_hypothesis_1, scale=std_dev_hypothesis_1)
    cdf_hypothesis_2 = norm.cdf(thresholds, loc=mean_hypothesis_2, scale=std_dev_hypothesis_2)

    # Calculate the True Positive Rate (TPR) and False Positive Rate (FPR)
    tpr_values = 1 - cdf_hypothesis_2
    fpr_values = 1 - cdf_hypothesis_1
    # Find the best operating point defined as the closest point to (0,1)
    dist_squared = fpr_values**2 + (tpr_values - 1) ** 2
    min_idx = np.argmin(dist_squared)
    fpr = fpr_values[min_idx]
    tpr = tpr_values[min_idx]

    denominator = tpr_values + fpr_values
    recall_values = tpr_values
    more_than_zero = denominator > 0
    precision_values = np.divide(tpr_values, denominator, where=more_than_zero)
    precision_values[~more_than_zero] = 1
    # =======================================================
    # Compute metrics now using the operating point
    # =======================================================

    # tier 1
    auprc = -np.trapz(precision_values, recall_values)
    auc = -np.trapz(tpr_values, fpr_values)
    sensitivity = tpr
    specificity = 1 - fpr


    return {
        # Tier 1
        "auc": auc,
        "auprc": auprc,
        "sensitivity+specificity": sensitivity + specificity,
        # Tier 2
        "sensitivity": sensitivity,
        "specificity": specificity,
        # Not needed for the sake of the challenge
        "recall": tpr,
        "precision_values": precision_values,
        "recall_values": recall_values,
        "tpr_points": tpr_points,
        "fpr_points": fpr_points,
        "tpr_values": tpr_values[::-1],
        "fpr_values": fpr_values[::-1],
    }



def create_mosaic(pred, true_masks, imgs, batch_idx, save_dir, mosaic_size=1024):
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image

    assert(pred.shape[2:] == true_masks.shape[2:] == imgs.shape[2:])
    assert(pred.shape[0] == true_masks.shape[0] == imgs.shape[0])
    assert(pred.shape[1] == 1 and true_masks.shape[1] == 1 and (imgs.shape[1] == 1 or imgs.shape[1] == 3))
    assert(pred.shape[-1] == 128 or pred.shape[-1] == 256)

    if pred.shape[0] * pred.shape[2] * pred.shape[3] < mosaic_size * mosaic_size:
        return
    
    if imgs.shape[1] == 3:
        #just pick channel 0 but keep original dimensions 
        imgs = imgs[:,0:1,:,:]

    # Initialize a 512x512x3 RGB image with zeros
    mosaic = np.zeros((mosaic_size, mosaic_size, 3), dtype=np.uint8)

    # Function to fill a specific channel with 64 images in an 8x8 grid

    image_size = pred.shape[-1]
    number_per_row = mosaic_size // image_size


    def fill_channel(channel, images):
        for i in range(number_per_row):  # number_per_row images per row
            for j in range(number_per_row):  # number_per_row images per column
                # Calculate the start index for the row and column
                row_start = i * image_size
                col_start = j * image_size
                # Place each image into the channel
                mosaic[row_start:row_start+image_size, col_start:col_start+image_size, channel] = (images[i*number_per_row+j,0]*255).astype(np.uint8)

    fill_channel(0, pred)
    fill_channel(1, true_masks)
    fill_channel(2, imgs)

    # Convert the mosaic numpy array to a PIL image and save it
    mosaic_image = Image.fromarray(mosaic)
    mosaic_image.save(save_dir + 'mosaic_image_'+str(batch_idx)+'_.png')

    # Display the final mosaic
    # plt.imshow(mosaic_image)
    # plt.axis('off')  # Turn off axis numbers and ticks
    # plt.show()



def display_patient(n_batch, n_patient, mammo=True, annon = True, 
                    base_path='/workspace/data_dir/xxx/BPM/alldata/phase3_ge_origin/darwin/'):
    
    print('Example of a patient')

    if int(n_batch) in [1,2,3,4,5,6,7,8,9,10]:
        pname_base = base_path+'2d_proc_batch_'+str(n_batch)+'/data/2d_proc.'+str(n_patient)+'.'
        annon_base = base_path+'2d_proc_batch_'+str(n_batch)+'/seg_pec/2d_proc.'+str(n_patient)+'.'
    else:
        print("Invalid batch number")

    views = ['rcc','lcc','rmlo','lmlo']

    images = []

    for a_view in views:
        fname = pname_base + a_view + '.dcm'

        image = None
        if not os.path.exists(fname):
            print(f"File not found: {fname}")
            return
        else:
            try:
                image = pydicom.dcmread(fname).pixel_array
            except Exception as e:
                print(f"Failed to read {fname}: {e}")
        images.append(image)

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    if images[0] is not None: axs[0, 0].imshow(images[0],cmap='gray')
    axs[0, 0].set_title('rcc')
    axs[0, 0].axis('off')  # Disable axis
    if images[1] is not None: axs[0, 1].imshow(images[1],cmap='gray')
    axs[0, 1].set_title('lcc')
    axs[0, 1].axis('off')
    if images[2] is not None: axs[1, 0].imshow(images[2],cmap='gray')
    axs[1, 0].set_title('rmlo')
    axs[1, 0].axis('off')
    if images[3] is not None: axs[1, 1].imshow(images[3],cmap='gray')
    axs[1, 1].set_title('lmlo')
    axs[1, 1].axis('off')
    # Show the plot
    plt.show()

    if not annon:
        return
    
    annons = []
    captions = []

    for a_view in views:
        fname = annon_base + a_view + '.dcm.*'

        files = glob.glob(fname)

        if len(files) != 3:
            print("Warning: patient has less than 3 annotations for " + a_view)

        for i in range(3):
            image = None
            caption = ''
            if i < len(files):
                if not os.path.exists(files[i]):
                    print(f"File not found: {files[i]}")
                else:
                    try:
                        image, header = nrrd.read(files[i])
                        caption = a_view + ' ' + files[i].split('.')[-3]
                    except Exception as e:
                        print(f"Failed to read {files[i]}: {e}")
            annons.append(image.transpose() )
            captions.append(caption)


    fig, axs = plt.subplots(2, 6, figsize=(12, 4))
    for i in range(12):
        row = i // 6  # Integer division to get the row index
        col = i % 6   # Modulus to get the column index
        if annons[i] is not None: 
            # Use 'coolwarm' colormap, blue for low values and red for high values
            axs[row, col].imshow(annons[i], cmap='coolwarm')
        axs[row, col].set_title(captions[i])
        axs[row, col].axis('off')  
    # Show the plot
    plt.show()


def find_closest_point_in_mask(mask, point_y, point_x, y_by_x_zoom = 1, mask_threshold = 255):
    """
    Find the closest point in the mask to the given point.

    Parameters:
    - mask: A 2D numpy array where points of interest are marked, e.g., with 1.
    - point_y, point_x coordinates for the given point.
    - y_by_x_zoom: The zoom factor for the y axis relative to the x axis. this typically will be < 1 as
    size of the image is larger in y direction than in x direction 

    Returns:
    - closest_point: point_y, point_x coordinates of the closest point in the mask.
    - min_distance: The Euclidean distance to the closest point.
    """
    # Get the indices of points in the mask
    mask_points_y, mask_points_x = np.where(mask >= mask_threshold)

    if len(mask_points_y) and len(mask_points_x) > 0:
        # Compute the Euclidean distances from the given point to all points in the mask
        distances = np.sqrt((mask_points_x - point_x)**2 + ((mask_points_y - point_y)/y_by_x_zoom)**2)
        # Find the index of the minimum distance
        min_index = np.argmin(distances)
    else:
        return None, None, None

    return mask_points_y[min_index], mask_points_x[min_index], distances[min_index]

def remove_border(binary_mask):
    # Step 1: Label the image
    labeled_array, num_features = ndimage.label(binary_mask)

    # Step 2: Find labels touching the border
    border_labels = set()
    # Check the first and last row
    border_labels.update(np.unique(labeled_array[0, :]))
    border_labels.update(np.unique(labeled_array[-1, :]))
    # Check the first and last column
    border_labels.update(np.unique(labeled_array[:, 0]))
    border_labels.update(np.unique(labeled_array[:, -1]))

    # Step 3: Remove border touching labels
    # Note that label 0 is the background, so we skip it
    for label in border_labels:
        if label != 0:
            labeled_array[labeled_array == label] = 0

    # Convert back to binary mask, removing border touching structures
    cleaned_mask = labeled_array > 0
    return cleaned_mask

def find_end_of_line(mask, image, point_y, point_x, y_by_x_zoom = 1):
    """
    Compute the center of mass of the image weighted by the pixel intensity
    """
    y, x = np.indices(image.shape)
    total_intensity = image.sum()
    if total_intensity == 0:
        return None, None, None
    x_end = (image * x).sum() / total_intensity

    if x_end < image.shape[1]//2:
        x_end = 0
        y_end = (image * (image.shape[0]-1 -x) * y).sum() 
        total_intensity = (image * (image.shape[0]-1 - x)).sum()
        y_end = y_end / total_intensity
    else:
        x_end = image.shape[1] - 1
        y_end = (image * x * y).sum() 
        total_intensity = image * x    
        total_intensity = total_intensity.sum()
        y_end = y_end / total_intensity    

    # Rasterize the line
    rro, cco = line(int(point_y), int(point_x), int(y_end), int(x_end))
    rr = []
    cc = []
    # remove pairs of corresponsing points if at least one of them outside of the image
    for r, c in zip(rro, cco):
        if r >= 0 and r < mask.shape[0] and c >= 0 and c < mask.shape[1]:
            rr.append(r)
            cc.append(c)
    
    # Filter out line points not in mask
    # difference in zoom is ignored here as typically line will be drawn in x direction and error made is isignificant
    line_points_in_mask = [(r, c, np.sqrt((point_y-r)*(point_y-r)+(point_x-c)*(point_x-c))) for r, c in zip(rr, cc) if mask[r, c]]
    if len(line_points_in_mask) == 0:
        return y_end,x_end,np.sqrt((point_y-y_end)*(point_y-y_end)/y_by_x_zoom/y_by_x_zoom+(point_x-x_end)*(point_x-x_end))
    # find element in line_points_in_mask with minimum distance
    closest_point_on_line = min(line_points_in_mask, key=lambda x: x[2])
    return closest_point_on_line[0], closest_point_on_line[1], closest_point_on_line[2]



def get_all_data_info(base_path='/workspace/data_dir/BPM/alldata/phase3_ge_origin/darwin/cleaned_labels/', 
                      version='20240404_no_sd', 
                      zooms='/workspace/segmammo/data/zooms.csv'):
    assert(version>='20240404_no_sd')
    path_manifest  = base_path + version + '/all/SegPec.csv'
    df_manifest = pd.read_csv(path_manifest)
    # read CCBorder.csv
    df_CCBorder = pd.read_csv(base_path + version + '/all/CCBorder.csv')
    df_manifest = df_manifest.merge(df_CCBorder[['Image', 'Annotator', 'BorderY0', 'BorderY1']], on=['Image', 'Annotator'], how='left')    
    # read NippleBase.csv
    df_NippleBase = pd.read_csv(base_path + version + '/all/NippleBase.csv')   
    df_manifest = df_manifest.merge(df_NippleBase[['Image', 'Annotator', 'NippleY', 'NippleX','NippleLineEndY','NippleLineEndX']], on=['Image', 'Annotator'], how='left')
    # read CCPecVisible.csv
    df_CCPecVisible = pd.read_csv(base_path + version + '/all/CCPecVisible.csv')
    df_manifest = df_manifest.merge(df_CCPecVisible[['Image', 'MajorityVoting']], on=['Image'], how='left')
    if zooms is not None:
        df_zooms = pd.read_csv(zooms)
        df_manifest = df_manifest.merge(df_zooms, how='left', left_index=True, right_index=True)  

    return df_manifest


if __name__ == "__main__":

    df_manifest = get_all_data_info()
    print(df_manifest.head())



    # count number of nans
    # print(df_manifest.isna().sum())

    # print(df_manifest.columns)

    # convert_data_to_png(df_manifest)

    # 20240208

    # print(int('1') in [1,2,3,4,5,6,7,8,9,10])

    # display_patient('1','00004')