import cv2
import numpy as np
import logging

LOG = logging.getLogger(__name__)

def apply_canny(image: np.ndarray, low_thresh: int, high_thresh: int, aperture_size: int = 3) -> np.ndarray:
    """Apply Canny edge detection."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image, threshold1=low_thresh, threshold2=high_thresh, apertureSize=aperture_size)
    return edges

def apply_sobel(image: np.ndarray, ksize: int = 3, scale: float = 1.0, delta: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Sobel edge detection.
    Returns:
        magnitude: Float magnitude map
        direction: Direction map in degrees
        sobel_combined: uint8 absolute combined sobel map
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize, scale=scale, delta=delta)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize, scale=scale, delta=delta)
    
    magnitude = cv2.magnitude(sobelx, sobely)
    direction = cv2.phase(sobelx, sobely, angleInDegrees=True)
    
    abs_sobelx = cv2.convertScaleAbs(sobelx)
    abs_sobely = cv2.convertScaleAbs(sobely)
    sobel_combined = cv2.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)
    
    return magnitude, direction, sobel_combined

def extract_contours_and_features(edge_map: np.ndarray, close_kernel_size: int = 15) -> list[dict]:
    """
    Close the edge map to form continuous boundaries and extract contours and features.
    """
    # Morphological closing to connect broken edges
    if close_kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
        # Apply a dilation then errosion to bridge the gaps in the edges
        edge_map = cv2.morphologyEx(edge_map, cv2.MORPH_CLOSE, kernel)
        
    # Find contours
    contours, _ = cv2.findContours(edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    features_list = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        # Calculate circularity
        circularity = 0.0
        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter * perimeter)
            
        features_list.append({
            'contour': cnt,
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity
        })
        
    return features_list

def count_objects_from_contours(features_list: list[dict], min_area: float = 0, max_area: float = float('inf'), min_circ: float = 0.0, circ=0.0) -> int:
    """Count valid objects based on contour features. Evaluates overlapping blobs."""
    import numpy as np
    
    valid_areas = []
    # First pass: collect areas of clearly valid, single seeds to find the typical size
    for feat in features_list:
        if min_area <= feat['area'] <= max_area and feat['circularity'] >= min_circ:
            valid_areas.append(feat['area'])
            
    # Calculate typical area (use median to ignore outliers), fallback to a reasonable size
    # Calculate typical area (use median to ignore outliers), fallback to a reasonable size
    typical_area = float(np.median(valid_areas)) if valid_areas else float(min_area * 1.5)
    if typical_area == 0 or typical_area == float('inf'):
        typical_area = 1000.0 # safety fallback

    valid_count = 0
    for feat in features_list:
        area = feat['area']
        
        if area < 50: # User's notebook uses hardcoded 50 area cutoff
            continue
            
        # Overlapping cluster: check if it's much larger than a typical single seed
        if area > 1.6 * typical_area or area > max_area:
            estimated_seeds = max(1, round(area / typical_area))
            valid_count += int(estimated_seeds)
            
        # Check standard valid single seed (User notebook drops circularity floor here)
        elif feat['circularity'] >= circ:
            valid_count += 1
            
    return valid_count
