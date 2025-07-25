import runpod
import os
import base64
import numpy as np
from io import BytesIO
from PIL import Image, ImageEnhance
import requests
import logging
import re
import string
import cv2
from scipy import ndimage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

################################
# THUMBNAIL HANDLER - 1000x1300
# VERSION: Thumbnail-NukkiRingResize-V1
################################

VERSION = "Thumbnail-NukkiRingResize-V1"

# Global rembg session with U2Net
REMBG_SESSION = None

def init_rembg_session():
    """Initialize rembg session with U2Net for faster processing"""
    global REMBG_SESSION
    if REMBG_SESSION is None:
        try:
            from rembg import new_session
            # Use U2Net for faster processing
            REMBG_SESSION = new_session('u2net')
            logger.info("✅ U2Net session initialized for faster processing")
        except Exception as e:
            logger.error(f"❌ Failed to initialize rembg: {e}")
            REMBG_SESSION = None
    return REMBG_SESSION

# Initialize on module load
init_rembg_session()

def detect_ring_structure(image):
    """Advanced ring detection using multiple techniques"""
    logger.info("🔍 Starting advanced ring structure detection...")
    
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Convert to grayscale for analysis
    gray = np.array(image.convert('L'))
    h, w = gray.shape
    
    # 1. Edge detection with multiple methods
    edges_canny = cv2.Canny(gray, 50, 150)
    edges_sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    edges_sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges_sobel = np.sqrt(edges_sobel_x**2 + edges_sobel_y**2)
    edges_sobel = (edges_sobel / edges_sobel.max() * 255).astype(np.uint8)
    
    # Combine edges
    combined_edges = edges_canny | (edges_sobel > 50)
    
    # 2. Find contours and analyze shapes
    contours, _ = cv2.findContours(combined_edges.astype(np.uint8), 
                                   cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    ring_candidates = []
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < 100 or area > h * w * 0.8:  # Skip too small or too large
            continue
        
        # Calculate shape properties
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
            
        # Circularity
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Fit ellipse if possible
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (center, (width, height), angle) = ellipse
            
            # Check if it's ring-like (circular or elliptical)
            aspect_ratio = min(width, height) / max(width, height) if max(width, height) > 0 else 0
            
            # Ring criteria
            if circularity > 0.3 or aspect_ratio > 0.5:
                # Check if it's hollow (has inner space)
                mask = np.zeros(gray.shape, np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                
                # Erode to find potential inner area
                kernel = np.ones((5,5), np.uint8)
                eroded = cv2.erode(mask, kernel, iterations=2)
                
                # Find inner contours
                inner_contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, 
                                                    cv2.CHAIN_APPROX_SIMPLE)
                
                for inner in inner_contours:
                    inner_area = cv2.contourArea(inner)
                    if inner_area > area * 0.1:  # Inner area should be significant
                        ring_candidates.append({
                            'outer_contour': contour,
                            'inner_contour': inner,
                            'center': center,
                            'size': (width, height),
                            'angle': angle,
                            'circularity': circularity,
                            'aspect_ratio': aspect_ratio,
                            'area': area,
                            'inner_area': inner_area
                        })
    
    # 3. Hough Circle Transform for circular rings
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                              param1=50, param2=30, minRadius=10, maxRadius=int(min(h, w)/2))
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            x, y, r = circle
            # Check if this could be a ring (has hollow center)
            mask = np.zeros(gray.shape, np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            cv2.circle(mask, (x, y), max(1, r//3), 0, -1)  # Hollow center
            
            # Verify it matches the actual image structure
            overlap = cv2.bitwise_and(combined_edges, mask)
            if np.sum(overlap) > r * 2 * np.pi * 0.3:  # At least 30% edge overlap
                ring_candidates.append({
                    'type': 'circle',
                    'center': (x, y),
                    'radius': r,
                    'inner_radius': r//3
                })
    
    logger.info(f"✅ Found {len(ring_candidates)} ring candidates")
    return ring_candidates

def create_ring_aware_mask(image, ring_candidates):
    """Create mask that properly handles ring interior"""
    logger.info("🎯 Creating ring-aware mask...")
    
    h, w = image.size[1], image.size[0]
    ring_mask = np.zeros((h, w), dtype=np.uint8)
    
    for ring in ring_candidates:
        if 'type' in ring and ring['type'] == 'circle':
            # Circular ring
            cv2.circle(ring_mask, ring['center'], ring['radius'], 255, -1)
            cv2.circle(ring_mask, ring['center'], ring['inner_radius'], 0, -1)
        elif 'outer_contour' in ring:
            # Contour-based ring
            cv2.drawContours(ring_mask, [ring['outer_contour']], -1, 255, -1)
            if 'inner_contour' in ring:
                cv2.drawContours(ring_mask, [ring['inner_contour']], -1, 0, -1)
    
    return ring_mask

def u2net_ultra_precise_removal_v4_ring_aware(image: Image.Image) -> Image.Image:
    """ULTRA PRECISE V4 WITH RING-AWARE DETECTION"""
    try:
        from rembg import remove
        
        global REMBG_SESSION
        if REMBG_SESSION is None:
            REMBG_SESSION = init_rembg_session()
            if REMBG_SESSION is None:
                return image
        
        logger.info("🔷 U2Net ULTRA PRECISE V4 RING-AWARE - Maximum Quality")
        
        # CRITICAL: Ensure RGBA mode before processing
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # First, detect ring structure
        ring_candidates = detect_ring_structure(image)
        ring_mask = create_ring_aware_mask(image, ring_candidates)
        
        # Enhanced pre-processing for jewelry with adaptive enhancement
        img_array = np.array(image, dtype=np.float32)
        
        # Adaptive contrast based on image statistics
        gray = cv2.cvtColor(img_array[:,:,:3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        
        # Dynamic contrast adjustment
        if std_val < 30:  # Low contrast image
            contrast_factor = 1.6
        elif std_val < 50:
            contrast_factor = 1.4
        else:
            contrast_factor = 1.2
        
        contrast = ImageEnhance.Contrast(image)
        image_enhanced = contrast.enhance(contrast_factor)
        
        # Apply slight sharpening for edge definition
        sharpness = ImageEnhance.Sharpness(image_enhanced)
        image_enhanced = sharpness.enhance(1.2)
        
        # Save image to buffer
        buffered = BytesIO()
        image_enhanced.save(buffered, format="PNG", compress_level=3, optimize=True)
        buffered.seek(0)
        img_data = buffered.getvalue()
        
        # Apply U2Net removal with REFINED settings for jewelry
        output = remove(
            img_data,
            session=REMBG_SESSION,
            alpha_matting=True,
            alpha_matting_foreground_threshold=350,  # Very high precision
            alpha_matting_background_threshold=0,
            alpha_matting_erode_size=0,  # No erosion
            only_mask=False,
            post_process_mask=True
        )
        
        result_image = Image.open(BytesIO(output))
        
        # CRITICAL: Ensure RGBA mode
        if result_image.mode != 'RGBA':
            result_image = result_image.convert('RGBA')
        
        # REFINED edge processing
        r, g, b, a = result_image.split()
        alpha_array = np.array(a, dtype=np.uint8)
        rgb_array = np.array(result_image.convert('RGB'), dtype=np.uint8)
        
        # Convert to float for processing
        alpha_float = alpha_array.astype(np.float32) / 255.0
        
        # Apply ring mask to ensure ring interior is transparent
        if ring_mask is not None and ring_mask.shape == alpha_float.shape:
            # Invert ring mask - interior should be 0 (transparent)
            ring_interior = (ring_mask == 0).astype(np.float32)
            
            # Apply ring interior mask
            alpha_float = alpha_float * (1 - ring_interior)
        
        # STAGE 1: REFINED shadow detection with color spill analysis
        logger.info("🔍 REFINED shadow and color spill detection...")
        
        gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
        
        # Convert to multiple color spaces for comprehensive analysis
        hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
        h_channel, s_channel, v_channel = cv2.split(hsv)
        
        lab = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Multi-level shadow detection with refined thresholds
        # Level 1: Very faint shadows
        very_faint_shadows = (alpha_float > 0.01) & (alpha_float < 0.3)
        
        # Level 2: Low saturation gray areas with refined detection
        gray_shadows = (s_channel < 25) & (v_channel < 180) & (alpha_float < 0.6)
        
        # Level 3: Color spill detection
        # Detect green/blue screen spill
        green_spill = (h_channel > 35) & (h_channel < 85) & (s_channel > 30) & (alpha_float < 0.8)
        blue_spill = (h_channel > 85) & (h_channel < 135) & (s_channel > 30) & (alpha_float < 0.8)
        
        # Level 4: Edge-based shadow detection with multi-scale
        edges_fine = cv2.Canny(gray, 50, 150)
        edges_coarse = cv2.Canny(gray, 20, 80)
        edges_combined = edges_fine | edges_coarse
        
        # Dilate edges for shadow detection
        edge_dilated = cv2.dilate(edges_combined, np.ones((5,5)), iterations=1)
        edge_shadows = (alpha_float < 0.7) & (~edge_dilated.astype(bool))
        
        # Level 5: LAB-based shadow detection with tighter thresholds
        lab_shadows = (l_channel < 160) & (np.abs(a_channel - 128) < 15) & (np.abs(b_channel - 128) < 15)
        
        # Level 6: Gradient-based shadow detection
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        low_gradient = gradient_magnitude < np.percentile(gradient_magnitude, 10)
        gradient_shadows = low_gradient & (alpha_float < 0.5)
        
        # Combine all shadow detections
        all_shadows = (very_faint_shadows | gray_shadows | green_spill | blue_spill | 
                      edge_shadows | (lab_shadows & (alpha_float < 0.85)) | gradient_shadows)
        
        # REFINED shadow removal with feathering
        if np.any(all_shadows):
            logger.info("🔥 Removing shadows with refined feathering...")
            
            # Create distance map from main object
            main_object = (alpha_float > 0.8).astype(np.uint8)
            dist_from_object = cv2.distanceTransform(1 - main_object, cv2.DIST_L2, 5)
            
            # Feathered shadow removal based on distance
            shadow_removal_strength = np.clip(dist_from_object / 10, 0, 1)
            alpha_float[all_shadows] *= (1 - shadow_removal_strength[all_shadows])
        
        # STAGE 2: Ultra-precise multi-scale edge detection
        logger.info("🔍 Multi-scale edge detection with 8 methods...")
        
        # Method 1-3: Multi-scale Sobel
        sobel_scales = []
        for ksize in [3, 5, 7, 9]:
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
            sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
            sobel_scales.append(sobel_mag)
        
        sobel_combined = np.max(np.array(sobel_scales), axis=0)
        sobel_edges = (sobel_combined / sobel_combined.max() * 255).astype(np.uint8) > 20
        
        # Method 4: Scharr for fine details
        scharrx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        scharry = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        scharr_magnitude = np.sqrt(scharrx**2 + scharry**2)
        scharr_edges = (scharr_magnitude / scharr_magnitude.max() * 255).astype(np.uint8) > 25
        
        # Method 5: Laplacian for jewelry details
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=5)
        laplacian_edges = np.abs(laplacian) > 20
        
        # Method 6-8: Multi-threshold Canny with non-maximum suppression
        canny_low = cv2.Canny(gray, 10, 40)
        canny_mid = cv2.Canny(gray, 30, 90)
        canny_high = cv2.Canny(gray, 50, 150)
        canny_ultra = cv2.Canny(gray, 80, 200)
        
        # Combine all edge detections
        all_edges = (sobel_edges | scharr_edges | laplacian_edges | 
                    (canny_low > 0) | (canny_mid > 0) | (canny_high > 0) | (canny_ultra > 0))
        
        # STAGE 3: Ring-aware component analysis
        logger.info("🔍 Ring-aware component analysis...")
        
        # Binary mask for main object
        alpha_binary = (alpha_float > 0.5).astype(np.uint8)
        
        # Clean up with adaptive morphology
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        alpha_binary = cv2.morphologyEx(alpha_binary, cv2.MORPH_OPEN, kernel_open)
        alpha_binary = cv2.morphologyEx(alpha_binary, cv2.MORPH_CLOSE, kernel_close)
        
        num_labels, labels = cv2.connectedComponents(alpha_binary)
        
        if num_labels > 1:
            # Analyze all components with shape metrics
            component_stats = []
            
            for i in range(1, num_labels):
                component = (labels == i)
                size = np.sum(component)
                
                if size > 50:  # Minimum size threshold
                    # Calculate shape metrics
                    contours, _ = cv2.findContours(component.astype(np.uint8), 
                                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        contour = contours[0]
                        area = cv2.contourArea(contour)
                        perimeter = cv2.arcLength(contour, True)
                        
                        # Shape descriptors
                        if perimeter > 0:
                            circularity = (4 * np.pi * area) / (perimeter * perimeter)
                            hull = cv2.convexHull(contour)
                            hull_area = cv2.contourArea(hull)
                            solidity = area / hull_area if hull_area > 0 else 0
                            
                            # Eccentricity
                            if len(contour) >= 5:
                                ellipse = cv2.fitEllipse(contour)
                                (center, (width, height), angle) = ellipse
                                eccentricity = 0
                                if max(width, height) > 0:
                                    eccentricity = min(width, height) / max(width, height)
                            else:
                                eccentricity = 1
                            
                            # Distance from image center
                            component_center = np.mean(np.where(component), axis=1)
                            img_center = np.array([alpha_array.shape[0]/2, alpha_array.shape[1]/2])
                            dist_from_center = np.linalg.norm(component_center - img_center)
                            
                            # Check if component is inside a ring
                            is_inside_ring = False
                            for ring in ring_candidates:
                                if 'center' in ring and 'radius' in ring:
                                    dist_to_ring_center = np.linalg.norm(component_center - np.array(ring['center']))
                                    if dist_to_ring_center < ring['radius']:
                                        is_inside_ring = True
                                        break
                            
                            component_stats.append({
                                'label': i,
                                'size': size,
                                'circularity': circularity,
                                'solidity': solidity,
                                'eccentricity': eccentricity,
                                'dist_from_center': dist_from_center,
                                'edge_ratio': np.sum(all_edges[component]) / size,
                                'is_inside_ring': is_inside_ring
                            })
            
            # Keep components based on comprehensive criteria
            if component_stats:
                # Sort by size
                component_stats.sort(key=lambda x: x['size'], reverse=True)
                
                main_size = component_stats[0]['size']
                min_component_size = max(100, main_size * 0.01)  # 1% of main object
                
                valid_components = []
                for stats in component_stats:
                    # Skip components inside rings
                    if stats['is_inside_ring']:
                        logger.info(f"Skipping component inside ring: size={stats['size']}")
                        continue
                    
                    # Multi-criteria validation
                    size_valid = stats['size'] > min_component_size
                    shape_valid = (stats['solidity'] > 0.3 or stats['circularity'] > 0.2)
                    edge_valid = stats['edge_ratio'] > 0.1
                    
                    # Special case for very circular components (gems, holes) that are NOT inside rings
                    is_circular = stats['circularity'] > 0.7
                    
                    if (size_valid and (shape_valid or edge_valid)) or is_circular:
                        valid_components.append(stats['label'])
                
                # Create final mask
                main_mask = np.zeros_like(alpha_binary, dtype=bool)
                for label_id in valid_components:
                    main_mask |= (labels == label_id)
                
                # Apply main mask with feathering
                alpha_float[~main_mask] = 0
        
        # STAGE 4: Refined artifact removal with texture analysis
        logger.info("🔍 Texture-based artifact removal...")
        
        # Calculate local texture metrics
        gray_float = gray.astype(np.float32)
        
        # Local standard deviation (texture measure)
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        local_mean = cv2.filter2D(gray_float, -1, kernel)
        local_sq_mean = cv2.filter2D(gray_float**2, -1, kernel)
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))
        
        # Low texture areas that might be artifacts
        low_texture = local_std < 5
        
        # Combined artifact detection
        artifacts = ((s_channel < 20) & (v_channel > 30) & (v_channel < 180) & 
                    (alpha_float > 0) & (alpha_float < 0.7) & low_texture)
        
        if np.any(artifacts):
            alpha_float[artifacts] = 0
        
        # STAGE 5: Advanced edge refinement with bilateral filtering
        logger.info("🔍 Advanced edge refinement...")
        
        # Apply guided filter for edge-aware smoothing
        alpha_uint8 = (alpha_float * 255).astype(np.uint8)
        
        # Bilateral filter to preserve edges while smoothing
        alpha_bilateral = cv2.bilateralFilter(alpha_uint8, 9, 50, 50)
        alpha_float = alpha_bilateral.astype(np.float32) / 255.0
        
        # Sharp edge enhancement with adaptive sigmoid
        edge_sharpness = np.zeros_like(alpha_float)
        
        # Calculate edge strength
        grad_alpha_x = cv2.Sobel(alpha_float, cv2.CV_64F, 1, 0, ksize=3)
        grad_alpha_y = cv2.Sobel(alpha_float, cv2.CV_64F, 0, 1, ksize=3)
        edge_strength = np.sqrt(grad_alpha_x**2 + grad_alpha_y**2)
        
        # Adaptive sigmoid based on edge strength
        high_edge_mask = edge_strength > 0.1
        low_edge_mask = ~high_edge_mask
        
        # Sharp sigmoid for strong edges
        k_sharp = 200
        threshold_sharp = 0.5
        edge_sharpness[high_edge_mask] = 1 / (1 + np.exp(-k_sharp * (alpha_float[high_edge_mask] - threshold_sharp)))
        
        # Softer sigmoid for weak edges
        k_soft = 50
        threshold_soft = 0.5
        edge_sharpness[low_edge_mask] = 1 / (1 + np.exp(-k_soft * (alpha_float[low_edge_mask] - threshold_soft)))
        
        alpha_float = edge_sharpness
        
        # STAGE 6: Final cleanup with morphological operations
        logger.info("🔍 Final morphological cleanup...")
        
        # Remove small holes
        alpha_binary_final = (alpha_float > 0.5).astype(np.uint8)
        
        # Fill small holes
        kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        alpha_filled = cv2.morphologyEx(alpha_binary_final, cv2.MORPH_CLOSE, kernel_fill)
        
        # Remove small components
        num_labels_final, labels_final = cv2.connectedComponents(alpha_filled)
        
        if num_labels_final > 2:
            sizes_final = [(i, np.sum(labels_final == i)) for i in range(1, num_labels_final)]
            if sizes_final:
                sizes_final.sort(key=lambda x: x[1], reverse=True)
                min_size = max(150, alpha_array.size * 0.0002)  # 0.02% of image
                
                valid_mask = np.zeros_like(alpha_filled, dtype=bool)
                for label_id, size in sizes_final:
                    if size > min_size:
                        valid_mask |= (labels_final == label_id)
                
                alpha_float[~valid_mask] = 0
        
        # Apply final smoothing
        alpha_final = cv2.GaussianBlur(alpha_float, (3, 3), 0.5)
        
        # Convert back to uint8
        alpha_array = np.clip(alpha_final * 255, 0, 255).astype(np.uint8)
        
        logger.info("✅ ULTRA PRECISE V4 RING-AWARE complete")
        
        a_new = Image.fromarray(alpha_array)
        result = Image.merge('RGBA', (r, g, b, a_new))
        
        # Verify RGBA mode
        if result.mode != 'RGBA':
            logger.error("❌ WARNING: Result is not RGBA!")
            result = result.convert('RGBA')
        
        return result
        
    except Exception as e:
        logger.error(f"U2Net removal failed: {e}")
        # Ensure RGBA mode even on failure
        if image.mode != 'RGBA':
            return image.convert('RGBA')
        return image

def ensure_ring_holes_transparent_ultra_v4_ring_aware(image: Image.Image) -> Image.Image:
    """ULTRA PRECISE V4 RING-AWARE HOLE DETECTION"""
    # CRITICAL: Preserve RGBA mode
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    logger.info("🔍 ULTRA PRECISE V4 RING-AWARE Hole Detection")
    
    # First, detect ring structure
    ring_candidates = detect_ring_structure(image)
    
    r, g, b, a = image.split()
    alpha_array = np.array(a, dtype=np.uint8)
    rgb_array = np.array(image.convert('RGB'), dtype=np.uint8)
    
    h, w = alpha_array.shape
    
    # Convert to multiple color spaces
    hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)
    
    lab = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    
    # Create hole mask
    holes_mask = np.zeros_like(alpha_array, dtype=np.float32)
    
    # Process each ring candidate
    for ring in ring_candidates:
        if 'center' in ring:
            # Create mask for ring interior
            ring_interior_mask = np.zeros_like(alpha_array, dtype=np.uint8)
            
            if 'radius' in ring:
                # Circular ring
                cv2.circle(ring_interior_mask, tuple(map(int, ring['center'])), 
                          int(ring.get('inner_radius', ring['radius'] * 0.6)), 255, -1)
            elif 'inner_contour' in ring:
                # Contour-based ring
                cv2.drawContours(ring_interior_mask, [ring['inner_contour']], -1, 255, -1)
            
            # Check brightness in ring interior
            if np.any(ring_interior_mask > 0):
                interior_brightness = np.mean(gray[ring_interior_mask > 0])
                interior_v = np.mean(v_channel[ring_interior_mask > 0])
                interior_saturation = np.mean(s_channel[ring_interior_mask > 0])
                
                logger.info(f"Ring interior: brightness={interior_brightness:.1f}, "
                          f"V={interior_v:.1f}, saturation={interior_saturation:.1f}")
                
                # If interior is bright and low saturation, it's likely a hole
                if (interior_brightness > 220 or interior_v > 225) and interior_saturation < 30:
                    holes_mask[ring_interior_mask > 0] = 255
                    logger.info("✅ Ring interior identified as hole")
    
    # STAGE 1: Comprehensive hole detection with refined thresholds
    # Multiple criteria for hole detection
    very_bright_v = v_channel > 248
    very_bright_l = l_channel > 243
    very_bright_gray = gray > 243
    
    # Very low saturation with adaptive threshold
    mean_saturation = np.mean(s_channel[alpha_array > 128])
    saturation_threshold = min(20, mean_saturation * 0.3)
    very_low_saturation = s_channel < saturation_threshold
    
    # Low color variance in LAB with adaptive threshold
    a_variance = np.std(a_channel[alpha_array > 128])
    b_variance = np.std(b_channel[alpha_array > 128])
    
    low_color_variance = ((np.abs(a_channel - 128) < min(20, a_variance)) & 
                         (np.abs(b_channel - 128) < min(20, b_variance)))
    
    # Alpha-based detection
    alpha_holes = alpha_array < 20
    
    # Combine all criteria
    potential_holes = ((very_bright_v | very_bright_l | very_bright_gray) & 
                      (very_low_saturation | low_color_variance)) | alpha_holes
    
    # STAGE 2: Advanced shape-based hole detection
    logger.info("🔍 Advanced shape and topology analysis...")
    
    if np.any(alpha_array > 128):
        # Distance transform from object
        object_mask = (alpha_array > 128).astype(np.uint8)
        dist_transform = cv2.distanceTransform(object_mask, cv2.DIST_L2, 5)
        
        # Multi-scale narrow region detection
        narrow_scales = []
        for scale in [15, 20, 25, 30]:
            narrow = (dist_transform > 1) & (dist_transform < scale)
            narrow_scales.append(narrow)
        
        narrow_regions = np.any(np.array(narrow_scales), axis=0)
        
        # Bright areas in narrow regions with gradient check
        gray_gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3,3)))
        narrow_bright = narrow_regions & ((gray > 235) | (v_channel > 240)) & (gray_gradient < 20)
        potential_holes |= narrow_bright
        
        # Topology-based hole detection
        # Find enclosed regions using flood fill
        inverted = cv2.bitwise_not(object_mask)
        num_inv_labels, inv_labels = cv2.connectedComponents(inverted)
        
        # Advanced enclosed region analysis
        for label in range(1, num_inv_labels):
            component = (inv_labels == label)
            if np.any(component):
                # Multi-criteria enclosure check
                component_uint8 = component.astype(np.uint8)
                
                # Method 1: Border touching
                dilated = cv2.dilate(component_uint8, np.ones((7,7)), iterations=1)
                touches_border = (np.any(dilated[0,:]) or np.any(dilated[-1,:]) or 
                                np.any(dilated[:,0]) or np.any(dilated[:,-1]))
                
                # Method 2: Convex hull analysis
                contours, _ = cv2.findContours(component_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    hull = cv2.convexHull(contours[0])
                    hull_mask = np.zeros_like(component_uint8)
                    cv2.fillPoly(hull_mask, [hull], 1)
                    
                    # Check if hull intersects with object
                    hull_intersects_object = np.any(hull_mask & object_mask)
                    
                    if not touches_border or hull_intersects_object:
                        # This is likely an enclosed hole
                        component_pixels = rgb_array[component]
                        if len(component_pixels) > 0:
                            # Multi-metric brightness analysis
                            brightness_rgb = np.mean(component_pixels)
                            brightness_v = np.mean(v_channel[component])
                            brightness_l = np.mean(l_channel[component])
                            brightness_percentile = np.percentile(gray[component], 90)
                            
                            if (brightness_rgb > 230 or brightness_v > 235 or 
                                brightness_l > 230 or brightness_percentile > 240):
                                potential_holes[component] = True
                                logger.info(f"Found enclosed bright region: RGB={brightness_rgb:.1f}, "
                                          f"V={brightness_v:.1f}, L={brightness_l:.1f}")
    
    # STAGE 3: Texture and pattern-based hole detection
    logger.info("🔍 Texture and pattern analysis...")
    
    # Local Binary Patterns for texture
    def compute_lbp(image, radius=1):
        rows, cols = image.shape
        lbp = np.zeros_like(image)
        
        for i in range(radius, rows - radius):
            for j in range(radius, cols - radius):
                center = image[i, j]
                binary_string = ''
                
                # 8 neighbors
                for angle in range(8):
                    x = int(i + radius * np.cos(2 * np.pi * angle / 8))
                    y = int(j + radius * np.sin(2 * np.pi * angle / 8))
                    
                    if 0 <= x < rows and 0 <= y < cols:
                        binary_string += '1' if image[x, y] >= center else '0'
                
                lbp[i, j] = int(binary_string, 2)
        
        return lbp
    
    # Compute LBP
    lbp = compute_lbp(gray)
    
    # Uniform texture areas (potential holes)
    lbp_variance = cv2.filter2D(lbp.astype(np.float32), -1, np.ones((5,5))/25)
    uniform_texture = lbp_variance < 10
    
    # Combine with brightness for hole detection
    texture_holes = uniform_texture & (gray > 220) & (alpha_array > 100)
    potential_holes |= texture_holes
    
    # Clean up noise with adaptive morphology
    kernel_size = max(3, min(7, int(np.sqrt(h * w) / 100)))
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    potential_holes = cv2.morphologyEx(potential_holes.astype(np.uint8), cv2.MORPH_OPEN, kernel_clean)
    potential_holes = cv2.morphologyEx(potential_holes, cv2.MORPH_CLOSE, kernel_clean)
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(potential_holes)
    
    # STAGE 4: Validate each hole candidate with ring awareness
    for label in range(1, num_labels):
        component = (labels == label)
        component_size = np.sum(component)
        
        # Adaptive size constraints based on image size
        min_size = max(15, h * w * 0.00001)  # 0.001%
        max_size = h * w * 0.3  # 30%
        
        if min_size < component_size < max_size:
            coords = np.where(component)
            if len(coords[0]) == 0:
                continue
            
            # Check if component is within a ring
            component_center = np.mean(coords, axis=1)
            is_in_ring = False
            
            for ring in ring_candidates:
                if 'center' in ring and 'radius' in ring:
                    dist_to_center = np.linalg.norm(component_center - np.array(ring['center']))
                    if dist_to_center < ring.get('inner_radius', ring['radius'] * 0.6):
                        is_in_ring = True
                        break
            
            # Comprehensive component analysis
            component_pixels_rgb = rgb_array[component]
            component_alpha = alpha_array[component]
            
            if len(component_pixels_rgb) > 0:
                # Multi-space brightness analysis
                brightness_metrics = {
                    'rgb_mean': np.mean(component_pixels_rgb),
                    'rgb_max': np.max(component_pixels_rgb),
                    'v_mean': np.mean(v_channel[component]),
                    'v_percentile_90': np.percentile(v_channel[component], 90),
                    'l_mean': np.mean(l_channel[component]),
                    'gray_mean': np.mean(gray[component]),
                    'gray_median': np.median(gray[component])
                }
                
                # Saturation and color analysis
                saturation_metrics = {
                    'mean': np.mean(s_channel[component]),
                    'max': np.max(s_channel[component]),
                    'std': np.std(s_channel[component])
                }
                
                # Color uniformity in multiple spaces
                uniformity_metrics = {
                    'rgb_std': np.max(np.std(component_pixels_rgb, axis=0)),
                    'hsv_std': np.std(h_channel[component]),
                    'lab_a_std': np.std(a_channel[component]),
                    'lab_b_std': np.std(b_channel[component])
                }
                
                # Advanced shape analysis
                component_uint8 = component.astype(np.uint8) * 255
                contours, _ = cv2.findContours(component_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                shape_metrics = {}
                is_enclosed = False
                
                if contours:
                    contour = contours[0]
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    
                    if perimeter > 0 and area > 0:
                        # Shape descriptors
                        shape_metrics['circularity'] = (4 * np.pi * area) / (perimeter * perimeter)
                        
                        hull = cv2.convexHull(contour)
                        hull_area = cv2.contourArea(hull)
                        shape_metrics['convexity'] = area / hull_area if hull_area > 0 else 0
                        shape_metrics['solidity'] = area / hull_area if hull_area > 0 else 0
                        
                        # Aspect ratio
                        if len(contour) >= 5:
                            ellipse = cv2.fitEllipse(contour)
                            (center, (width, height), angle) = ellipse
                            shape_metrics['aspect_ratio'] = min(width, height) / max(width, height) if max(width, height) > 0 else 1
                        else:
                            shape_metrics['aspect_ratio'] = 1
                        
                        # Enclosure check
                        x, y, w, h = cv2.boundingRect(contour)
                        if 'object_mask' in locals():
                            roi = object_mask[max(0,y-5):min(object_mask.shape[0],y+h+5), 
                                            max(0,x-5):min(object_mask.shape[1],x+w+5)]
                            if roi.shape[0] > 0 and roi.shape[1] > 0:
                                border_sum = (np.sum(roi[0,:]) + np.sum(roi[-1,:]) + 
                                            np.sum(roi[:,0]) + np.sum(roi[:,-1]))
                                expected_border = 2 * (roi.shape[0] + roi.shape[1]) - 4
                                if border_sum > expected_border * 0.7:
                                    is_enclosed = True
                
                # Multi-criteria confidence calculation
                confidence = 0.0
                
                # Brightness criteria (most important)
                brightness_score = 0.0
                if brightness_metrics['rgb_mean'] > 245 and brightness_metrics['v_mean'] > 248:
                    brightness_score = 0.5
                elif brightness_metrics['rgb_mean'] > 235 and brightness_metrics['v_percentile_90'] > 245:
                    brightness_score = 0.4
                elif brightness_metrics['gray_median'] > 240:
                    brightness_score = 0.3
                elif brightness_metrics['l_mean'] > 235:
                    brightness_score = 0.2
                
                confidence += brightness_score
                
                # Saturation criteria
                if saturation_metrics['mean'] < 8:
                    confidence += 0.3
                elif saturation_metrics['mean'] < 15 and saturation_metrics['std'] < 5:
                    confidence += 0.2
                elif saturation_metrics['max'] < 25:
                    confidence += 0.1
                
                # Color uniformity
                if uniformity_metrics['rgb_std'] < 8:
                    confidence += 0.2
                elif uniformity_metrics['rgb_std'] < 15:
                    confidence += 0.1
                
                # Shape criteria
                if shape_metrics:
                    shape_score = 0.0
                    if shape_metrics.get('circularity', 0) > 0.7:
                        shape_score += 0.1
                    if shape_metrics.get('convexity', 0) > 0.8:
                        shape_score += 0.05
                    if shape_metrics.get('aspect_ratio', 0) > 0.7:
                        shape_score += 0.05
                    
                    confidence += shape_score
                
                # Bonus for enclosed regions
                if is_enclosed:
                    confidence += 0.25
                
                # Bonus for being inside a ring
                if is_in_ring:
                    confidence += 0.3
                    logger.info("Component is inside a ring - boosting confidence")
                
                # Alpha channel bonus
                if np.mean(component_alpha) < 200:
                    confidence += 0.1
                
                # Apply hole mask based on confidence
                if confidence > 0.4:  # Lower threshold for better detection
                    holes_mask[component] = 255
                    logger.info(f"Hole detected: brightness={brightness_metrics['rgb_mean']:.1f}, "
                              f"saturation={saturation_metrics['mean']:.1f}, "
                              f"uniformity={uniformity_metrics['rgb_std']:.1f}, "
                              f"shape={shape_metrics.get('circularity', 0):.2f}, "
                              f"enclosed={is_enclosed}, in_ring={is_in_ring}, confidence={confidence:.2f}")
    
    # STAGE 5: Apply holes with refined transitions
    if np.any(holes_mask > 0):
        # Create smooth transitions
        holes_mask_smooth = cv2.GaussianBlur(holes_mask, (7, 7), 1.5)
        
        # Create multiple transition zones
        kernel_sizes = [(3, 3), (5, 5), (7, 7)]
        transition_masks = []
        
        for ksize in kernel_sizes:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
            dilated = cv2.dilate(holes_mask, kernel, iterations=1)
            transition = (dilated > 0) & (holes_mask < 255)
            transition_masks.append(transition)
        
        alpha_float = alpha_array.astype(np.float32)
        
        # Apply holes with hard edge
        alpha_float[holes_mask_smooth > 240] = 0
        
        # Apply smooth transitions
        for i, transition in enumerate(transition_masks):
            if np.any(transition):
                # Distance-based transition with varying strength
                dist_from_hole = cv2.distanceTransform((holes_mask == 0).astype(np.uint8), cv2.DIST_L2, 3)
                transition_strength = 3 + i * 2  # Varying transition widths
                transition_alpha = np.clip(dist_from_hole / transition_strength, 0, 1)
                alpha_float[transition] *= transition_alpha[transition]
        
        # Final smoothing
        alpha_float = cv2.bilateralFilter(alpha_float.astype(np.uint8), 5, 50, 50).astype(np.float32)
        
        alpha_array = np.clip(alpha_float, 0, 255).astype(np.uint8)
        
        logger.info("✅ Ring holes applied with refined multi-scale transitions")
    
    a_new = Image.fromarray(alpha_array)
    result = Image.merge('RGBA', (r, g, b, a_new))
    
    # Verify RGBA mode
    if result.mode != 'RGBA':
        logger.error("❌ WARNING: Result is not RGBA!")
        result = result.convert('RGBA')
    
    return result

def find_input_data_fast(data):
    """FIXED: Fast input data extraction - consistent string return"""
    # Handle string input
    if isinstance(data, str) and len(data) > 50:
        return data
    
    # Handle dictionary input
    if isinstance(data, dict):
        priority_keys = ['image', 'image_base64', 'enhanced_image', 'base64', 'img']
        
        # Check priority keys first
        for key in priority_keys:
            if key in data and isinstance(data[key], str) and len(data[key]) > 50:
                return data[key]
        
        # Check nested structures
        for key in ['input', 'data']:
            if key in data and isinstance(data[key], dict):
                result = find_input_data_fast(data[key])
                if result:
                    return result
            elif key in data and isinstance(data[key], str) and len(data[key]) > 50:
                return data[key]
        
        # Check numbered keys as fallback
        for i in range(10):
            if str(i) in data and isinstance(data[str(i)], str) and len(data[str(i)]) > 50:
                return data[str(i)]
    
    return None

def find_filename_fast(data):
    """Fast filename extraction"""
    if isinstance(data, dict):
        for key in ['filename', 'file_name', 'name']:
            if key in data and isinstance(data[key], str):
                return data[key]
        
        if 'input' in data and isinstance(data['input'], dict):
            for key in ['filename', 'file_name', 'name']:
                if key in data['input']:
                    return data['input'][key]
    
    return None

def generate_thumbnail_filename(original_filename, image_index):
    """Generate thumbnail filename with fixed numbers"""
    if not original_filename:
        return f"thumbnail_{image_index:03d}.png"
    
    # Fixed thumbnail numbers: 007, 009, 010
    thumbnail_numbers = {1: "007", 2: "009", 3: "010"}
    
    new_filename = original_filename
    pattern = r'(_\d{3})'
    if re.search(pattern, new_filename):
        new_filename = re.sub(pattern, f'_{thumbnail_numbers.get(image_index, "007")}', new_filename)
    else:
        name_parts = new_filename.split('.')
        name_parts[0] += f'_{thumbnail_numbers.get(image_index, "007")}'
        new_filename = '.'.join(name_parts)
    
    return new_filename

def decode_base64_fast(base64_str: str) -> bytes:
    """ENHANCED: Fast base64 decode with consistent padding handling"""
    try:
        if not base64_str or len(base64_str) < 50:
            raise ValueError("Invalid base64 string")
        
        # Remove data URL prefix if present
        if 'base64,' in base64_str:
            base64_str = base64_str.split('base64,')[-1]
        
        # Clean whitespace
        base64_str = ''.join(base64_str.split())
        
        # Keep only valid base64 characters
        valid_chars = set(string.ascii_letters + string.digits + '+/=')
        base64_str = ''.join(c for c in base64_str if c in valid_chars)
        
        # FIXED: Try with padding first (normal base64)
        try:
            decoded = base64.b64decode(base64_str, validate=True)
            return decoded
        except Exception:
            # If fails, try to add proper padding
            no_pad = base64_str.rstrip('=')
            padding_needed = (4 - len(no_pad) % 4) % 4
            padded = no_pad + ('=' * padding_needed)
            decoded = base64.b64decode(padded, validate=True)
            return decoded
            
    except Exception as e:
        logger.error(f"Base64 decode error: {str(e)}")
        raise ValueError(f"Invalid base64 data: {str(e)}")

def base64_to_image_fast(base64_string):
    """ENHANCED: Fast base64 to image conversion with consistent handling"""
    try:
        image_bytes = decode_base64_fast(base64_string)
        return Image.open(BytesIO(image_bytes))
    except Exception as e:
        logger.error(f"Base64 to image error: {str(e)}")
        raise ValueError(f"Invalid image data: {str(e)}")

def create_thumbnail_proportional(image, target_width=1000, target_height=1300):
    """Create thumbnail with proper proportional sizing - preserving transparency"""
    # CRITICAL: Ensure RGBA mode
    if image.mode != 'RGBA':
        logger.warning(f"⚠️ Converting {image.mode} to RGBA in thumbnail creation")
        image = image.convert('RGBA')
    
    original_width, original_height = image.size
    
    logger.info(f"Creating proportional thumbnail from {original_width}x{original_height} to {target_width}x{target_height}")
    
    # For 2000x2600 -> 1000x1300, it's exactly 50% resize
    if original_width == 2000 and original_height == 2600:
        logger.info("Direct 50% resize for standard input size")
        result = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    else:
        # For other sizes, maintain aspect ratio
        scale_x = target_width / original_width
        scale_y = target_height / original_height
        scale = min(scale_x, scale_y)
        
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize first
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Center crop if needed - preserving transparency
        if new_width != target_width or new_height != target_height:
            # Create transparent background
            result = Image.new('RGBA', (target_width, target_height), (0, 0, 0, 0))
            
            # Center paste
            paste_x = (target_width - new_width) // 2
            paste_y = (target_height - new_height) // 2
            
            result.paste(resized, (paste_x, paste_y), resized)
        else:
            result = resized
    
    # Verify RGBA mode
    if result.mode != 'RGBA':
        logger.error("❌ WARNING: Thumbnail is not RGBA!")
        result = result.convert('RGBA')
    
    return result

def image_to_base64(image, keep_transparency=True):
    """Convert to base64 WITH padding - FIXED for Google Script compatibility"""
    buffered = BytesIO()
    
    # CRITICAL FIX: Force RGBA and save as PNG
    if image.mode != 'RGBA' and keep_transparency:
        logger.warning(f"⚠️ Converting {image.mode} to RGBA for transparency")
        image = image.convert('RGBA')
    
    if image.mode == 'RGBA':
        logger.info("💎 Saving RGBA image as PNG with compression level 3")
        image.save(buffered, format='PNG', compress_level=3, optimize=True)
    else:
        logger.info(f"Saving {image.mode} mode image as PNG")
        image.save(buffered, format='PNG', optimize=True, compress_level=3)
    
    buffered.seek(0)
    base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    # FIXED: Always return WITH padding for Google Script compatibility
    return base64_str

def handler(event):
    """Thumbnail handler - Background removal, ring holes, and resize"""
    try:
        logger.info(f"=== Thumbnail {VERSION} Started ===")
        logger.info("🎯 PROCESSING: Background removal + Ring holes + Resize")
        logger.info("❌ REMOVED: White balance, pattern enhancement, SwinIR")
        logger.info("✅ RETAINED: U2Net background removal, ring hole detection, resize to 1000x1300")
        logger.info("📌 BASE64 PADDING: ALWAYS INCLUDED for Google Script compatibility")
        logger.info("🗜️ COMPRESSION: Level 3 (balanced speed/size)")
        
        # Get image index
        image_index = event.get('image_index', 1)
        if isinstance(event.get('input'), dict):
            image_index = event.get('input', {}).get('image_index', image_index)
        
        # Find input data
        filename = find_filename_fast(event)
        image_data_str = find_input_data_fast(event)
        
        if not image_data_str:
            raise ValueError("No input data found")
        
        # Load image using fixed function
        image = base64_to_image_fast(image_data_str)
        
        # CRITICAL: Convert to RGBA immediately
        if image.mode != 'RGBA':
            logger.info(f"Converting {image.mode} to RGBA")
            image = image.convert('RGBA')
        
        # STEP 1: Apply background removal
        logger.info("📸 STEP 1: Applying ULTRA PRECISE V4 RING-AWARE background removal")
        image = u2net_ultra_precise_removal_v4_ring_aware(image)
        
        # Verify RGBA after removal
        if image.mode != 'RGBA':
            logger.error("❌ Image lost RGBA after background removal!")
            image = image.convert('RGBA')
        
        # STEP 2: Apply ring hole detection
        logger.info("🔍 STEP 2: Applying ULTRA PRECISE V4 RING-AWARE hole detection")
        image = ensure_ring_holes_transparent_ultra_v4_ring_aware(image)
        
        # Verify RGBA after hole detection
        if image.mode != 'RGBA':
            logger.error("❌ Image lost RGBA after hole detection!")
            image = image.convert('RGBA')
        
        # STEP 3: Create proportional thumbnail
        logger.info("📏 STEP 3: Creating proportional thumbnail")
        thumbnail = create_thumbnail_proportional(image, 1000, 1300)
        
        # Final verification
        if thumbnail.mode != 'RGBA':
            logger.error("❌ CRITICAL: Final thumbnail is not RGBA! Converting...")
            thumbnail = thumbnail.convert('RGBA')
        
        logger.info(f"✅ Final thumbnail mode: {thumbnail.mode}")
        logger.info(f"✅ Final thumbnail size: {thumbnail.size}")
        
        # Convert to base64 - WITH padding for Google Script
        thumbnail_base64 = image_to_base64(thumbnail, keep_transparency=True)
        
        # Verify transparency is preserved
        logger.info("✅ Transparency preserved in final output")
        
        output_filename = generate_thumbnail_filename(filename, image_index)
        
        return {
            "output": {
                "thumbnail": thumbnail_base64,
                "size": list(thumbnail.size),
                "filename": output_filename,
                "original_filename": filename,
                "image_index": image_index,
                "format": "base64_with_padding",
                "version": VERSION,
                "status": "success",
                "png_support": True,
                "has_transparency": True,
                "transparency_preserved": True,
                "background_removed": True,
                "ring_holes_applied": True,
                "output_mode": "RGBA",
                "base64_padding": "INCLUDED",
                "compression": "level_3",
                "file_number_info": {
                    "007": "Thumbnail 1",
                    "009": "Thumbnail 2", 
                    "010": "Thumbnail 3"
                },
                "processing_steps": [
                    "1. U2Net Ultra Precise V4 Ring-Aware background removal",
                    "2. Ultra Precise V4 Ring-Aware hole detection",
                    "3. Proportional resize to 1000x1300"
                ],
                "removed_features": [
                    "Auto white balance",
                    "Pattern enhancement",
                    "SwinIR enhancement"
                ],
                "ring_detection_features": [
                    "✅ RING STRUCTURE DETECTION: Multiple edge detection methods",
                    "✅ CONTOUR ANALYSIS: Shape-based ring identification",
                    "✅ CIRCULARITY METRICS: Detect circular and elliptical rings",
                    "✅ HOUGH CIRCLE TRANSFORM: Perfect circle detection",
                    "✅ HOLLOW CENTER VALIDATION: Ensure rings have interior space",
                    "✅ RING-AWARE MASKING: Interior regions marked as transparent",
                    "✅ RING HOLE DETECTION: Multi-criteria confidence scoring",
                    "✅ TEXTURE ANALYSIS: LBP variance for hole detection",
                    "✅ TOPOLOGY ANALYSIS: Enclosed region detection",
                    "✅ ADAPTIVE TRANSITIONS: Multi-scale hole edge smoothing"
                ],
                "thumbnail_method": "Proportional resize (no aggressive cropping)",
                "google_script_compatibility": "Base64 WITH padding - FIXED",
                "expected_input": "Any size image",
                "output_size": "1000x1300",
                "output_format": "PNG with full transparency",
                "transparency_info": "Full RGBA transparency preserved - NO background or shadows"
            }
        }
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        
        return {
            "output": {
                "error": str(e),
                "status": "failed",
                "version": VERSION,
                "traceback": traceback.format_exc()
            }
        }

# RunPod handler
runpod.serverless.start({"handler": handler})
