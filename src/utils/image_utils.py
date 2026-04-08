"""Utility functions for image processing."""

from pathlib import Path
from PIL import Image
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import io

from src.utils.image_transforms import apply_transform

# Import SIFT-related modules at module level to avoid repeated import overhead
from skimage.feature import SIFT, match_descriptors
from skimage.measure import ransac
from skimage.transform import AffineTransform


def normalize_image(img: Image.Image) -> Image.Image:
    """
    Normalize a non-8-bit image to 8-bit using min-max normalization.
    
    Handles float32 TIFFs (mode 'F'), int32 TIFFs (mode 'I'), and other
    non-standard bit depths by scaling pixel values to the [0, 255] range.
    Standard 8-bit images (modes 'L', 'RGB', 'RGBA', etc.) pass through unchanged.
    
    Args:
        img: PIL Image in any mode
        
    Returns:
        PIL Image normalized to 8-bit ('L' for single-channel, 'RGB' for multi-channel)
    """
    if img.mode not in ('F', 'I', 'I;16'):
        return img
    
    arr = np.array(img, dtype=np.float64)
    min_val = arr.min()
    max_val = arr.max()
    
    if max_val - min_val > 0:
        normalized = ((arr - min_val) / (max_val - min_val) * 255.0)
    else:
        normalized = np.zeros_like(arr)
    
    return Image.fromarray(normalized.astype(np.uint8), 'L')


def create_thumbnail(image_path: Path, size: tuple[int, int] = (100, 100), colormap: str = "gray", transform: str = "none", cached_image: Image.Image = None, force_grayscale: bool = False) -> QPixmap | None:
    """
    Create a thumbnail QPixmap from an image file or cached image.
    
    Args:
        image_path: Path to image file
        size: Thumbnail size (width, height)
        colormap: Colormap to apply (default: "gray")
        transform: Image transformation to apply (default: "none")
        cached_image: Optional pre-loaded PIL Image to use instead of loading from disk
        force_grayscale: Whether to force conversion to grayscale (default: False)
        
    Returns:
        QPixmap thumbnail or None if error
    """
    try:
        # Use cached image if available, otherwise load from disk
        if cached_image is not None:
            img = cached_image.copy()  # Make a copy to avoid modifying cached version
        else:
            img = Image.open(image_path)
        
        # Normalize non-8-bit images (float32/int32 TIFFs) to 0-255
        img = normalize_image(img)
        
        # Determine if grayscale conversion is needed:
        # - force_grayscale setting is on
        # - A non-gray colormap is being applied (colormaps require grayscale)
        needs_grayscale = force_grayscale or colormap != "gray"
        
        # Track whether the image is natively color for color-preserving transforms
        is_color = img.mode in ('RGB', 'RGBA')
        
        if needs_grayscale and img.mode != 'L':
            img = img.convert('L')
            is_color = False
        elif img.mode not in ('L', 'RGB', 'RGBA'):
            # Convert exotic modes (CMYK, etc.) to RGB for display
            img = img.convert('RGB')
            is_color = True
        
        # Apply transformation if specified
        if transform != "none":
            if is_color and colormap == "gray":
                # Preserve color: apply transform to V channel of HSV
                has_alpha = img.mode == 'RGBA'
                alpha = None
                if has_alpha:
                    alpha = img.split()[3]
                    img = img.convert('RGB')
                
                hsv = img.convert('HSV')
                h, s, v = hsv.split()
                
                v_array = np.array(v)
                v_array = apply_transform(v_array, transform)
                v_transformed = Image.fromarray(v_array, 'L')
                
                img = Image.merge('HSV', (h, s, v_transformed)).convert('RGB')
                if has_alpha:
                    img.putalpha(alpha)
            else:
                # Grayscale path
                img_array = np.array(img)
                img_array = apply_transform(img_array, transform)
                img = Image.fromarray(img_array, 'L')
        
        # Apply colormap if not gray (requires grayscale)
        if colormap != "gray":
            img_array = np.array(img)
            img = apply_colormap(img_array, colormap)
        
        # Create thumbnail
        img.thumbnail(size, Image.Resampling.LANCZOS)
        
        # Convert PIL image to QPixmap
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        pixmap = QPixmap()
        pixmap.loadFromData(img_byte_arr.read())
        
        return pixmap
        
    except Exception as e:
        print(f"Error creating thumbnail for {image_path}: {e}")
        return None


def apply_colormap(gray_image: np.ndarray, colormap_name: str) -> Image.Image:
    """
    Apply a colormap to a grayscale image.
    
    Args:
        gray_image: Grayscale image as numpy array
        colormap_name: Name of colormap to apply
        
    Returns:
        PIL Image with colormap applied
    """
    try:
        # Try cmcrameri colormaps first
        import cmcrameri.cm as cmc
        import matplotlib.pyplot as plt
        
        # Get the colormap
        cmap_dict = {
            'batlow': cmc.batlow,
            'berlin': cmc.berlin,
            'broc': cmc.broc,
            'cork': cmc.cork,
            'hawaii': cmc.hawaii,
            'imola': cmc.imola,
            'lajolla': cmc.lajolla,
            'lapaz': cmc.lapaz,
            'nuuk': cmc.nuuk,
            'oslo': cmc.oslo,
            'roma': cmc.roma,
            'tokyo': cmc.tokyo,
            'turku': cmc.turku,
            'vik': cmc.vik,
        }
        
        if colormap_name in cmap_dict:
            cmap = cmap_dict[colormap_name]
        else:
            # Fall back to matplotlib colormaps
            cmap = plt.get_cmap(colormap_name)
        
        # Normalize image to 0-1 range using min-max normalization
        arr_float = gray_image.astype(np.float64)
        min_val = arr_float.min()
        max_val = arr_float.max()
        if max_val - min_val > 0:
            normalized = (arr_float - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(arr_float)
        
        # Apply colormap
        colored = cmap(normalized.astype(np.float32))
        
        # Convert to 0-255 RGB
        rgb = (colored[:, :, :3] * 255).astype(np.uint8)
        
        return Image.fromarray(rgb, 'RGB')
        
    except Exception as e:
        print(f"Error applying colormap {colormap_name}: {e}")
        # Return original as grayscale
        return Image.fromarray(gray_image, 'L')


def _load_image(img_path: Path, image_cache: dict = None) -> Image.Image:
    """
    Load an image from cache or disk.
    
    Args:
        img_path: Path to image file
        image_cache: Optional dict of cached images {path: PIL.Image}
        
    Returns:
        PIL Image in grayscale
    """
    # Try to get from cache first
    if image_cache is not None:
        cached_img = image_cache.get(str(img_path))
        if cached_img is not None:
            return cached_img.copy()
    
    # Load from disk
    img = Image.open(img_path)
    # Normalize non-8-bit images (float32/int32 TIFFs) to 0-255
    img = normalize_image(img)
    if img.mode != 'L':
        img = img.convert('L')
    return img


def compute_image_correlation(img1_path: Path, img2_path: Path, method: str = 'ncc', transform: str = 'none', image_cache: dict = None) -> float:
    """
    Compute similarity score between two images.
    
    Args:
        img1_path: Path to first image (base image)
        img2_path: Path to second image (comparison image)
        method: Correlation method ('ncc' for normalized cross-correlation, 'mse' for mean squared error)
        transform: Image transformation to apply before correlation (default: 'none')
        image_cache: Optional dict of cached images {path: PIL.Image}
        
    Returns:
        Similarity score (higher is more similar for ncc, lower is more similar for mse)
    """
    try:
        # Load both images (from cache if available)
        img1 = _load_image(img1_path, image_cache)
        img2 = _load_image(img2_path, image_cache)
        
        # Resize img2 to match img1 dimensions
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
        
        # Convert to numpy arrays
        arr1 = np.array(img1, dtype=np.float32)
        arr2 = np.array(img2, dtype=np.float32)
        
        # Apply transformation if specified
        if transform != 'none':
            arr1 = apply_transform(arr1.astype(np.uint8), transform).astype(np.float32)
            arr2 = apply_transform(arr2.astype(np.uint8), transform).astype(np.float32)
        
        if method == 'ncc':
            # Normalized Cross-Correlation
            # Normalize both arrays
            arr1_norm = (arr1 - arr1.mean()) / (arr1.std() + 1e-10)
            arr2_norm = (arr2 - arr2.mean()) / (arr2.std() + 1e-10)
            
            # Compute correlation
            correlation = np.mean(arr1_norm * arr2_norm)
            return float(correlation)
            
        elif method == 'mse':
            # Mean Squared Error (lower is better, so we return negative for consistency)
            mse = np.mean((arr1 - arr2) ** 2)
            return float(-mse)
        
        else:
            # Default to NCC
            arr1_norm = (arr1 - arr1.mean()) / (arr1.std() + 1e-10)
            arr2_norm = (arr2 - arr2.mean()) / (arr2.std() + 1e-10)
            correlation = np.mean(arr1_norm * arr2_norm)
            return float(correlation)
            
    except Exception as e:
        print(f"Error computing correlation between {img1_path} and {img2_path}: {e}")
        return 0.0


def compute_ssim(img1_path: Path, img2_path: Path) -> float:
    """
    Compute Structural Similarity Index (SSIM) between two images.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        
    Returns:
        SSIM score (between -1 and 1, higher is more similar)
    """
    try:
        from scipy.ndimage import gaussian_filter
        
        # Load both images
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        
        # Convert to grayscale
        if img1.mode != 'L':
            img1 = img1.convert('L')
        if img2.mode != 'L':
            img2 = img2.convert('L')
        
        # Resize img2 to match img1 dimensions
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
        
        # Convert to numpy arrays
        arr1 = np.array(img1, dtype=np.float32)
        arr2 = np.array(img2, dtype=np.float32)
        
        # SSIM calculation
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        # Compute means
        mu1 = gaussian_filter(arr1, sigma=1.5)
        mu2 = gaussian_filter(arr2, sigma=1.5)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Compute variances and covariance
        sigma1_sq = gaussian_filter(arr1 ** 2, sigma=1.5) - mu1_sq
        sigma2_sq = gaussian_filter(arr2 ** 2, sigma=1.5) - mu2_sq
        sigma12 = gaussian_filter(arr1 * arr2, sigma=1.5) - mu1_mu2
        
        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return float(np.mean(ssim_map))
        
    except Exception as e:
        print(f"Error computing SSIM between {img1_path} and {img2_path}: {e}")
        return 0.0


def compute_histogram_correlation(img1_path: Path, img2_path: Path, transform: str = 'none', image_cache: dict = None, base_histogram: np.ndarray = None) -> float:
    """
    Compute histogram correlation between two images.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        transform: Image transformation to apply before correlation
        image_cache: Optional dict of cached images {path: PIL.Image}
        base_histogram: Optional pre-computed normalized histogram for base image
        
    Returns:
        Histogram correlation score (higher is more similar)
    """
    try:
        # Use pre-computed histogram or compute it
        if base_histogram is not None:
            hist1 = base_histogram
        else:
            img1 = _load_image(img1_path, image_cache)
            arr1 = np.array(img1, dtype=np.uint8)
            if transform != 'none':
                arr1 = apply_transform(arr1, transform)
            hist1, _ = np.histogram(arr1.flatten(), bins=256, range=(0, 256))
            hist1 = hist1.astype(np.float32) / hist1.sum()
        
        # Compute histogram for second image
        img2 = _load_image(img2_path, image_cache)
        arr2 = np.array(img2, dtype=np.uint8)
        if transform != 'none':
            arr2 = apply_transform(arr2, transform)
        hist2, _ = np.histogram(arr2.flatten(), bins=256, range=(0, 256))
        
        # Normalize
        hist1 = hist1.astype(np.float32) / hist1.sum()
        hist2 = hist2.astype(np.float32) / hist2.sum()
        
        # Compute correlation
        corr = np.corrcoef(hist1, hist2)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0
        
    except Exception as e:
        print(f"Error computing histogram correlation: {e}")
        return 0.0


def compute_chi_square_distance(img1_path: Path, img2_path: Path, transform: str = 'none', image_cache: dict = None, base_histogram: np.ndarray = None) -> float:
    """
    Compute chi-square distance between histograms (lower is more similar).
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        transform: Image transformation to apply
        image_cache: Optional dict of cached images {path: PIL.Image}
        base_histogram: Optional pre-computed histogram for base image (with epsilon added)
        
    Returns:
        Negative chi-square distance (higher is more similar for consistency)
    """
    try:
        # Use pre-computed histogram or compute it
        if base_histogram is not None:
            hist1 = base_histogram
        else:
            img1 = _load_image(img1_path, image_cache)
            arr1 = np.array(img1, dtype=np.uint8)
            if transform != 'none':
                arr1 = apply_transform(arr1, transform)
            hist1, _ = np.histogram(arr1.flatten(), bins=256, range=(0, 256))
            hist1 = hist1.astype(np.float32) + 1e-10
        
        # Compute histogram for second image
        img2 = _load_image(img2_path, image_cache)
        arr2 = np.array(img2, dtype=np.uint8)
        if transform != 'none':
            arr2 = apply_transform(arr2, transform)
        hist2, _ = np.histogram(arr2.flatten(), bins=256, range=(0, 256))
        
        # Add small epsilon to avoid division by zero
        hist2 = hist2.astype(np.float32) + 1e-10
        
        # Chi-square distance
        chi_square = np.sum((hist1 - hist2) ** 2 / (hist1 + hist2))
        
        # Return negative for consistency (higher = more similar)
        return float(-chi_square)
        
    except Exception as e:
        print(f"Error computing chi-square distance: {e}")
        return 0.0


def compute_bhattacharyya_distance(img1_path: Path, img2_path: Path, transform: str = 'none', image_cache: dict = None, base_histogram: np.ndarray = None) -> float:
    """
    Compute Bhattacharyya distance between histograms.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        transform: Image transformation to apply
        image_cache: Optional dict of cached images {path: PIL.Image}
        base_histogram: Optional pre-computed normalized histogram for base image
        
    Returns:
        Negative Bhattacharyya distance (higher is more similar)
    """
    try:
        # Use pre-computed histogram or compute it
        if base_histogram is not None:
            hist1 = base_histogram
        else:
            img1 = _load_image(img1_path, image_cache)
            arr1 = np.array(img1, dtype=np.uint8)
            if transform != 'none':
                arr1 = apply_transform(arr1, transform)
            hist1, _ = np.histogram(arr1.flatten(), bins=256, range=(0, 256))
            hist1 = hist1.astype(np.float32) / hist1.sum()
        
        # Compute histogram for second image
        img2 = _load_image(img2_path, image_cache)
        arr2 = np.array(img2, dtype=np.uint8)
        if transform != 'none':
            arr2 = apply_transform(arr2, transform)
        hist2, _ = np.histogram(arr2.flatten(), bins=256, range=(0, 256))
        
        # Normalize
        hist2 = hist2.astype(np.float32) / hist2.sum()
        
        # Bhattacharyya coefficient
        bc = np.sum(np.sqrt(hist1 * hist2))
        
        # Bhattacharyya distance
        distance = -np.log(bc + 1e-10)
        
        # Return negative for consistency
        return float(-distance)
        
    except Exception as e:
        print(f"Error computing Bhattacharyya distance: {e}")
        return 0.0


def compute_emd(img1_path: Path, img2_path: Path, transform: str = 'none', image_cache: dict = None) -> float:
    """
    Compute Earth Mover's Distance (Wasserstein distance) between histograms.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        transform: Image transformation to apply
        image_cache: Optional dict of cached images {path: PIL.Image}
        
    Returns:
        Negative EMD (higher is more similar)
    """
    try:
        from scipy.stats import wasserstein_distance
        
        img1 = _load_image(img1_path, image_cache)
        img2 = _load_image(img2_path, image_cache)
        
        arr1 = np.array(img1, dtype=np.uint8)
        arr2 = np.array(img2, dtype=np.uint8)
        
        if transform != 'none':
            arr1 = apply_transform(arr1, transform)
            arr2 = apply_transform(arr2, transform)
        
        # Compute EMD on flattened arrays
        emd = wasserstein_distance(arr1.flatten(), arr2.flatten())
        
        # Return negative for consistency
        return float(-emd)
        
    except Exception as e:
        print(f"Error computing EMD: {e}")
        return 0.0


def compute_mae(img1_path: Path, img2_path: Path, transform: str = 'none', image_cache: dict = None) -> float:
    """
    Compute Mean Absolute Error.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        transform: Image transformation to apply
        image_cache: Optional dict of cached images {path: PIL.Image}
        
    Returns:
        Negative MAE (higher is more similar)
    """
    try:
        img1 = _load_image(img1_path, image_cache)
        img2 = _load_image(img2_path, image_cache)
        
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
        
        arr1 = np.array(img1, dtype=np.float32)
        arr2 = np.array(img2, dtype=np.float32)
        
        if transform != 'none':
            arr1 = apply_transform(arr1.astype(np.uint8), transform).astype(np.float32)
            arr2 = apply_transform(arr2.astype(np.uint8), transform).astype(np.float32)
        
        mae = np.mean(np.abs(arr1 - arr2))
        return float(-mae)
        
    except Exception as e:
        print(f"Error computing MAE: {e}")
        return 0.0


def compute_cosine_similarity(img1_path: Path, img2_path: Path, transform: str = 'none', image_cache: dict = None) -> float:
    """
    Compute cosine similarity between images.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        transform: Image transformation to apply
        image_cache: Optional dict of cached images {path: PIL.Image}
        
    Returns:
        Cosine similarity (higher is more similar)
    """
    try:
        img1 = _load_image(img1_path, image_cache)
        img2 = _load_image(img2_path, image_cache)
        
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
        
        arr1 = np.array(img1, dtype=np.float32)
        arr2 = np.array(img2, dtype=np.float32)
        
        if transform != 'none':
            arr1 = apply_transform(arr1.astype(np.uint8), transform).astype(np.float32)
            arr2 = apply_transform(arr2.astype(np.uint8), transform).astype(np.float32)
        
        # Flatten arrays
        vec1 = arr1.flatten()
        vec2 = arr2.flatten()
        
        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        similarity = dot_product / (norm1 * norm2 + 1e-10)
        return float(similarity)
        
    except Exception as e:
        print(f"Error computing cosine similarity: {e}")
        return 0.0


def compute_mutual_information(img1_path: Path, img2_path: Path, transform: str = 'none', image_cache: dict = None) -> float:
    """
    Compute mutual information between images.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        transform: Image transformation to apply
        image_cache: Optional dict of cached images {path: PIL.Image}
        
    Returns:
        Mutual information (higher is more similar)
    """
    try:
        img1 = _load_image(img1_path, image_cache)
        img2 = _load_image(img2_path, image_cache)
        
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
        
        arr1 = np.array(img1, dtype=np.uint8)
        arr2 = np.array(img2, dtype=np.uint8)
        
        if transform != 'none':
            arr1 = apply_transform(arr1, transform)
            arr2 = apply_transform(arr2, transform)
        
        # Compute 2D histogram
        hist_2d, _, _ = np.histogram2d(arr1.flatten(), arr2.flatten(), bins=256)
        
        # Compute marginal distributions
        pxy = hist_2d / hist_2d.sum()
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)
        
        # Compute mutual information
        px_py = px[:, None] * py[None, :]
        
        # Avoid log(0)
        nzs = pxy > 0
        mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / (px_py[nzs] + 1e-10)))
        
        return float(mi)
        
    except Exception as e:
        print(f"Error computing mutual information: {e}")
        return 0.0


def compute_hog_similarity(img1_path: Path, img2_path: Path, transform: str = 'none', image_cache: dict = None, base_hog: np.ndarray = None) -> float:
    """
    Compute similarity based on Histogram of Oriented Gradients.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        transform: Image transformation to apply
        image_cache: Optional dict of cached images {path: PIL.Image}
        base_hog: Optional pre-computed normalized HOG histogram for base image
        
    Returns:
        HOG similarity (higher is more similar)
    """
    try:
        # Use pre-computed HOG or compute it
        if base_hog is not None:
            hist1 = base_hog
        else:
            img1 = _load_image(img1_path, image_cache)
            arr1 = np.array(img1, dtype=np.uint8)
            if transform != 'none':
                arr1 = apply_transform(arr1, transform)
            gx1 = np.gradient(arr1.astype(float), axis=1)
            gy1 = np.gradient(arr1.astype(float), axis=0)
            mag1 = np.sqrt(gx1**2 + gy1**2)
            ori1 = np.arctan2(gy1, gx1)
            bins = 9
            hist1, _ = np.histogram(ori1.flatten(), bins=bins, range=(-np.pi, np.pi), weights=mag1.flatten())
            hist1 = hist1 / (hist1.sum() + 1e-10)
        
        # Compute HOG for second image
        img2 = _load_image(img2_path, image_cache)
        arr2 = np.array(img2, dtype=np.uint8)
        if transform != 'none':
            arr2 = apply_transform(arr2, transform)
        
        gx2 = np.gradient(arr2.astype(float), axis=1)
        gy2 = np.gradient(arr2.astype(float), axis=0)
        mag2 = np.sqrt(gx2**2 + gy2**2)
        ori2 = np.arctan2(gy2, gx2)
        bins = 9
        hist2, _ = np.histogram(ori2.flatten(), bins=bins, range=(-np.pi, np.pi), weights=mag2.flatten())
        
        # Normalize
        hist2 = hist2 / (hist2.sum() + 1e-10)
        
        # Compute correlation
        corr = np.corrcoef(hist1, hist2)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0
        
    except Exception as e:
        print(f"Error computing HOG similarity: {e}")
        return 0.0


def compute_perceptual_hash(img1_path: Path, img2_path: Path, transform: str = 'none', image_cache: dict = None) -> float:
    """
    Compute perceptual hash similarity.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        transform: Image transformation to apply
        image_cache: Optional dict of cached images {path: PIL.Image}
        
    Returns:
        Similarity based on hash distance (higher is more similar)
    """
    try:
        img1 = _load_image(img1_path, image_cache)
        img2 = _load_image(img2_path, image_cache)
        
        arr1 = np.array(img1, dtype=np.uint8)
        arr2 = np.array(img2, dtype=np.uint8)
        
        if transform != 'none':
            arr1 = apply_transform(arr1, transform)
            arr2 = apply_transform(arr2, transform)
            img1 = Image.fromarray(arr1)
            img2 = Image.fromarray(arr2)
        
        # Resize to 8x8
        img1_small = img1.resize((8, 8), Image.Resampling.LANCZOS)
        img2_small = img2.resize((8, 8), Image.Resampling.LANCZOS)
        
        arr1_small = np.array(img1_small, dtype=np.float64)
        arr2_small = np.array(img2_small, dtype=np.float64)
        
        # Compute DCT (simplified using mean as threshold)
        mean1 = arr1_small.mean()
        mean2 = arr2_small.mean()
        
        hash1 = arr1_small > mean1
        hash2 = arr2_small > mean2
        
        # Hamming distance
        hamming = np.sum(hash1 != hash2)
        
        # Convert to similarity (0-64 bits, higher similarity = lower hamming)
        similarity = 1.0 - (hamming / 64.0)
        
        return float(similarity)
        
    except Exception as e:
        print(f"Error computing perceptual hash: {e}")
        return 0.0


def compute_difference_hash(img1_path: Path, img2_path: Path, transform: str = 'none', image_cache: dict = None) -> float:
    """
    Compute difference hash similarity.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        transform: Image transformation to apply
        image_cache: Optional dict of cached images {path: PIL.Image}
        
    Returns:
        Similarity based on dHash (higher is more similar)
    """
    try:
        img1 = _load_image(img1_path, image_cache)
        img2 = _load_image(img2_path, image_cache)
        
        arr1 = np.array(img1, dtype=np.uint8)
        arr2 = np.array(img2, dtype=np.uint8)
        
        if transform != 'none':
            arr1 = apply_transform(arr1, transform)
            arr2 = apply_transform(arr2, transform)
            img1 = Image.fromarray(arr1)
            img2 = Image.fromarray(arr2)
        
        # Resize to 9x8
        img1_small = img1.resize((9, 8), Image.Resampling.LANCZOS)
        img2_small = img2.resize((9, 8), Image.Resampling.LANCZOS)
        
        arr1_small = np.array(img1_small, dtype=np.float64)
        arr2_small = np.array(img2_small, dtype=np.float64)
        
        # Compute horizontal gradient
        hash1 = arr1_small[:, 1:] > arr1_small[:, :-1]
        hash2 = arr2_small[:, 1:] > arr2_small[:, :-1]
        
        # Hamming distance
        hamming = np.sum(hash1 != hash2)
        
        # Convert to similarity
        similarity = 1.0 - (hamming / 64.0)
        
        return float(similarity)
        
    except Exception as e:
        print(f"Error computing difference hash: {e}")
        return 0.0


def compute_sift_similarity(img1_path: Path, img2_path: Path, transform: str = 'none', image_cache: dict = None, base_sift_data: tuple = None) -> float:
    """
    Compute SIFT feature matching similarity using inlier ratio.
    
    Extracts SIFT keypoints and descriptors, matches them, and uses RANSAC
    to find geometric transformation. Returns the inlier ratio as similarity.
    
    Args:
        img1_path: Path to first image (base image)
        img2_path: Path to second image
        transform: Image transformation to apply before SIFT extraction
        image_cache: Optional dict of cached images {path: PIL.Image}
        base_sift_data: Optional pre-computed (keypoints, descriptors) for base image
        
    Returns:
        Inlier ratio (0.0 to 1.0, higher is more similar)
    """
    try:
        # Extract or use pre-computed SIFT features for base image
        if base_sift_data is not None:
            keypoints1, descriptors1 = base_sift_data
        else:
            # Load and process base image
            img1 = _load_image(img1_path, image_cache)
            arr1 = np.array(img1, dtype=np.uint8)
            
            if transform != 'none':
                arr1 = apply_transform(arr1, transform)
            
            # Normalize to 0-1 range for SIFT (float32 is sufficient and faster)
            arr1 = arr1.astype(np.float32) / 255.0
            
            # Extract SIFT features with reduced parameters for speed
            sift1 = SIFT(n_octaves=3, n_scales=3, sigma_min=1.6)
            sift1.detect_and_extract(arr1)
            keypoints1 = sift1.keypoints
            descriptors1 = sift1.descriptors
        
        # Load and process second image
        img2 = _load_image(img2_path, image_cache)
        arr2 = np.array(img2, dtype=np.uint8)
        
        if transform != 'none':
            arr2 = apply_transform(arr2, transform)
        
        # Normalize to 0-1 range for SIFT (float32 is sufficient and faster)
        arr2 = arr2.astype(np.float32) / 255.0
        
        # Extract SIFT features with reduced parameters for speed
        sift2 = SIFT(n_octaves=3, n_scales=3, sigma_min=1.6)
        sift2.detect_and_extract(arr2)
        keypoints2 = sift2.keypoints
        descriptors2 = sift2.descriptors
        
        # Check if enough features were found
        if len(keypoints1) < 4 or len(keypoints2) < 4:
            return 0.0
        
        # Match descriptors using ratio test
        matches = match_descriptors(descriptors1, descriptors2, cross_check=True, max_ratio=0.8)
        
        if len(matches) < 4:
            return 0.0
        
        # Extract matched keypoint coordinates
        src_pts = keypoints1[matches[:, 0]][:, :2]
        dst_pts = keypoints2[matches[:, 1]][:, :2]
        
        # Use RANSAC to find affine transform and filter outliers
        try:
            model, inliers = ransac(
                (src_pts, dst_pts),
                AffineTransform,
                min_samples=3,
                residual_threshold=2.0,
                max_trials=500
            )
            
            num_inliers = np.sum(inliers)
            inlier_ratio = num_inliers / len(matches)
            
            return float(inlier_ratio)
            
        except Exception as ransac_error:
            # RANSAC failed (not enough good matches)
            return 0.0
        
    except Exception as e:
        print(f"Error computing SIFT similarity for {img2_path.name}: {e}")
        return 0.0
