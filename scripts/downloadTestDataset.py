"""Download MNIST digit images and CIFAR-10 color images for testing."""

import gzip
import urllib.request
import tarfile
import pickle
from pathlib import Path
import numpy as np
from PIL import Image


def download_file(url: str, destination: Path):
    """Download a file from URL to destination."""
    print(f"  Downloading {url.split('/')[-1]}...")
    urllib.request.urlretrieve(url, destination)


def parse_mnist_images(filepath: Path) -> np.ndarray:
    """Parse MNIST image file format."""
    with gzip.open(filepath, 'rb') as f:
        # Read magic number and dimensions
        magic = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        
        # Read image data
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num_images, rows, cols)
        
    return data


def parse_mnist_labels(filepath: Path) -> np.ndarray:
    """Parse MNIST label file format."""
    with gzip.open(filepath, 'rb') as f:
        # Read magic number and count
        magic = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')
        
        # Read labels
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        
    return labels


def download_mnist_images(output_dir: str, num_images: int = None):
    """
    Download and save MNIST digit images.
    
    Args:
        output_dir: Directory to save images
        num_images: Number of images to download (default: None = all images)
    """
    print(f"Downloading {'all' if num_images is None else num_images} MNIST images...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create temp directory for downloaded files
    temp_dir = output_path.parent / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # MNIST dataset URLs (using reliable mirror)
        base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
        files = {
            "train_images": "train-images-idx3-ubyte.gz",
            "train_labels": "train-labels-idx1-ubyte.gz",
            "test_images": "t10k-images-idx3-ubyte.gz",
            "test_labels": "t10k-labels-idx1-ubyte.gz",
        }
        
        # Download files
        print("Downloading MNIST dataset...")
        train_images_path = temp_dir / files["train_images"]
        train_labels_path = temp_dir / files["train_labels"]
        test_images_path = temp_dir / files["test_images"]
        test_labels_path = temp_dir / files["test_labels"]
        
        if not train_images_path.exists():
            download_file(base_url + files["train_images"], train_images_path)
        if not train_labels_path.exists():
            download_file(base_url + files["train_labels"], train_labels_path)
        if not test_images_path.exists():
            download_file(base_url + files["test_images"], test_images_path)
        if not test_labels_path.exists():
            download_file(base_url + files["test_labels"], test_labels_path)
        
        # Parse MNIST files
        print("Parsing MNIST data...")
        train_images = parse_mnist_images(train_images_path)
        train_labels = parse_mnist_labels(train_labels_path)
        test_images = parse_mnist_images(test_images_path)
        test_labels = parse_mnist_labels(test_labels_path)
        
        # Combine training and test sets for full dataset
        all_images = np.concatenate([train_images, test_images])
        all_labels = np.concatenate([train_labels, test_labels])
        
        # Take only the requested number if specified
        if num_images is not None:
            all_images = all_images[:num_images]
            all_labels = all_labels[:num_images]
        
        total_images = len(all_images)
        
        # Save images
        print(f"Saving {total_images} images (scaling to 280x280)...")
        for idx, (image, label) in enumerate(zip(all_images, all_labels)):
            # Convert to PIL Image
            img = Image.fromarray(image, mode='L')
            
            # Scale by factor of 10 using bilinear interpolation (28x28 -> 280x280)
            img = img.resize((280, 280), Image.BILINEAR)
            
            # Save with label in filename: mnist_0001_digit5.png
            filename = f"mnist_{idx:04d}_digit{label}.png"
            filepath = output_path / filename
            img.save(filepath)
            
            if (idx + 1) % 1000 == 0:
                print(f"  Saved {idx + 1}/{total_images} images...")
        
        # Clean up temp files
        print("Cleaning up temporary files...")
        import shutil
        shutil.rmtree(temp_dir)
        
        print(f"✓ Successfully saved {total_images} images to {output_dir}")
        
    except Exception as e:
        print(f"Error downloading MNIST dataset: {e}")
        return False
    
    return True


def download_cifar10_images(output_dir: str, num_images: int = None):
    """
    Download and save CIFAR-10 color images.
    
    Args:
        output_dir: Directory to save images
        num_images: Number of images to download (default: None = all images)
    """
    print(f"\nDownloading {'all' if num_images is None else num_images} CIFAR-10 images...")
    
    # CIFAR-10 class names
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create temp directory for downloaded files
    temp_dir = output_path.parent / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Download CIFAR-10 tarball
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        tar_path = temp_dir / "cifar-10-python.tar.gz"
        
        if not tar_path.exists():
            print("Downloading CIFAR-10 dataset...")
            download_file(url, tar_path)
        
        # Extract tarball
        print("Extracting CIFAR-10 data...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(temp_dir, filter='data')
        
        cifar_dir = temp_dir / "cifar-10-batches-py"
        
        # Load all data batches
        all_images = []
        all_labels = []
        
        batch_files = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]
        for batch_file in batch_files:
            batch_path = cifar_dir / batch_file
            with open(batch_path, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
            all_images.append(batch[b'data'])
            all_labels.extend(batch[b'labels'])
        
        # Combine all batches
        all_images = np.concatenate(all_images)
        all_labels = np.array(all_labels)
        
        # Reshape: (N, 3072) -> (N, 3, 32, 32) -> (N, 32, 32, 3)
        all_images = all_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        
        # Take only the requested number if specified
        if num_images is not None:
            all_images = all_images[:num_images]
            all_labels = all_labels[:num_images]
        
        total_images = len(all_images)
        
        # Save images
        print(f"Saving {total_images} images (scaling to 64x64)...")
        for idx, (image, label) in enumerate(zip(all_images, all_labels)):
            # Convert to PIL Image (already RGB)
            img = Image.fromarray(image, mode='RGB')
            
            # Scale up using bilinear interpolation (32x32 -> 64x64)
            img = img.resize((64, 64), Image.BILINEAR)
            
            # Save with label in filename: cifar10_0001_cat.png
            class_name = class_names[label]
            filename = f"cifar10_{idx:05d}_{class_name}.png"
            filepath = output_path / filename
            img.save(filepath)
            
            if (idx + 1) % 1000 == 0:
                print(f"  Saved {idx + 1}/{total_images} images...")
        
        # Clean up temp files
        print("Cleaning up temporary files...")
        import shutil
        shutil.rmtree(temp_dir)
        
        print(f"Successfully saved {total_images} images to {output_dir}")
        
    except Exception as e:
        print(f"Error downloading CIFAR-10 dataset: {e}")
        return False
    
    return True


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    
    # Download MNIST (grayscale)
    #mnist_dir = project_root / "data" / "mnist" / "images"
    #success = download_mnist_images(str(mnist_dir))
    #if success:
    #    print(f"\nMNIST images are ready in: {mnist_dir}")
    #    print(f"Total files: {len(list(mnist_dir.glob('*.png')))}")
    
    # Download CIFAR-10 (color)
    cifar_dir = project_root / "data" / "cifar10" / "images"
    success = download_cifar10_images(str(cifar_dir), num_images=1000)
    if success:
        print(f"\nCIFAR-10 images are ready in: {cifar_dir}")
        print(f"Total files: {len(list(cifar_dir.glob('*.png')))}")


if __name__ == "__main__":
    main()
