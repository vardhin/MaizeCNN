import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import gc  # For garbage collection

# Define constants
IMG_WIDTH = 400
IMG_HEIGHT = 400
IMG_CHANNELS = 3
BATCH_SIZE = 32
DATASET_PATH = 'dataset'

# Classes in the dataset
CLASSES = [
    'maize_armyworm', 
    'maize_grasshopper', 
    'maize_healthy', 
    'maize_leafbeetle',
    'maize_leafblight', 
    'maize_leafspot', 
    'maize_streakvirus'
]

def is_valid_image(img_path):
    """Check if an image file is valid and not corrupted"""
    try:
        with Image.open(img_path) as img:
            img.verify()  # Verify the file is a valid image
        return True
    except Exception:
        return False

def get_image_paths_and_labels():
    """Get paths to all valid images and their labels without loading them"""
    image_paths = []
    labels = []
    
    print("Finding valid images...")
    
    # Track statistics
    total_images = 0
    corrupted_images = 0
    
    # Loop through each class folder
    for class_idx, class_name in enumerate(CLASSES):
        class_path = os.path.join(DATASET_PATH, class_name)
        print(f"Processing {class_name} images...")
        
        # Skip if folder doesn't exist
        if not os.path.exists(class_path):
            print(f"Warning: {class_path} does not exist, skipping.")
            continue
            
        # Get all image files in the folder
        image_files = [f for f in os.listdir(class_path) 
            if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        total_class_images = len(image_files)
        processed_class_images = 0
        
        # Process each image
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            total_images += 1
            
            # Check if image is valid before adding it
            if not is_valid_image(img_path):
                print(f"Skipping corrupted image: {img_path}")
                corrupted_images += 1
                continue
                
            image_paths.append(img_path)
            labels.append(class_idx)
            processed_class_images += 1
        
        print(f"Found {processed_class_images}/{total_class_images} valid images in class {class_name}")
    
    print(f"Found {len(image_paths)} valid images across {len(CLASSES)} classes")
    print(f"Skipped {corrupted_images} corrupted images out of {total_images} total")
    
    return image_paths, labels

def load_and_preprocess_image(img_path):
    """Load and preprocess a single image"""
    try:
        with Image.open(img_path) as img:
            img = img.resize((IMG_WIDTH, IMG_HEIGHT))
            img = img.convert('RGB')  # Ensure 3 channels
            # Convert to numpy array and normalize
            img_array = np.array(img) / 255.0
            return img_array
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return None

def visualize_samples(image_paths, labels, num_samples=5):
    """Visualize random samples from each class"""
    plt.figure(figsize=(15, 10))
    
    for class_idx, class_name in enumerate(CLASSES):
        # Get indices for this class
        indices = np.where(np.array(labels) == class_idx)[0]
        
        # Skip if no images for this class
        if len(indices) == 0:
            continue
            
        # Randomly select samples
        samples = np.random.choice(indices, 
                                size=min(num_samples, len(indices)), 
                                replace=False)
        
        # Plot samples
        for i, sample_idx in enumerate(samples):
            img_path = image_paths[sample_idx]
            img_array = load_and_preprocess_image(img_path)
            if img_array is not None:
                plt.subplot(len(CLASSES), num_samples, class_idx * num_samples + i + 1)
                plt.imshow(img_array)
                plt.title(class_name if i == 0 else "")
                plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_images.png')
    plt.close()
    print("Sample visualization saved as 'sample_images.png'")

def process_and_save_data(image_paths, labels, output_dir='preprocessed_data'):
    """Process images in batches and save directly to disk"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Split data paths: 70% train, 15% validation, 15% test
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=0.15, stratify=labels, random_state=42
    )
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=0.1765, stratify=train_labels, random_state=42
    )
    
    print(f"Train set: {len(train_paths)} images")
    print(f"Validation set: {len(val_paths)} images")
    print(f"Test set: {len(test_paths)} images")
    
    # Save each split's paths and labels
    np.save(f'{output_dir}/train_paths.npy', train_paths)
    np.save(f'{output_dir}/train_labels.npy', train_labels)
    np.save(f'{output_dir}/val_paths.npy', val_paths)
    np.save(f'{output_dir}/val_labels.npy', val_labels)
    np.save(f'{output_dir}/test_paths.npy', test_paths)
    np.save(f'{output_dir}/test_labels.npy', test_labels)
    
    print("Data split information saved")

class MemoryEfficientDataGenerator(tf.keras.utils.Sequence):
    """Memory-efficient data generator that loads images on demand"""
    def __init__(self, image_paths, labels, batch_size=32, shuffle=True, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = np.arange(len(image_paths))
        self.datagen = None
        
        if augment:
            self.datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
            
        self.on_epoch_end()
        
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Initialize batch arrays
        batch_images = []
        batch_labels = []
        
        # Load and process images for this batch
        for i in batch_indexes:
            img_array = load_and_preprocess_image(self.image_paths[i])
            if img_array is not None:
                batch_images.append(img_array)
                batch_labels.append(self.labels[i])
        
        # Convert to numpy arrays
        X = np.array(batch_images)
        y = tf.keras.utils.to_categorical(batch_labels, len(CLASSES))
        
        # Apply augmentation if needed
        if self.augment and self.datagen:
            # Get a batch of augmented images
            for x_batch, y_batch in self.datagen.flow(X, y, batch_size=len(X), shuffle=False):
                X, y = x_batch, y_batch
                break
                
        return X, y
    
    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)

def create_data_generators(train_paths, train_labels, val_paths, val_labels, test_paths, test_labels):
    """Create memory-efficient data generators"""
    train_generator = MemoryEfficientDataGenerator(
        train_paths, train_labels, batch_size=BATCH_SIZE, shuffle=True, augment=True
    )
    
    val_generator = MemoryEfficientDataGenerator(
        val_paths, val_labels, batch_size=BATCH_SIZE, shuffle=False, augment=False
    )
    
    test_generator = MemoryEfficientDataGenerator(
        test_paths, test_labels, batch_size=BATCH_SIZE, shuffle=False, augment=False
    )
    
    return train_generator, val_generator, test_generator

def main():
    try:
        # Get paths and labels without loading images
        image_paths, labels = get_image_paths_and_labels()
        
        # Visualize some samples
        visualize_samples(image_paths, labels)
        print("Visualized samples, now processing data splits")
        
        # Process and save data splits
        process_and_save_data(image_paths, labels)
        
        # Load split information
        train_paths = np.load('preprocessed_data/train_paths.npy')
        train_labels = np.load('preprocessed_data/train_labels.npy')
        val_paths = np.load('preprocessed_data/val_paths.npy')
        val_labels = np.load('preprocessed_data/val_labels.npy')
        test_paths = np.load('preprocessed_data/test_paths.npy')
        test_labels = np.load('preprocessed_data/test_labels.npy')
        
        # Create data generators
        train_generator, val_generator, test_generator = create_data_generators(
            train_paths, train_labels, val_paths, val_labels, test_paths, test_labels
        )
        
        print("Preprocessing complete!")
        return train_generator, val_generator, test_generator
    
    except Exception as e:
        print(f"An error occurred during processing: {str(e)}")
        return None, None, None

if __name__ == "__main__":
    train_generator, val_generator, test_generator = main()
    print("Script completed successfully")