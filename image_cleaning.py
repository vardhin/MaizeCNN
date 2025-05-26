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

def load_and_preprocess_data():
    """Load images from subfolders and preprocess them"""
    images = []
    labels = []
    
    print("Loading and preprocessing images...")
    
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
            
            # Check if image is valid before attempting to open it
            if not is_valid_image(img_path):
                print(f"Skipping corrupted image: {img_path}")
                corrupted_images += 1
                continue
                
            try:
                # Open and resize image
                with Image.open(img_path) as img:
                    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
                    img = img.convert('RGB')  # Ensure 3 channels
                    
                    # Convert to numpy array and normalize
                    img_array = np.array(img) / 255.0
                
                images.append(img_array)
                labels.append(class_idx)
                processed_class_images += 1
                
                # Periodically force garbage collection
                if len(images) % 500 == 0:
                    gc.collect()
                    
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                corrupted_images += 1
        
        print(f"Processed {processed_class_images}/{total_class_images} images in class {class_name}")
    
    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    print(f"Loaded {len(X)} images across {len(CLASSES)} classes")
    print(f"Skipped {corrupted_images} corrupted images out of {total_images} total")
    
    return X, y

def visualize_samples(X, y, num_samples=5):
    """Visualize random samples from each class"""
    plt.figure(figsize=(15, 10))
    
    for class_idx, class_name in enumerate(CLASSES):
        # Get indices for this class
        indices = np.where(y == class_idx)[0]
        
        # Skip if no images for this class
        if len(indices) == 0:
            continue
            
        # Randomly select samples
        samples = np.random.choice(indices, 
                                size=min(num_samples, len(indices)), 
                                replace=False)
        
        # Plot samples
        for i, sample_idx in enumerate(samples):
            plt.subplot(len(CLASSES), num_samples, class_idx * num_samples + i + 1)
            plt.imshow(X[sample_idx])
            plt.title(class_name if i == 0 else "")
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_images.png')
    plt.close()
    print("Sample visualization saved as 'sample_images.png'")

def create_data_generators(X_train, y_train, X_val, y_val, X_test, y_test):
    """Create data generators with augmentation for training"""
    
    # Create one-hot encoded labels
    y_train_cat = tf.keras.utils.to_categorical(y_train, len(CLASSES))
    y_val_cat = tf.keras.utils.to_categorical(y_val, len(CLASSES))
    y_test_cat = tf.keras.utils.to_categorical(y_test, len(CLASSES))
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # No augmentation for validation and test
    test_datagen = ImageDataGenerator()
    
    # Create generators
    train_generator = train_datagen.flow(
        X_train, y_train_cat,
        batch_size=BATCH_SIZE
    )
    
    val_generator = test_datagen.flow(
        X_val, y_val_cat,
        batch_size=BATCH_SIZE
    )
    
    test_generator = test_datagen.flow(
        X_test, y_test_cat,
        batch_size=BATCH_SIZE
    )
    
    return train_generator, val_generator, test_generator

def save_preprocessed_data(X_train, y_train, X_val, y_val, X_test, y_test):
    """Save preprocessed data to numpy files"""
    os.makedirs('preprocessed_data', exist_ok=True)
    
    np.save('preprocessed_data/X_train.npy', X_train)
    np.save('preprocessed_data/y_train.npy', y_train)
    np.save('preprocessed_data/X_val.npy', X_val)
    np.save('preprocessed_data/y_val.npy', y_val)
    np.save('preprocessed_data/X_test.npy', X_test)
    np.save('preprocessed_data/y_test.npy', y_test)
    
    print("Preprocessed data saved to 'preprocessed_data' directory")

def main():
    try:
        # Load and preprocess data
        X, y = load_and_preprocess_data()
        
        # Visualize some samples
        visualize_samples(X, y)
        
        # Free up memory before splitting
        gc.collect()
        
        # Split data: 70% train, 15% validation, 15% test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.15, stratify=y, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.1765, stratify=y_train_val, random_state=42
        )
        
        # Free memory
        del X, y, X_train_val, y_train_val
        gc.collect()
        
        print(f"Train set: {X_train.shape[0]} images")
        print(f"Validation set: {X_val.shape[0]} images")
        print(f"Test set: {X_test.shape[0]} images")
        
        # Save preprocessed data
        save_preprocessed_data(X_train, y_train, X_val, y_val, X_test, y_test)
        
        # Create data generators
        train_generator, val_generator, test_generator = create_data_generators(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        print("Preprocessing complete!")
        return train_generator, val_generator, test_generator
    
    except Exception as e:
        print(f"An error occurred during processing: {str(e)}")
        return None, None, None

if __name__ == "__main__":
    train_generator, val_generator, test_generator = main()
    print("Script completed successfully")