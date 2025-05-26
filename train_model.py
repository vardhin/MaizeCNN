import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Import our preprocessing module
import image_cleaning

# Use the same constants from the image cleaning module
IMG_WIDTH = image_cleaning.IMG_WIDTH
IMG_HEIGHT = image_cleaning.IMG_HEIGHT 
IMG_CHANNELS = image_cleaning.IMG_CHANNELS
CLASSES = image_cleaning.CLASSES
NUM_CLASSES = len(CLASSES)

def build_model():
    """
    Build a CNN model using transfer learning with EfficientNetB0
    """
    # Create the base pre-trained model
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Add custom classification head
    inputs = Input(shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history):
    """Plot training and validation accuracy/loss"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def fine_tune_model(model, train_generator, val_generator, epochs=10):
    """Fine-tune the model by unfreezing some layers"""
    # Unfreeze the top layers of the model
    for layer in model.layers[0].layers[-20:]:
        layer.trainable = True
    
    # Recompile the model with a lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Continue training with fine-tuning
    callbacks = [
        ModelCheckpoint('model_fine_tuned.h5', save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)
    ]
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks
    )
    
    return model, history

def evaluate_model(model, test_generator):
    """Evaluate the model on test data"""
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Get predictions
    predictions = []
    true_labels = []
    
    for i in range(len(test_generator)):
        x, y = test_generator[i]
        batch_pred = model.predict(x)
        predictions.extend(np.argmax(batch_pred, axis=1))
        true_labels.extend(np.argmax(y, axis=1))
    
    # Calculate and print classification report
    from sklearn.metrics import classification_report, confusion_matrix
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=CLASSES))
    
    # Plot confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(CLASSES))
    plt.xticks(tick_marks, CLASSES, rotation=90)
    plt.yticks(tick_marks, CLASSES)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    # Get data generators from our preprocessing module
    train_generator, val_generator, test_generator = image_cleaning.main()
    
    if train_generator is None:
        print("Error: Failed to get data generators")
        return
    
    # Build the model
    model = build_model()
    print("Model created. Summary:")
    model.summary()
    
    # Set up callbacks for training
    callbacks = [
        ModelCheckpoint('model_checkpoint.h5', save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    ]
    
    # Train the model (first phase)
    print("\nStarting training...")
    history = model.fit(
        train_generator,
        epochs=20,
        validation_data=val_generator,
        callbacks=callbacks
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Fine-tune the model
    print("\nFine-tuning the model...")
    model, ft_history = fine_tune_model(model, train_generator, val_generator)
    
    # Plot fine-tuning history
    plot_training_history(ft_history)
    
    # Evaluate model on test set
    print("\nEvaluating model on test set...")
    evaluate_model(model, test_generator)
    
    # Save the final model
    model.save('maize_disease_model.h5')
    print("Model saved as 'maize_disease_model.h5'")

if __name__ == "__main__":
    main()