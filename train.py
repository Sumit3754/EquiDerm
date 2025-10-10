import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

# --- Configuration ---
TRAIN_DIR = 'data/train'
VALIDATION_DIR = 'data/validation'
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 30
FINE_TUNE_EPOCHS = 10 # Number of epochs for fine-tuning
INNER_LAYER_SIZE = 25
DROPRATE = 0.2

# --- Data Preparation and Augmentation ---
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# --- Class Weight Calculation ---
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights_dict = dict(enumerate(class_weights))
print("Calculated class weights:", class_weights_dict)

# --- Model Definition (Transfer Learning) ---
def make_model(input_shape, droprate, inner_layer_size):
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False
    
    inputs = keras.Input(shape=input_shape)
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    dense = keras.layers.Dense(inner_layer_size, activation='relu')(vectors)
    dropout = keras.layers.Dropout(droprate)(dense)
    outputs = keras.layers.Dense(7, activation="linear")(dropout)
    
    return keras.Model(inputs, outputs)

# --- Initial Training ---
model = make_model(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    droprate=DROPRATE,
    inner_layer_size=INNER_LAYER_SIZE
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

callbacks = [
    ModelCheckpoint(
        'skin_lesion_model_best.h5', # Always save the best model to the same file
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    ),
    EarlyStopping(
        monitor='val_loss', 
        patience=5,
        restore_best_weights=True
    )
]

print("Starting initial training...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    class_weight=class_weights_dict
)

# --- Fine-Tuning ---
print("\nStarting fine-tuning...")
base_model = model.layers[1]
base_model.trainable = True

# Freeze the bottom layers and only fine-tune the top ones
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5), # Use a very low learning rate
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

total_epochs = EPOCHS + FINE_TUNE_EPOCHS
history_fine_tune = model.fit(
    train_generator,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1] + 1,
    validation_data=validation_generator,
    callbacks=callbacks,
    class_weight=class_weights_dict
)

print("Fine-tuning complete!")

# --- Plotting Results ---
acc = history.history['accuracy'] + history_fine_tune.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine_tune.history['val_accuracy']
loss = history.history['loss'] + history_fine_tune.history['loss']
val_loss = history.history['val_loss'] + history_fine_tune.history['val_loss']

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Model Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Model Loss')
plt.legend()

plt.savefig('improved_training_history.png')
print("\nImproved training history plot saved as improved_training_history.png")