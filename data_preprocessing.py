from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import tensorflow as tf

def process_image(image):
    resized_image = cv2.resize(image,(224, 224))
    processed_image = tf.keras.applications.densenet.preprocess_input(resized_image)
    return processed_image

def create_data_generators(train_dir, val_dir, test_dir, img_size=(224, 224), batch_size=32):
    # Image Data Generator with augmentation for training
    train_datagen = ImageDataGenerator(
        preprocessing_function=process_image,
        rotation_range=40,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip = True,
        fill_mode='nearest'
    )

    # Image Data Generator without augmentation for validation and testing
    test_val_datagen = ImageDataGenerator(preprocessing_function=process_image)

    # Create Generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    validation_generator = test_val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    test_generator = test_val_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    return train_generator, validation_generator, test_generator
