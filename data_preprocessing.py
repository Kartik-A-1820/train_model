from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import tensorflow as tf
from functools import partial
from model_build import load_config

def process_image(image, config):
    resize = tuple(config['data']['img_size'])
    resized_image = cv2.resize(image, resize)

    if config['preprocessing']['apply_preprocessing']:
        if config['model']['type'] == 'DenseNet121':
            processed_image = tf.keras.applications.densenet.preprocess_input(resized_image)
        elif config['model']['type'] == 'MobileNetV2':
            processed_image = tf.keras.applications.mobilenet_v2.preprocess_input(resized_image)
        elif config['preprocessing']['normalize_if_not_found']:
            # Normalize image if model is not found
            processed_image = resized_image / 255.0
        else:
            raise ValueError(f"Unsupported model type: {config['model']['type']}")
    else:
        processed_image = resized_image

    return processed_image

def create_data_generators(train_dir, val_dir, test_dir, img_size=(224, 224), batch_size=32, config=None):
    # Ensure config is passed, otherwise load default config
    if config is None:
        config = load_config()

    # Use functools.partial to pass the config to process_image
    preprocess_func = partial(process_image, config=config)

    # Image Data Generator with augmentation for training
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_func,
        rotation_range=40,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    # Image Data Generator without augmentation for validation and testing
    test_val_datagen = ImageDataGenerator(preprocessing_function=preprocess_func)

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


def get_train_data_gen(config=None):
    # Ensure config is passed, otherwise load default config
    if config is None:
        config = load_config()

    preprocess_func = partial(process_image, config=config)

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_func,
        rotation_range=40,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    return train_datagen
