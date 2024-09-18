from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, Callback
import os


# Custom callback to dynamically adjust augmentation based on loss difference
class DynamicAugmentationCallback(Callback):
    def __init__(self, train_datagen, mild_params, medium_params, aggressive_params, threshold_medium, threshold_aggressive):
        super(DynamicAugmentationCallback, self).__init__()
        self.train_datagen = train_datagen
        self.mild_params = mild_params
        self.medium_params = medium_params
        self.aggressive_params = aggressive_params
        self.threshold_medium = threshold_medium
        self.threshold_aggressive = threshold_aggressive

    def set_augmentation(self, params):
        self.train_datagen.rotation_range = params['rotation_range']
        self.train_datagen.width_shift_range = params['width_shift_range']
        self.train_datagen.height_shift_range = params['height_shift_range']
        self.train_datagen.shear_range = params['shear_range']
        self.train_datagen.zoom_range = [params['zoom_range'], params['zoom_range']] \
                                        if isinstance(params['zoom_range'], float) else params['zoom_range']
        self.train_datagen.brightness_range = params['brightness_range']
        self.train_datagen.channel_shift_range = params['channel_shift_range']
        self.train_datagen.horizontal_flip = params['horizontal_flip']
        self.train_datagen.vertical_flip = params['vertical_flip']
        print(f"Updated augmentation to: {params}")

    def on_epoch_end(self, epoch, logs=None):
        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        loss_diff = val_loss - train_loss

        if loss_diff > self.threshold_aggressive:
            print("Applying Aggressive augmentation")
            self.set_augmentation(self.aggressive_params)
        elif loss_diff > self.threshold_medium:
            print("Applying Medium augmentation")
            self.set_augmentation(self.medium_params)
        else:
            print("Applying Mild augmentation")
            self.set_augmentation(self.mild_params)

def get_callbacks(config, results_dir, train_datagen=None):
    callbacks = []
    # Early Stopping Callback
    if config['training']['callbacks'].get('early_stopping', False):
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)

    # Model Checkpoint Callback
    if config['training']['callbacks'].get('model_checkpoint', False):
        checkpoint = ModelCheckpoint(
            filepath=os.path.join(results_dir, 'best_model.h5'),
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        )
        callbacks.append(checkpoint)

    # Learning Rate Scheduler Callback
    lr_scheduler_type = config['training']['callbacks'].get('learning_rate_scheduler', 'none')
    if lr_scheduler_type == 'reduce_on_plateau':
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1
        )
        callbacks.append(lr_scheduler)
    elif lr_scheduler_type == 'exponential_decay':
        def exponential_decay(epoch, lr):
            return lr * 0.96 ** (epoch // 10)
        lr_scheduler = LearningRateScheduler(exponential_decay, verbose=1)
        callbacks.append(lr_scheduler)

    # Check for augmentation and dynamic augmentation
    if config['data'].get('augmentation', False) and config['training']['callbacks'].get('dynamic_augmentation', False):
        # Define augmentation parameter ranges for each stage
        mild_params = {
            'rotation_range': 10,
            'width_shift_range': [-0.05, 0.05],
            'height_shift_range': [-0.05, 0.05],
            'shear_range': 0.01,
            'zoom_range': [0.95, 1.05],
            'brightness_range': [0.9, 1.1],
            'channel_shift_range': 10,
            'horizontal_flip': True,
            'vertical_flip': False
        }

        medium_params = {
            'rotation_range': 20,
            'width_shift_range': [-0.1, 0.1],
            'height_shift_range': [-0.1, 0.1],
            'shear_range': 0.05,
            'zoom_range': [0.9, 1.1],
            'brightness_range': [0.8, 1.4],
            'channel_shift_range': 20,
            'horizontal_flip': True,
            'vertical_flip': True
        }

        aggressive_params = {
            'rotation_range': 30,
            'width_shift_range': [-0.2, 0.2],
            'height_shift_range': [-0.2, 0.2],
            'shear_range': 0.1,
            'zoom_range': [0.8, 1.2],
            'brightness_range': [0.7, 1.5],
            'channel_shift_range': 30,
            'horizontal_flip': True,
            'vertical_flip': True
        }

        # Define thresholds for dynamic augmentation adjustment
        threshold_medium = 0.05
        threshold_aggressive = 0.1

        # Local import to avoid circular import
        from data_preprocessing import get_train_data_gen  
        train_datagen = get_train_data_gen()

        # Instantiate the DynamicAugmentationCallback
        dynamic_augmentation_callback = DynamicAugmentationCallback(
            train_datagen=train_datagen,
            mild_params=mild_params,
            medium_params=medium_params,
            aggressive_params=aggressive_params,
            threshold_medium=threshold_medium,
            threshold_aggressive=threshold_aggressive
        )
        callbacks.append(dynamic_augmentation_callback)

    return callbacks
