from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
import os

def get_callbacks(config, results_dir):
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

    return callbacks
