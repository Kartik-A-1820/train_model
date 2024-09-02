import os
import yaml
import numpy as np
import mlflow
import mlflow.keras
import subprocess  # Import subprocess to run DVC commands
from sklearn.metrics import confusion_matrix
from data_preprocessing import create_data_generators
from model_build import load_config, create_model
from utils import plot_metrics, save_confusion_matrix
from callbacks import get_callbacks
from hyperparameter_tuning import tune_hyperparameters

# Function to get the DVC hash of a file
def get_dvc_hash(file_path):
    """Get the DVC hash of a tracked file."""
    result = subprocess.run(['dvc', 'get', file_path], capture_output=True, text=True)
    return result.stdout.strip()

# Load the configuration
config = load_config()

# Perform hyperparameter tuning if required
if config['tuning']['perform_tuning']:
    best_hps = tune_hyperparameters()
    
    # Build the list of custom layers based on the best hyperparameters
    custom_layers = []
    for i in range(best_hps.get('num_layers')):
        custom_layers.append({'type': 'Dense', 'units': best_hps.get(f'units_{i}'), 'activation': 'relu', 'l2': best_hps.get(f'l2_{i}')})
        custom_layers.append({'type': 'Dropout', 'rate': best_hps.get(f'dropout_rate_{i}')})

    # Add the final output layer
    custom_layers.append({'type': 'Dense', 'units': 1, 'activation': 'sigmoid', 'l2': best_hps.get('l2_output')})

    # Override configuration with the best hyperparameters
    config_override = {
        'model': {
            'type': best_hps.get('base_model'),
            'input_shape': [best_hps.get('image_size'), best_hps.get('image_size'), 3],
            'custom_layers': custom_layers
        },
        'training': {
            'learning_rate': float(best_hps.get('learning_rate')),
            'optimizer': best_hps.get('optimizer')
        }
    }

    config['model'].update(config_override['model'])
    config['training'].update(config_override['training'])

# Directories
train_dir = config['data']['train_dir']
val_dir = config['data']['val_dir']
test_dir = config['data']['test_dir']
results_dir = 'results'
metrics_dir = 'metrics'

# Create directories if they don't exist
os.makedirs(results_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)

# MLflow experiment setup
mlflow.set_experiment("image_classification_experiment")

with mlflow.start_run():
    # Log parameters from config
    mlflow.log_param("batch_size", config['training']['batch_size'])
    mlflow.log_param("epochs", config['training']['epochs'])
    mlflow.log_param("img_size", config['model']['input_shape'])
    mlflow.log_params(config['model'])  # Log model parameters
    mlflow.log_params(config['training'])  # Log training parameters

    # Track DVC data versions
    train_data_version = get_dvc_hash('data/train')
    val_data_version = get_dvc_hash('data/val')
    test_data_version = get_dvc_hash('data/test')

    mlflow.log_param("train_data_version", train_data_version)
    mlflow.log_param("val_data_version", val_data_version)
    mlflow.log_param("test_data_version", test_data_version)

    # Create data generators
    img_size = tuple(config['model']['input_shape'][:2])
    train_generator, validation_generator, test_generator = create_data_generators(
        train_dir, val_dir, test_dir, img_size=img_size, batch_size=config['training']['batch_size']
    )

    # Create model
    model = create_model(config)

    # Get callbacks
    callbacks = get_callbacks(config, results_dir)

    # Train the model
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=config['training']['epochs'],
        callbacks=callbacks  # Use the callbacks
    )

    # Determine the correct accuracy metric key
    accuracy_key = 'binary_accuracy' if 'binary_accuracy' in history.history else 'accuracy'

    # Log metrics and losses
    for epoch in range(len(history.history['loss'])):  # Using 'loss' as a reference for the number of epochs
        for metric_key, values in history.history.items():
            metric_name = metric_key.replace('val_', 'val-')  # MLflow prefers 'val-' prefix for validation metrics
            mlflow.log_metric(metric_name, values[epoch], step=epoch)

    # Save and log training metrics plots
    accuracy_plot_path = os.path.join(metrics_dir, 'accuracy_plot.png')
    loss_plot_path = os.path.join(metrics_dir, 'loss_plot.png')
    plot_metrics(history, save_dir=metrics_dir)
    mlflow.log_artifact(accuracy_plot_path)
    mlflow.log_artifact(loss_plot_path)

    # Evaluate the model on the test set
    evaluation_results = model.evaluate(test_generator)
    test_loss = evaluation_results[0]  # The first value is always the loss
    test_accuracy = evaluation_results[1] 
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_accuracy", test_accuracy)
    print(f'Test Accuracy: {test_accuracy:.2f}')

    # Confusion Matrix
    y_true = test_generator.classes
    y_pred = model.predict(test_generator)
    y_pred_classes = np.where(y_pred > 0.5, 1, 0).flatten()

    cm = confusion_matrix(y_true, y_pred_classes)
    confusion_matrix_path = os.path.join(metrics_dir, 'confusion_matrix.png')
    save_confusion_matrix(cm, save_dir=metrics_dir)
    mlflow.log_artifact(confusion_matrix_path)

    # Save the best model after tuning
    if config['tuning']['perform_tuning']:
        best_model_path = os.path.join(results_dir, 'best_model_tuned.h5')
        model.save(best_model_path)
        subprocess.run(['dvc', 'add', best_model_path])
        subprocess.run(['dvc', 'push'])
        mlflow.log_artifact(best_model_path)

    # Log the model (even if tuning is not performed)
    mlflow.keras.log_model(model, "model")

    # Track the model in DVC
    model_dvc_path = os.path.join(results_dir, 'best_model.h5')
    subprocess.run(['dvc', 'add', model_dvc_path])
    subprocess.run(['dvc', 'push'])
    
    mlflow.end_run()
