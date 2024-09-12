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
from instance_control import stop_instance
import time


def get_dvc_hash(dvc_file_path):
    """Get the DVC hash of a tracked file or directory."""
    try:
        if not os.path.exists(dvc_file_path):
            print(f"DVC file {dvc_file_path} does not exist.")
            return None

        with open(dvc_file_path, 'r') as f:
            dvc_file_content = yaml.safe_load(f)

        md5_hash = dvc_file_content['outs'][0]['md5']
        if md5_hash.endswith('.dir'):
            md5_hash = md5_hash.replace('.dir', '')

        return md5_hash
    except Exception as e:
        print(f"Error retrieving DVC hash for {dvc_file_path}: {e}")
        return None


# Load the configuration
config = load_config()
instance_id = config['instance_id']
dataset_name = config['dataset_name']

# Perform hyperparameter tuning if required
if config['tuning']['perform_tuning']:
    best_hps = tune_hyperparameters()
    
    custom_layers = []
    for i in range(best_hps.get('num_layers')):
        custom_layers.append({'type': 'Dense', 'units': best_hps.get(f'units_{i}'), 'activation': 'relu', 'l2': best_hps.get(f'l2_{i}')})
        custom_layers.append({'type': 'Dropout', 'rate': best_hps.get(f'dropout_rate_{i}')})

    custom_layers.append({'type': 'Dense', 'units': 1, 'activation': 'sigmoid', 'l2': best_hps.get('l2_output')})

    config_override = {
        'model': {
            'type': best_hps.get('base_model'),
            'custom_layers': custom_layers
        },
        'training': {
            'learning_rate': float(best_hps.get('learning_rate')),
            'optimizer': best_hps.get('optimizer')
        }
    }

    config['model'].update(config_override['model'])
    config['training'].update(config_override['training'])
    best_img_size =  best_hps.get('image_size')
    config['model']['input_shape'] = [best_img_size, best_img_size, 3]
    config['data']['img_size'] = [best_img_size, best_img_size]
    print(f"After hyper parameter tuning input shape: {config['model']['input_shape']} and target_size is {config['data']['img_size']}")

# Directories
train_dir = config['data']['train_dir']
val_dir = config['data']['val_dir']
test_dir = config['data']['test_dir']
results_dir = f'results/{dataset_name}'
metrics_dir = f'metrics/{dataset_name}'

# Create directories if they don't exist
os.makedirs(results_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)

# MLflow experiment setup
experiment_name = f"{config['dataset_name']}_experiments"
mlflow.set_experiment(experiment_name)

with mlflow.start_run():
    mlflow.log_params(config['model'])
    mlflow.log_params(config['training'])

    # Track DVC data versions
    train_data_version = get_dvc_hash(f'{train_dir}.dvc')
    val_data_version = get_dvc_hash(f'{val_dir}.dvc')
    test_data_version = get_dvc_hash(f'{test_dir}.dvc')

    mlflow.log_param("train_data_version", train_data_version)
    mlflow.log_param("val_data_version", val_data_version)
    mlflow.log_param("test_data_version", test_data_version)

    # Create data generators
    img_size = tuple(config['model']['input_shape'][:2])
    train_generator, validation_generator, test_generator = create_data_generators(
        train_dir, val_dir, test_dir, img_size=img_size, batch_size=config['training']['batch_size'], config=config
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
        callbacks=callbacks
    )

    # Log metrics and losses
    for epoch in range(len(history.history['loss'])):
        for metric_key, values in history.history.items():
            metric_name = metric_key.replace('val_', 'val-')
            mlflow.log_metric(metric_name, values[epoch], step=epoch)

    # Save and log training metrics plots
    plot_metrics(history, save_dir=metrics_dir)
    mlflow.log_artifact(os.path.join(metrics_dir, 'accuracy_plot.png'))
    mlflow.log_artifact(os.path.join(metrics_dir, 'loss_plot.png'))

    # Evaluate the model on the test set
    evaluation_results = model.evaluate(test_generator)
    test_loss = evaluation_results[0]
    test_accuracy = evaluation_results[1]
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_accuracy", test_accuracy)

    # Confusion Matrix
    y_true = test_generator.classes
    y_pred = model.predict(test_generator)
    y_pred_classes = np.where(y_pred > 0.5, 1, 0).flatten()

    cm = confusion_matrix(y_true, y_pred_classes)
    save_confusion_matrix(cm, save_dir=metrics_dir)
    mlflow.log_artifact(os.path.join(metrics_dir, 'confusion_matrix.png'))

    # Save and track the best model
    best_model_path = os.path.join(results_dir, 'best_model.h5')
    model.save(best_model_path)
    mlflow.log_artifact(best_model_path)

    try:
        subprocess.run(['dvc', 'add', best_model_path], check=True)
        subprocess.run(['dvc', 'push'], check=True)
        subprocess.run(['git', 'add', f'{best_model_path}.dvc', '.gitignore'], check=True)
        subprocess.run(['git', 'commit', '-m', 'Track best model'], check=True)
        subprocess.run(['git', 'push'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error adding/pushing model to DVC: {e}")

    # MLflow Signature (use test input instead of classes)
    test_inputs = next(iter(test_generator))[0]
    signature = mlflow.models.infer_signature(test_inputs, model.predict(test_inputs))
    mlflow.keras.log_model(model, "model", signature=signature)

    mlflow.end_run()

# Stop instance after a 5-minute delay
try:
    print("Training completed. Instance will stop in 5 minutes.")
    time.sleep(300)
    stop_instance(instance_id)
except Exception as e:
    print(f"Failed to stop the instance {instance_id}. Error: {e}")
