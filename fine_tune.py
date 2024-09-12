import os
import yaml
import numpy as np
import mlflow
import mlflow.keras
import subprocess
from sklearn.metrics import confusion_matrix
from data_preprocessing import create_data_generators
from utils import plot_metrics, save_confusion_matrix
from callbacks import get_callbacks
from tensorflow.keras.models import load_model
import time

def load_config(config_path='config.yaml'):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Prepare dataset paths
    dataset_name = config.get('dataset_name', 'default_dataset')
    config['data']['train_dir'] = config['data']['train_dir'].format(dataset_name=dataset_name)
    config['data']['val_dir'] = config['data']['val_dir'].format(dataset_name=dataset_name)
    config['data']['test_dir'] = config['data']['test_dir'].format(dataset_name=dataset_name)

    # Prepare model path
    model_path = os.path.join('current_models', dataset_name, 'model.h5')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' does not exist.")
    
    config['model_path'] = model_path
    return config

def fine_tune_model(config):
    """Fine-tune a pre-trained model using new data."""
    # Load pre-trained model
    model = load_model(config['model_path'])
    print(f"Loaded model from {config['model_path']}")

    # Create data generators
    img_size = tuple(config['data']['img_size'])
    train_generator, validation_generator, test_generator = create_data_generators(
        config['data']['train_dir'],
        config['data']['val_dir'],
        config['data']['test_dir'],
        img_size=img_size,
        batch_size=config['training']['batch_size'],
        config=config
    )

    # Compile the model
    model.compile(
        optimizer=config['training']['optimizer'],
        loss=config['training']['loss_function'],
        metrics=config['training']['metrics']
    )

    # Set callbacks
    callbacks = get_callbacks(config, results_dir=f"fine_tuned_models/{config['dataset_name']}")

    # Start logging with MLflow
    mlflow.set_experiment(f"{config['dataset_name']}_fine_tuning")

    with mlflow.start_run():
        # Log params
        mlflow.log_params(config['model'])
        mlflow.log_params(config['training'])

        # Train the model
        history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=config['training']['epochs'],
            callbacks=callbacks
        )

        # Log training metrics
        for epoch in range(len(history.history['loss'])):
            for metric_key, values in history.history.items():
                metric_name = metric_key.replace('val_', 'val-')
                mlflow.log_metric(metric_name, values[epoch], step=epoch)

        # Save and log training metrics plots
        metrics_dir = f"metrics/{config['dataset_name']}"
        os.makedirs(metrics_dir, exist_ok=True)
        plot_metrics(history, save_dir=metrics_dir)
        mlflow.log_artifact(os.path.join(metrics_dir, 'accuracy_plot.png'))
        mlflow.log_artifact(os.path.join(metrics_dir, 'loss_plot.png'))

        # Evaluate the model on the test set
        test_loss, test_accuracy = model.evaluate(test_generator)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)

        # Confusion Matrix
        y_true = test_generator.classes
        y_pred = model.predict(test_generator)
        y_pred_classes = np.where(y_pred > 0.5, 1, 0).flatten()

        cm = confusion_matrix(y_true, y_pred_classes)
        save_confusion_matrix(cm, save_dir=metrics_dir)
        mlflow.log_artifact(os.path.join(metrics_dir, 'confusion_matrix.png'))

        # Save fine-tuned model
        fine_tuned_model_dir = f"fine_tuned_models/{config['dataset_name']}"
        os.makedirs(fine_tuned_model_dir, exist_ok=True)
        model.save(os.path.join(fine_tuned_model_dir, 'fine_tuned_model.h5'))
        mlflow.log_artifact(os.path.join(fine_tuned_model_dir, 'fine_tuned_model.h5'))

        # MLflow signature for input-output tracking
        test_inputs = next(iter(test_generator))[0]
        signature = mlflow.models.infer_signature(test_inputs, model.predict(test_inputs))
        mlflow.keras.log_model(model, "model", signature=signature)

        mlflow.end_run()

    print("Fine-tuning complete.")
    return model

# Main function to load config and fine-tune the model
if __name__ == "__main__":
    # Load the configuration
    config = load_config()

    # Fine-tune the model
    fine_tune_model(config)

    # Optional: Stop cloud instance if required
    instance_id = config.get('instance_id', None)
    if instance_id:
        try:
            print("Training completed. Instance will stop in 5 minutes.")
            time.sleep(300)
            stop_instance(instance_id)
        except Exception as e:
            print(f"Failed to stop the instance {instance_id}. Error: {e}")
