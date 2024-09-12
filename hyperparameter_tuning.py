import keras_tuner as kt
from tensorflow.keras.applications import DenseNet121, MobileNetV2
from tensorflow.keras import layers, models, regularizers
from model_build import load_config, get_loss_function, get_metrics
from data_preprocessing import create_data_generators
from optimizers import get_optimizer
from callbacks import get_callbacks
import mlflow
import pandas as pd
def build_model(hp, img_size):
    # Load the configuration
    config = load_config()

    # Choose the base model dynamically
    base_model_type = hp.Choice('base_model', values=config['tuning']['search_space']['base_model'])
    if base_model_type == 'DenseNet121':
        base_model = DenseNet121(
            weights='imagenet' if config['model']['pretrained'] else None,
            include_top=False,
            input_shape=(img_size, img_size, 3)
        )
    elif base_model_type == 'MobileNetV2':
        base_model = MobileNetV2(
            weights='imagenet' if config['model']['pretrained'] else None,
            include_top=False,
            input_shape=(img_size, img_size, 3)
        )
    else:
        raise ValueError(f"Base model {base_model_type} is not supported.")

    # Unfreeze model layers based on config or from a specific layer
    if config['model']['trainable']:
        unfreeze_from_layer = config['model'].get('unfreeze_from_layer', None)
        if unfreeze_from_layer:
            for layer in base_model.layers:
                layer.trainable = False
                if layer.name == unfreeze_from_layer:
                    layer.trainable = True
        else:
            base_model.trainable = True

    # Build the Sequential model
    model = models.Sequential([
        layers.Input(shape=(img_size, img_size, 3)),
        base_model,
        layers.GlobalAveragePooling2D()
    ])

    # Add hyper-tuned Dense and Dropout layers
    num_layers = hp.Int('num_layers', min_value=1, max_value=5)
    for i in range(num_layers):
        units = hp.Choice(f'units_{i}', values=config['tuning']['search_space']['units'])
        dropout_rate = float(hp.Choice(f'dropout_rate_{i}', values=config['tuning']['search_space']['dropout_rate']))
        l2_reg = float(hp.Choice(f'l2_{i}', values=config['tuning']['search_space']['l2']))

        model.add(layers.Dense(units, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
        model.add(layers.Dropout(dropout_rate))

    # Add output layer
    l2_output = float(hp.Choice('l2_output', values=config['tuning']['search_space']['l2']))
    model.add(layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(l2_output)))

    # Compile the model
    learning_rate = float(hp.Choice('learning_rate', values=config['tuning']['search_space']['learning_rate']))
    optimizer = get_optimizer(hp.Choice('optimizer', values=config['tuning']['search_space']['optimizer']), learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=get_loss_function(config['training']['loss_function']),
        metrics=get_metrics(config['training']['metrics'])
    )

    return model


def tune_hyperparameters():
    # Load the configuration
    config = load_config()

    # Select the image size once and use it consistently for both model and data generators
    hp_img_size = max(config['tuning']['search_space']['image_size'])
    
    # Data generators
    train_generator, validation_generator, _ = create_data_generators(
        config['data']['train_dir'], config['data']['val_dir'], config['data']['test_dir'],
        img_size=(hp_img_size, hp_img_size), batch_size=config['training']['batch_size']
    )

    # Initialize the tuner with RandomSearch
    tuner = kt.RandomSearch(
        lambda hp: build_model(hp, hp_img_size),  # Pass the image size to the model building function
        objective=config['tuning']['objective'],
        max_trials=config['tuning']['max_trials'],
        executions_per_trial=config['tuning']['executions_per_trial'],
        directory='tuner_results',
        project_name='image_classification_tuning'
    )

    # Callbacks for training
    callbacks = get_callbacks(config, 'results')

    # Start hyperparameter tuning
    tuner.search(train_generator, validation_data=validation_generator, epochs=config['training']['epochs'], callbacks=callbacks)

    # Retrieve the best hyperparameters and log them to MLflow
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Log hyperparameters and results to MLflow
    mlflow.set_experiment("Hyperparameter Tuning")
    with mlflow.start_run(run_name="Hyperparameter Tuning"):
        mlflow.log_params({
            'base_model': best_hps.get('base_model'),
            'image_size': best_hps.get('image_size'),
            'num_layers': best_hps.get('num_layers'),
            'learning_rate': best_hps.get('learning_rate'),
            'optimizer': best_hps.get('optimizer'),
        })
        for i in range(best_hps.get('num_layers')):
            mlflow.log_param(f'units_layer_{i}', best_hps.get(f'units_{i}'))
            mlflow.log_param(f'dropout_rate_layer_{i}', best_hps.get(f'dropout_rate_{i}'))
            mlflow.log_param(f'l2_reg_layer_{i}', best_hps.get(f'l2_{i}'))
        mlflow.log_param('output_l2', best_hps.get('l2_output'))

        # Log best trial results
        print(f"""
        Best hyperparameters:
        - Base Model: {best_hps.get('base_model')}
        - Image Size: {best_hps.get('image_size')}
        - Number of Layers: {best_hps.get('num_layers')}
        - Units per Layer: {[best_hps.get(f'units_{i}') for i in range(best_hps.get('num_layers'))]}
        - Dropout Rates: {[best_hps.get(f'dropout_rate_{i}') for i in range(best_hps.get('num_layers'))]}
        - L2 Regularization per Layer: {[best_hps.get(f'l2_{i}') for i in range(best_hps.get('num_layers'))]}
        - Output L2: {best_hps.get('l2_output')}
        - Learning Rate: {best_hps.get('learning_rate')}
        - Optimizer: {best_hps.get('optimizer')}
        """)

    return best_hps


if __name__ == '__main__':
    config = load_config()
    if config['tuning']['perform_tuning']:
        best_hps = tune_hyperparameters()
