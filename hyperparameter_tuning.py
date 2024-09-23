import keras_tuner as kt
import mlflow
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import DenseNet121, MobileNetV2
from model_build import load_config, get_loss_function, get_metrics
from data_preprocessing import create_data_generators
from optimizers import get_optimizer
from callbacks import get_callbacks
import mlflow.keras

# Create a HyperModel subclass for custom model building
class SGNNHyperModel(kt.HyperModel):
    
    def __init__(self, img_size=224):
        self.default_img_size = img_size # Ensure it's an integer

    def build(self, hp):
        # Load configuration
        config = load_config()

        # Choose the base model dynamically
        base_model_type = hp.Choice('base_model', values=config['tuning']['search_space']['base_model'])

        # Tune the image size hyperparameter
        img_size = hp.Choice('image_size', values=config['tuning']['search_space']['image_size'], default=self.default_img_size)

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

        # Unfreeze layers if necessary
        if config['model']['trainable']:
            unfreeze_from_layer = config['model'].get('unfreeze_from_layer', None)
            
            if unfreeze_from_layer:
                unfreeze = False  # Flag to track when to start unfreezing
                for layer in base_model.layers:
                    if layer.name == unfreeze_from_layer:
                        unfreeze = True  # Start unfreezing from this layer onward
                    layer.trainable = unfreeze  # Set trainable based on flag
            else:
                base_model.trainable = True  # If no specific layer, unfreeze all layers
        else:
            base_model.trainable = False  # Freeze the entire model if 'trainable' is False

        # Build the model sequentially
        model = models.Sequential([layers.Input(shape=(img_size, img_size, 3)), base_model, layers.GlobalAveragePooling2D()])

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

    def fit(self, hp, model, *args, **kwargs):
        config = load_config()
        experiment_name = f"{config['dataset_name']}_Hyper_Parameter_Tuning"
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run():
            # Log hyperparameters to MLflow
            mlflow.log_params(hp.values)
            # Automatically log TensorFlow metrics
            mlflow.tensorflow.autolog(log_datasets=False)

            # Fit the model
            history = model.fit(*args, **kwargs)

            # Manually infer and log the model signature
            validation_data = kwargs.get('validation_data')
            sample_batch = next(iter(validation_data))  # Fetch a batch of data
            test_data = sample_batch[0]  # Split into inputs only (drop labels)

            # Log the model with explicit input example to infer the signature
            mlflow.keras.log_model(model, "model", 
                                input_example=test_data[:1],  # Pass one sample
                                signature=mlflow.models.infer_signature(test_data, model.predict(test_data)))

            return history



# Tune hyperparameters using a selected tuner
def tune_hyperparameters(tuner_type="random_search"):
    # Load the configuration
    config = load_config()

    # Select image size from config (ensure it's an integer if passed as a list)
    hp_img_size = config['tuning']['search_space']['image_size']
    hp_img_size = hp_img_size if isinstance(hp_img_size, int) else hp_img_size[0]

    # Data generators
    train_generator, validation_generator, _ = create_data_generators(
        config['data']['train_dir'], config['data']['val_dir'], config['data']['test_dir'],
        img_size=(hp_img_size, hp_img_size), batch_size=config['training']['batch_size']
    )

    # Define the HyperModel
    hypermodel = SGNNHyperModel(img_size=hp_img_size)

    # Initialize tuner based on tuner_type
    if tuner_type == "random_search":
        tuner = kt.RandomSearch(
            hypermodel,
            objective=config['tuning']['objective'],
            max_trials=config['tuning']['max_trials'],
            executions_per_trial=config['tuning']['executions_per_trial'],
            directory='tuner_results',
            project_name='image_classification_tuning',
            overwrite=True
        )
    elif tuner_type == "bayesian_optimization":
        tuner = kt.BayesianOptimization(
            hypermodel,
            objective=config['tuning']['objective'],
            max_trials=config['tuning']['max_trials'],
            executions_per_trial=config['tuning']['executions_per_trial'],
            directory='tuner_results',
            project_name='image_classification_tuning',
            overwrite=True
        )
    elif tuner_type == "hyperband":
        tuner = kt.Hyperband(
            hypermodel,
            objective=config['tuning']['objective'],
            max_epochs=config['training']['epochs'],  # Max epochs for Hyperband
            factor=3,  # Hyperband reduction factor
            executions_per_trial=config['tuning']['executions_per_trial'],
            directory='tuner_results',
            project_name='image_classification_tuning',
            overwrite=True
        )
    else:
        raise ValueError(f"Unknown tuner type: {tuner_type}")

    # Callbacks for training
    dataset_name = config['dataset_name']
    results_dir = f'results/{dataset_name}'
    callbacks = get_callbacks(config, f'{results_dir}')

    # Start hyperparameter tuning
    tuner.search(train_generator, validation_data=validation_generator, epochs=config['training']['epochs'], callbacks=callbacks)

    # Retrieve best hyperparameters and log them
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Log hyperparameters and results to MLflow
    mlflow.set_experiment("Hyperparameter Tuning")
    with mlflow.start_run(run_name="Hyperparameter Tuning"):
        mlflow.log_params(best_hps.values)

        print(f"Best hyperparameters: {best_hps.values}")

    return best_hps


# Main function
if __name__ == '__main__':
    config = load_config()
    if config['tuning']['perform_tuning']:
        # Fetch the tuner type from the config file (random_search, bayesian_optimization, hyperband)
        tuner_type = config['tuning'].get('algorithm', 'random_search')  # Default to 'random_search' if not specified
        
        # Pass the tuner type to the hyperparameter tuning function
        best_hps = tune_hyperparameters(tuner_type=tuner_type)
