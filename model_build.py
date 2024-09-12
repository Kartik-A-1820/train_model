import yaml
from tensorflow.keras.applications import DenseNet121, MobileNetV2
from tensorflow.keras import layers, models, regularizers
from metrics_losses import get_loss_function, get_metrics
from optimizers import get_optimizer
from callbacks import get_callbacks  # Import the callbacks

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Ensure img_size and input_shape match
    if 'input_shape' in config['model'] and 'img_size' in config['data']:
        img_size = tuple(config['data']['img_size'])
        input_shape = tuple(config['model']['input_shape'][:2])
        
        if img_size != input_shape:
            # Update the input_shape to match img_size
            config['model']['input_shape'] = [*img_size, 3]
    
    return config

def create_model(config):
    model_config = config['model']
    input_shape = tuple(model_config.get('input_shape', [224, 224, 3]))

    # Load the base model
    if model_config['type'] == 'DenseNet121':
        base_model = DenseNet121(
            weights='imagenet' if model_config.get('pretrained', True) else None,
            include_top=False,
            input_shape=input_shape
        )
    elif model_config['type'] == 'MobileNetV2':
        base_model = MobileNetV2(
            weights='imagenet' if model_config.get('pretrained', True) else None,
            include_top=False,
            input_shape=input_shape
        )
    else:
        raise ValueError(f"Model type {model_config['type']} not supported.")
    
    if model_config.get('trainable', False):
        try:
            base_model.trainable=True
            for layer in base_model.layers:
                if layer.name != f'conv3_block1_0_bn'.strip(): #conv4_block1_0_bn best
                    layer.trainable=False
                else:
                    break
        except:
            base_model.trainable = model_config.get('trainable', False)
    
    else:
        base_model.trainable = model_config.get('trainable', False)

    # Create the Sequential model and add base model
    model = models.Sequential([base_model, layers.GlobalAveragePooling2D()])

    # Add custom layers if specified, or add default Dense layers
    if 'custom_layers' in model_config and model_config['custom_layers']:
        for layer in model_config['custom_layers']:
            layer_type = layer['type']
            if layer_type == 'Dense':
                model.add(layers.Dense(
                    units=layer['units'],
                    activation=layer['activation'],
                    kernel_regularizer=regularizers.l2(layer.get('l2', 0.0))
                ))
            elif layer_type == 'Dropout':
                model.add(layers.Dropout(rate=layer['rate']))
            elif layer_type == 'BatchNormalization':
                model.add(layers.BatchNormalization())
            elif layer_type == 'Flatten':
                model.add(layers.Flatten())
            elif layer_type == 'Activation':
                model.add(layers.Activation(layer['activation']))
            else:
                raise ValueError(f"Layer type {layer_type} not supported.")
    else:
        # Add default Dense layers if no custom layers are specified
        model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        model.add(layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))

    # Compile the model
    loss_function = get_loss_function(config['training']['loss_function'])
    metric_functions = get_metrics(config['training']['metrics'])
    optimizer = get_optimizer(config['training']['optimizer'], float(config['training']['learning_rate']))

    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=metric_functions
    )

    return model
