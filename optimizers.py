from tensorflow.keras import optimizers

def get_optimizer(optimizer_name, learning_rate):
    """Returns the appropriate optimizer based on the configuration."""
    optimizer_dict = {
        'adam': optimizers.Adam,
        'sgd': optimizers.SGD,
        'rmsprop': optimizers.RMSprop,
        'adamax': optimizers.Adamax,
        'nadam': optimizers.Nadam,
        'adagrad': optimizers.Adagrad,
        'adadelta': optimizers.Adadelta
    }
    
    if optimizer_name in optimizer_dict:
        return optimizer_dict[optimizer_name](learning_rate=learning_rate)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported.")
