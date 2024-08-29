from tensorflow.keras import losses, metrics as tf_metrics

def get_loss_function(loss_name):
    """Returns the appropriate loss function based on the configuration."""
    loss_dict = {
        'binary_crossentropy': losses.BinaryCrossentropy,
        'categorical_crossentropy': losses.CategoricalCrossentropy,
        'mean_squared_error': losses.MeanSquaredError,
        'mean_absolute_error': losses.MeanAbsoluteError
    }
    return loss_dict.get(loss_name, losses.BinaryCrossentropy)()

def get_metrics(metrics_list):
    """Returns a list of metrics based on the configuration."""
    available_metrics = {
        'accuracy': tf_metrics.BinaryAccuracy(name='accuracy'),
        'categorical_accuracy': tf_metrics.CategoricalAccuracy(),
        'precision': tf_metrics.Precision(),
        'recall': tf_metrics.Recall(),
        'auc': tf_metrics.AUC(),
        'mean_squared_error': tf_metrics.MeanSquaredError(),
        'mean_absolute_error': tf_metrics.MeanAbsoluteError()
    }
    return [available_metrics[metric] for metric in metrics_list if metric in available_metrics]
