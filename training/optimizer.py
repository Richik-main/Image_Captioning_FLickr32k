import torch

def get_optimizer(models, learning_rate=1e-4, weight_decay=1e-2):
    """
    Initializes an AdamW optimizer for the given models.

    Args:
        models (list): List of models for which the optimizer parameters are defined.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for regularization.

    Returns:
        torch.optim.AdamW: Initialized optimizer.
    """
    parameters = []
    for model in models:
        parameters += list(model.parameters())

    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    return optimizer

