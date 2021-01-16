
def dice_coeff(pred, target):
    """
    Compute Dice Coefficient
    """
    eps = 1e-10
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2.0 * intersection) / (m1.sum() + m2.sum() + eps)
