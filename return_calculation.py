import numpy as np

def pad_axis0(array, value):
    """
    Pads an array along the first axis with a constant value.

    Parameters:
    array (np.ndarray): The input array to be padded.
    value (int or float): The constant value to use for padding.

    Returns:
    np.ndarray: The padded array.
    """
    return np.pad(array, pad_width=(0, 1), mode='constant', constant_values=value)

def shift(array):
    """
    Shifts an array downwards by one position, padding the top with zeros.

    Parameters:
    array (np.ndarray): The input array to be shifted.

    Returns:
    np.ndarray: The shifted array.
    """
    return pad_axis0(array, 0)[1:]

def calculate_lambda_returns(rewards, qvalues, dones, mask, discount, lambd):
    """
    Calculates lambda returns for reinforcement learning.

    Parameters:
    rewards (np.ndarray): Array of rewards.
    qvalues (np.ndarray): Array of Q-values.
    dones (np.ndarray): Array indicating if the episode has ended.
    mask (np.ndarray): Mask array.
    discount (float): Discount factor.
    lambd (float): Lambda value for lambda returns.

    Returns:
    np.ndarray: Array of lambda returns.
    """
    dones = dones.astype(np.float32)
    qvalues[-1] *= (1.0 - dones[-1])
    lambda_returns = rewards + (discount * qvalues[1:])
    for i in reversed(range(len(rewards) - 1)):
        a = lambda_returns[i] + (discount * lambd * mask[i]) * (lambda_returns[i + 1] - qvalues[i + 1])
        b = rewards[i]
        lambda_returns[i] = (1.0 - dones[i]) * a + dones[i] * b
    return lambda_returns

def calculate_omega_returns(rewards, qvalues, dones, discount, L):
    """
    Calculates omega returns for reinforcement learning.

    Parameters:
    rewards (np.ndarray): Array of rewards.
    qvalues (np.ndarray): Array of Q-values.
    dones (np.ndarray): Array indicating if the episode has ended.
    discount (float): Discount factor.
    L (int): The number of steps to look ahead for omega returns.

    Returns:
    np.ndarray: Array of omega returns.
    """
    rewards = pad_axis0(rewards, qvalues[-1])
    dones = pad_axis0(dones, 1.0)

    mask = np.ones_like(rewards)
    decay = 1.0
    returns_L = np.copy(rewards)

    for i in range(L):
        decay *= discount
        mask *= (1.0 - dones)

        rewards = shift(rewards)
        qvalues = shift(qvalues)
        dones = shift(dones)

        if i == 0:
            returns_1 = returns_L + (mask * decay * qvalues)

        if i != (L - 1):
            returns_L += (mask * decay * rewards)
        else:
            returns_L += (mask * decay * qvalues)

        v_1 = np.var(returns_1)
        v_L = np.var(returns_L)

        returns = v_L / (v_1 + v_L) * returns_1 + v_1 / (v_1 + v_L) * returns_L

    return returns[:-1]  # Remove bootstrap placeholder

def calculate_nstep_returns(rewards, qvalues, dones, discount, n):
    """
    Calculates n-step returns for reinforcement learning.

    Parameters:
    rewards (np.ndarray): Array of rewards.
    qvalues (np.ndarray): Array of Q-values.
    dones (np.ndarray): Array indicating if the episode has ended.
    discount (float): Discount factor.
    n (int): The number of steps to look ahead for n-step returns.

    Returns:
    np.ndarray: Array of n-step returns.
    """
    rewards = pad_axis0(rewards, qvalues[-1])
    dones = pad_axis0(dones, 1.0)

    mask = np.ones_like(rewards)
    decay = 1.0
    returns = np.copy(rewards)

    for i in range(n):
        decay *= discount
        mask *= (1.0 - dones)

        rewards = shift(rewards)
        qvalues = shift(qvalues)
        dones = shift(dones)

        if i != (n - 1):
            returns += (mask * decay * rewards)
        else:
            returns += (mask * decay * qvalues)

    return returns[:-1]  # Remove bootstrap placeholder
