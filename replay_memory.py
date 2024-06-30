import numpy as np
from scipy.optimize import curve_fit
import re
from return_calculation import calculate_lambda_returns, calculate_omega_returns, calculate_nstep_returns

class ReplayMemory:
    """
    ReplayMemory is a class that stores and manages experiences for reinforcement learning.

    Parameters:
    capacity (int): The maximum number of experiences to store.
    history_len (int): The length of the history for each experience.
    discount (float): The discount factor for future rewards.
    cache_size (int): The size of the cache for storing samples.
    block_size (int): The size of each block of samples.
    priority (float): The priority factor for sampling experiences.
    """
    
    def __init__(self, capacity, history_len, discount, cache_size, block_size, priority):
        """
        Initializes the ReplayMemory.

        Parameters:
        capacity (int): The maximum number of experiences to store.
        history_len (int): The length of the history for each experience.
        discount (float): The discount factor for future rewards.
        cache_size (int): The size of the cache for storing samples.
        block_size (int): The size of each block of samples.
        priority (float): The priority factor for sampling experiences.
        """
        assert (cache_size % block_size) == 0
        self.capacity = capacity + (history_len - 1) + block_size
        self.history_len = history_len
        self.discount = discount
        self.num_samples = 0

        self.cache_size = cache_size
        self.block_size = block_size
        self.priority = priority
        self.refresh_func = None

        self.obs = None  # Allocated dynamically once shape/dtype are known
        self.actions = np.empty([self.capacity], dtype=np.int32)
        self.rewards = np.empty([self.capacity], dtype=np.float32)
        self.dones = np.empty([self.capacity], dtype=np.bool_)
        self.next = 0  # Points to next transition to be overwritten

        self.cached_states = None  # Allocated dynamically once shape/dtype are known
        self.cached_actions = np.empty([self.cache_size], dtype=np.int32)
        self.cached_returns = np.empty([self.cache_size], dtype=np.float32)
        self.cached_errors = np.empty([self.cache_size], dtype=np.float32)
        self.cached_indices = np.empty([self.cache_size], dtype=np.int32)

    def register_refresh_func(self, f):
        """
        Registers a refresh function for updating the cache.

        Parameters:
        f (function): The refresh function.
        """
        assert self.refresh_func is None
        self.refresh_func = f

    def sample(self, batch_size):
        """
        Samples a batch of experiences from the cache.

        Parameters:
        batch_size (int): The number of experiences to sample.

        Returns:
        tuple: A tuple containing arrays of states, actions, and returns.
        """
        start = self.batch_counter * batch_size
        end = start + batch_size
        indices = self.cached_indices[start:end]

        state_batch = self.cached_states[indices]
        action_batch = self.cached_actions[indices]
        return_batch = self.cached_returns[indices]

        self.batch_counter += 1

        return np.array(state_batch), np.array(action_batch), np.array(return_batch)

    def encode_recent_observation(self):
        """
        Encodes the most recent observation.

        Returns:
        np.ndarray: The encoded observation.
        """
        i = self.len()
        return self._encode_observation(i)

    def _encode_observation(self, i):
        """
        Encodes an observation at a given index.

        Parameters:
        i (int): The index of the observation to encode.

        Returns:
        np.ndarray: The encoded observation.
        """
        i = self._align(i)
        state = np.zeros(self.obs[0].shape, dtype=self.obs[0].dtype)
        state = self.obs[i]
        return state

    def _align(self, i):
        """
        Aligns an index relative to the current pointer.

        Parameters:
        i (int): The index to align.

        Returns:
        int: The aligned index.
        """
        if not self.full(): return i
        return (i + self.next) % self.capacity

    def store_obs(self, obs):
        """
        Stores an observation in the memory.

        Parameters:
        obs (np.ndarray): The observation to store.
        """
        if self.obs is None:
            self.obs = np.empty([self.capacity, *obs.shape], dtype=obs.dtype)
        if self.cached_states is None:
            self.cached_states = np.empty([self.cache_size, *obs.shape], dtype=obs.dtype)
        self.obs[self.next] = obs

    def store_effect(self, action, reward, done):
        """
        Stores the effect of an action in the memory.

        Parameters:
        action (int): The action taken.
        reward (float): The reward received.
        done (bool): Whether the episode has ended.
        """
        self.actions[self.next] = action
        self.rewards[self.next] = reward
        self.dones[self.next] = done

        self.next = (self.next + 1) % self.capacity
        self.num_samples = min(self.capacity, self.num_samples + 1)

    def len(self):
        """
        Returns the current number of samples in the memory.

        Returns:
        int: The number of samples.
        """
        return self.num_samples

    def full(self):
        """
        Checks if the memory is full.

        Returns:
        bool: True if the memory is full, False otherwise.
        """
        return self.len() == self.capacity

    def refresh(self, train_frac):
        """
        Refreshes the memory by sampling blocks and updating the cache.

        Parameters:
        train_frac (float): The fraction of the training set to use.
        """
        self.batch_counter = 0
        num_blocks = self.cache_size // self.block_size
        block_ids = self._sample_block_ids(num_blocks)
        self._refresh(train_frac, block_ids)

    def _refresh(self, train_frac, block_ids):
        """
        Refreshes specific blocks in the memory.

        Parameters:
        train_frac (float): The fraction of the training set to use.
        block_ids (np.ndarray): The block indices to refresh.
        """
        for k, i in enumerate(block_ids):
            states = self._extract_block(None, i, states=True)
            actions = self._extract_block(self.actions, i)
            rewards = self._extract_block(self.rewards, i)
            dones = self._extract_block(self.dones, i)

            max_qvalues, mask, onpolicy_qvalues = self.refresh_func(states, actions)
            returns = self._calculate_returns(rewards, max_qvalues, dones, mask)
            errors = np.abs(returns - onpolicy_qvalues)

            start = self.block_size * k
            end = start + self.block_size

            self.cached_states[start:end] = states[:-1]
            self.cached_actions[start:end] = actions
            self.cached_returns[start:end] = returns
            self.cached_errors[start:end] = errors

        distr = self._prioritized_distribution(self.cached_errors, train_frac)
        self.cached_indices = np.random.choice(self.cache_size, size=self.cache_size, replace=True, p=distr)

    def _sample_block_ids(self, n):
        """
        Samples block IDs for refreshing the cache.

        Parameters:
        n (int): The number of block IDs to sample.

        Returns:
        np.ndarray: The sampled block IDs.
        """
        return np.random.randint(self.history_len - 1, self.len() - self.block_size, size=n)

    def _extract_block(self, a, start, states=False):
        """
        Extracts a block of samples from the memory.

        Parameters:
        a (np.ndarray): The array to extract from.
        start (int): The starting index of the block.
        states (bool): Whether to extract states.

        Returns:
        np.ndarray: The extracted block of samples.
        """
        end = start + self.block_size
        if states:
            assert a is None
            return np.array([self._encode_observation(i) for i in range(start, end + 1)])
        return a[self._align(np.arange(start, end))]

    def _prioritized_distribution(self, errors, train_frac):
        """
        Calculates a prioritized distribution for sampling.

        Parameters:
        errors (np.ndarray): The array of errors.
        train_frac (float): The fraction of the training set to use. (see 3.3 Directly prioritized replay)

        Returns:
        np.ndarray: The prioritized distribution.
        """
        distr = np.ones_like(errors) / self.cache_size
        p = self.priority_now(train_frac)
        med = np.median(errors)
        distr[errors > med] *= (1.0 + p)
        distr[errors < med] *= (1.0 - p)
        return distr / distr.sum()

    def priority_now(self, train_frac):
        """
        Calculates the current priority factor.

        Parameters:
        train_frac (float): The fraction of the training set to use. (see 3.3 Directly prioritized replay)

        Returns:
        float: The current priority factor.
        """
        return self.priority * (1.0 - train_frac)

    def _calculate_returns(self, rewards, qvalues, dones, mask):
        """
        Calculates returns based on rewards, Q-values, and dones.

        Parameters:
        rewards (np.ndarray): The array of rewards.
        qvalues (np.ndarray): The array of Q-values.
        dones (np.ndarray): The array of dones.
        mask (np.ndarray): The mask array.

        Returns:
        np.ndarray: The calculated returns.
        """
        raise NotImplementedError


class LambdaReplayMemory(ReplayMemory):
    """
    LambdaReplayMemory is a subclass of ReplayMemory that calculates lambda returns.

    Parameters:
    capacity (int): The maximum number of experiences to store.
    history_len (int): The length of the history for each experience.
    discount (float): The discount factor for future rewards.
    cache_size (int): The size of the cache for storing samples.
    block_size (int): The size of each block of samples.
    priority (float): The priority factor for sampling experiences.
    lambd (float): The lambda parameter for calculating returns.
    use_watkins (bool): Whether to use Watkins' Q(lambda).
    """
    
    def __init__(self, capacity, history_len, discount, cache_size, block_size, priority, lambd, use_watkins):
        """
        Initializes the LambdaReplayMemory.

        Parameters:
        capacity (int): The maximum number of experiences to store.
        history_len (int): The length of the history for each experience.
        discount (float): The discount factor for future rewards.
        cache_size (int): The size of the cache for storing samples.
        block_size (int): The size of each block of samples.
        priority (float): The priority factor for sampling experiences.
        lambd (float): The lambda parameter for calculating returns.
        use_watkins (bool): Whether to use Watkins' Q(lambda).
        """
        self.lambd = lambd
        self.use_watkins = use_watkins
        super().__init__(capacity, history_len, discount, cache_size, block_size, priority)

    def _calculate_returns(self, rewards, qvalues, dones, mask):
        """
        Calculates lambda returns.

        Parameters:
        rewards (np.ndarray): The array of rewards.
        qvalues (np.ndarray): The array of Q-values.
        dones (np.ndarray): The array of dones.
        mask (np.ndarray): The mask array.

        Returns:
        np.ndarray: The calculated lambda returns.
        """
        if not self.use_watkins:
            mask = np.ones_like(qvalues)
        return calculate_lambda_returns(rewards, qvalues, dones, mask, self.discount, self.lambd)


class MedianLambdaReplayMemory(LambdaReplayMemory):
    """
    MedianLambdaReplayMemory is a subclass of LambdaReplayMemory that calculates median lambda returns.

    Parameters:
    capacity (int): The maximum number of experiences to store.
    history_len (int): The length of the history for each experience.
    discount (float): The discount factor for future rewards.
    cache_size (int): The size of the cache for storing samples.
    block_size (int): The size of each block of samples.
    priority (float): The priority factor for sampling experiences.
    use_watkins (bool): Whether to use Watkins' Q(lambda).
    """
    
    def __init__(self, capacity, history_len, discount, cache_size, block_size, priority, use_watkins):
        """
        Initializes the MedianLambdaReplayMemory.

        Parameters:
        capacity (int): The maximum number of experiences to store.
        history_len (int): The length of the history for each experience.
        discount (float): The discount factor for future rewards.
        cache_size (int): The size of the cache for storing samples.
        block_size (int): The size of each block of samples.
        priority (float): The priority factor for sampling experiences.
        use_watkins (bool): Whether to use Watkins' Q(lambda).
        """
        lambd = None
        super().__init__(capacity, history_len, discount, cache_size, block_size, priority, lambd, use_watkins)

    def _calculate_returns(self, rewards, qvalues, dones, mask, k=21):
        """
        Calculates median lambda returns.

        Parameters:
        rewards (np.ndarray): The array of rewards.
        qvalues (np.ndarray): The array of Q-values.
        dones (np.ndarray): The array of dones.
        mask (np.ndarray): The mask array.
        k (int): The number of lambda values to consider.

        Returns:
        np.ndarray: The calculated median lambda returns.
        """
        if not self.use_watkins:
            mask = np.ones_like(qvalues)
        assert k > 1
        returns = np.empty(shape=[k, rewards.size], dtype=np.float32)
        for i in range(0, k):
            returns[i] = calculate_lambda_returns(rewards, qvalues, dones, mask, self.discount, lambd=i/(k-1))
        return np.median(returns, axis=0)


class MeanSquaredTDLambdaReplayMemory(LambdaReplayMemory):
    """
    MeanSquaredTDLambdaReplayMemory is a subclass of LambdaReplayMemory that minimizes mean squared TD error.

    Parameters:
    capacity (int): The maximum number of experiences to store.
    history_len (int): The length of the history for each experience.
    discount (float): The discount factor for future rewards.
    cache_size (int): The size of the cache for storing samples.
    block_size (int): The size of each block of samples.
    priority (float): The priority factor for sampling experiences.
    max_td (float): The maximum allowable TD error.
    use_watkins (bool): Whether to use Watkins' Q(lambda).
    """
    
    def __init__(self, capacity, history_len, discount, cache_size, block_size, priority, max_td, use_watkins):
        """
        Initializes the MeanSquaredTDLambdaReplayMemory.

        Parameters:
        capacity (int): The maximum number of experiences to store.
        history_len (int): The length of the history for each experience.
        discount (float): The discount factor for future rewards.
        cache_size (int): The size of the cache for storing samples.
        block_size (int): The size of each block of samples.
        priority (float): The priority factor for sampling experiences.
        max_td (float): The maximum allowable TD error.
        use_watkins (bool): Whether to use Watkins' Q(lambda).
        """
        lambd = None
        self.max_td = max_td
        super().__init__(capacity, history_len, discount, cache_size, block_size, priority, lambd, use_watkins)

    def _calculate_returns(self, rewards, qvalues, dones, mask, k=7):
        """
        Calculates lambda returns minimizing mean squared TD error.

        Parameters:
        rewards (np.ndarray): The array of rewards.
        qvalues (np.ndarray): The array of Q-values.
        dones (np.ndarray): The array of dones.
        mask (np.ndarray): The mask array.
        k (int): The number of iterations to find the optimal lambda.

        Returns:
        np.ndarray: The calculated lambda returns.
        """
        f = super()._calculate_returns  # Use parent function to compute returns

        returns, ok = self._try_lambda(f, rewards, qvalues, dones, mask, lambd=1.0)
        if ok:
            return returns

        returns, ok = self._try_lambda(f, rewards, qvalues, dones, mask, lambd=0.0)
        if not ok:
            return returns

        best_returns = None
        lambd = 0.5

        for i in range(2, 2 + k):
            returns, ok = self._try_lambda(f, rewards, qvalues, dones, mask, lambd)

            if ok:
                best_returns = returns
                lambd += 1.0 / (2.0 ** i)
            else:
                lambd -= 1.0 / (2.0 ** i)

        return best_returns if best_returns is not None else returns

    def _try_lambda(self, f, rewards, qvalues, dones, mask, lambd):
        """
        Tries a specific lambda value and calculates the returns.

        Parameters:
        f (function): The function to calculate returns.
        rewards (np.ndarray): The array of rewards.
        qvalues (np.ndarray): The array of Q-values.
        dones (np.ndarray): The array of dones.
        mask (np.ndarray): The mask array.
        lambd (float): The lambda value to try.

        Returns:
        tuple: A tuple containing the returns and a boolean indicating if the TD error is acceptable.
        """
        self.lambd = lambd  # Pass implicitly to parent function
        returns = f(rewards, qvalues, dones, mask)
        td_error = np.square(returns - qvalues[:-1]).mean()
        ok = (td_error <= self.max_td)
        return returns, ok
   
class NStepReplayMemory(ReplayMemory):
    """
    NStepReplayMemory is a subclass of ReplayMemory that calculates n-step returns.

    Parameters:
    capacity (int): The maximum number of experiences to store.
    history_len (int): The length of the history for each experience.
    discount (float): The discount factor for future rewards.
    cache_size (int): The size of the cache for storing samples.
    block_size (int): The size of each block of samples.
    priority (float): The priority factor for sampling experiences.
    n (int): The number of steps for n-step returns.
    """
    
    def __init__(self, capacity, history_len, discount, cache_size, block_size, priority, n):
        """
        Initializes the NStepReplayMemory.

        Parameters:
        capacity (int): The maximum number of experiences to store.
        history_len (int): The length of the history for each experience.
        discount (float): The discount factor for future rewards.
        cache_size (int): The size of the cache for storing samples.
        block_size (int): The size of each block of samples.
        priority (float): The priority factor for sampling experiences.
        n (int): The number of steps for n-step returns.
        """
        self.n = n
        super().__init__(capacity, history_len, discount, cache_size, block_size, priority)

    def _calculate_returns(self, rewards, qvalues, dones, mask):
        """
        Calculates n-step returns.

        Parameters:
        rewards (np.ndarray): The array of rewards.
        qvalues (np.ndarray): The array of Q-values.
        dones (np.ndarray): The array of dones.
        mask (np.ndarray): The mask array.

        Returns:
        np.ndarray: The calculated n-step returns.
        """
        return calculate_nstep_returns(rewards, qvalues, dones, self.discount, self.n)

class OmegaReplayMemory(ReplayMemory):
    """
    OmegaReplayMemory is a subclass of ReplayMemory that calculates omega returns.

    Parameters:
    capacity (int): The maximum number of experiences to store.
    history_len (int): The length of the history for each experience.
    discount (float): The discount factor for future rewards.
    cache_size (int): The size of the cache for storing samples.
    block_size (int): The size of each block of samples.
    priority (float): The priority factor for sampling experiences.
    n (int): The number of steps for omega returns.
    """
    
    def __init__(self, capacity, history_len, discount, cache_size, block_size, priority, n):
        """
        Initializes the OmegaReplayMemory.

        Parameters:
        capacity (int): The maximum number of experiences to store.
        history_len (int): The length of the history for each experience.
        discount (float): The discount factor for future rewards.
        cache_size (int): The size of the cache for storing samples.
        block_size (int): The size of each block of samples.
        priority (float): The priority factor for sampling experiences.
        n (int): The number of steps for omega returns.
        """
        self.n = n
        super().__init__(capacity, history_len, discount, cache_size, block_size, priority)
        
    def _calculate_returns(self, rewards, qvalues, dones, mask):
        """
        Calculates omega returns.

        Parameters:
        rewards (np.ndarray): The array of rewards.
        qvalues (np.ndarray): The array of Q-values.
        dones (np.ndarray): The array of dones.
        mask (np.ndarray): The mask array.

        Returns:
        np.ndarray: The calculated omega returns.
        """
        return calculate_omega_returns(rewards, qvalues, dones, self.discount, self.n)
