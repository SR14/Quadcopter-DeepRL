import numpy as np
import copy

class OUNoise():
    """Ornstein-Uhlenbeck process."""

    def __init__(self, action_size, theta, sigma, mu=0.45, t_step=0.02):
        """Initialize parameters and noise process."""
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.t_step = t_step
        self.a_sz = action_size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.dt = 0

    def sample(self, action, batch=1, target=False):
        """Update internal state and return it as a noise sample."""            
        reversion = self.theta * (self.mu - action) * self.dt
        noise = self.sigma * np.random.normal(scale=np.sqrt(self.dt), size=(batch, self.a_sz))
        dx = reversion + noise
        if target:
            return np.clip(dx, -0.5, 0.5)
        self.dt = self.t_step        
        return dx