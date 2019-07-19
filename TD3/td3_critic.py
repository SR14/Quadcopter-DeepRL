from keras import layers, models, optimizers
from keras import backend as K
import tensorflow as tf

class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, lr):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size        
        self.lr = lr

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        states = layers.Input(shape=(self.state_size,))
        actions = layers.Input(shape=(self.action_size,))
        stat_act = layers.Concatenate()([states, actions])

        net = layers.Dense(units=400)(stat_act)
        net = layers.Activation('relu')(net)
        net = layers.Dense(units=300)(net)
        net = layers.Activation('relu')(net)
        
        Q_values = layers.Dense(1, bias_initializer='ones')(net)
        
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)        

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=self.lr)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        gradients = K.gradients(Q_values, actions)
        self.get_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()], 
            outputs=gradients)        
