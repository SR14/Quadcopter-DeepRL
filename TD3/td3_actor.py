from keras import layers, models
from keras import backend as K
import tensorflow as tf

class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, lr):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        
        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')
                 
        net = layers.Dense(400)(states)
        net = layers.Activation('relu')(net)
        net = layers.Dense(300)(net)
        net = layers.Activation('relu')(net)

        actions = layers.Dense(self.action_size)(net)
        actions = layers.Activation('sigmoid')(actions)
                    
        self.model = models.Model(inputs=states, outputs=actions)
        
        # training function
        action_grads = layers.Input(shape=(self.action_size,))
        actor_weights = self.model.trainable_weights
        actor_grads = tf.gradients(actions, actor_weights, -action_grads)
        grads = zip(actor_grads, actor_weights)
        updates_op = tf.train.AdamOptimizer(self.lr).apply_gradients(grads)
        
        self.train_fn = K.function(
            inputs=[self.model.input, action_grads, K.learning_phase()],
            outputs=[],
            updates=[updates_op])

        

