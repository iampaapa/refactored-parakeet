import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import os

# Hyperparameters
BUFFER_SIZE = 100000
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.005
LEARNING_RATE = 3e-4
ALPHA = 0.2
EPISODES = 1000
STEPS_PER_EPISODE = 96  # 24 hours with 15-minute intervals

class ReplayBuffer:
    def __init__(self, state_dim, action_dim):
        self.state_buffer = np.zeros((BUFFER_SIZE, state_dim), dtype=np.float32)
        self.action_buffer = np.zeros((BUFFER_SIZE, action_dim), dtype=np.float32)
        self.reward_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
        self.next_state_buffer = np.zeros((BUFFER_SIZE, state_dim), dtype=np.float32)
        self.done_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
        self.pointer, self.size = 0, 0

    def add(self, state, action, reward, next_state, done):
        self.state_buffer[self.pointer] = state
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.next_state_buffer[self.pointer] = next_state
        self.done_buffer[self.pointer] = done
        self.pointer = (self.pointer + 1) % BUFFER_SIZE
        self.size = min(self.size + 1, BUFFER_SIZE)

    def sample(self, batch_size):
        idxs = np.random.choice(self.size, batch_size, replace=False)
        return (
            self.state_buffer[idxs],
            self.action_buffer[idxs],
            self.reward_buffer[idxs],
            self.next_state_buffer[idxs],
            self.done_buffer[idxs]
        )

def create_actor_network(state_dim, action_dim):
    inputs = layers.Input(shape=(state_dim,))
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.Dense(256, activation='relu')(x)
    mean = layers.Dense(action_dim, activation='tanh')(x)
    log_std = layers.Dense(action_dim, activation='tanh')(x)
    return tf.keras.Model(inputs, [mean, log_std])

def create_critic_network(state_dim, action_dim):
    state_input = layers.Input(shape=(state_dim,))
    action_input = layers.Input(shape=(action_dim,))
    x = layers.Concatenate()([state_input, action_input])
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    output = layers.Dense(1)(x)
    return tf.keras.Model([state_input, action_input], output)

class SolarPanelSAC:
    def __init__(self, state_dim, action_dim):
        self.actor = create_actor_network(state_dim, action_dim)
        self.critic_1 = create_critic_network(state_dim, action_dim)
        self.critic_2 = create_critic_network(state_dim, action_dim)
        self.target_critic_1 = create_critic_network(state_dim, action_dim)
        self.target_critic_2 = create_critic_network(state_dim, action_dim)
        
        self.target_critic_1.set_weights(self.critic_1.get_weights())
        self.target_critic_2.set_weights(self.critic_2.get_weights())
        
        self.actor_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
        self.critic_1_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
        self.critic_2_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
        
        self.log_alpha = tf.Variable(0.0)
        self.alpha_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
        
        self.target_entropy = -tf.constant(action_dim, dtype=tf.float32)
        
    def get_action(self, state):
        mean, log_std = self.actor(state)
        std = tf.exp(log_std)
        normal = tfp.distributions.Normal(mean, std)
        action = normal.sample()
        action = tf.tanh(action)
        log_prob = normal.log_prob(action) - tf.math.log(1 - tf.square(action) + 1e-6)
        log_prob = tf.reduce_sum(log_prob, axis=1, keepdims=True)
        return action, log_prob
    
    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        alpha = tf.exp(self.log_alpha)
        
        with tf.GradientTape(persistent=True) as tape:
            next_action, next_log_prob = self.get_action(next_state_batch)
            
            target_q1 = self.target_critic_1([next_state_batch, next_action])
            target_q2 = self.target_critic_2([next_state_batch, next_action])
            target_q = tf.minimum(target_q1, target_q2) - alpha * next_log_prob
            target_q = reward_batch + GAMMA * (1 - done_batch) * target_q
            
            current_q1 = self.critic_1([state_batch, action_batch])
            current_q2 = self.critic_2([state_batch, action_batch])
            
            critic_loss_1 = tf.reduce_mean(tf.square(current_q1 - target_q))
            critic_loss_2 = tf.reduce_mean(tf.square(current_q2 - target_q))
            
            new_action, log_prob = self.get_action(state_batch)
            q1 = self.critic_1([state_batch, new_action])
            q2 = self.critic_2([state_batch, new_action])
            q = tf.minimum(q1, q2)
            
            actor_loss = tf.reduce_mean(alpha * log_prob - q)
            
            alpha_loss = -tf.reduce_mean(self.log_alpha * tf.stop_gradient(log_prob + self.target_entropy))
        
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        critic_1_grads = tape.gradient(critic_loss_1, self.critic_1.trainable_variables)
        critic_2_grads = tape.gradient(critic_loss_2, self.critic_2.trainable_variables)
        alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
        
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.critic_1_optimizer.apply_gradients(zip(critic_1_grads, self.critic_1.trainable_variables))
        self.critic_2_optimizer.apply_gradients(zip(critic_2_grads, self.critic_2.trainable_variables))
        self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))
        
        self.update_target_networks()
        
    def update_target_networks(self):
        for (a, b) in zip(self.target_critic_1.variables, self.critic_1.variables):
            a.assign(TAU * b + (1 - TAU) * a)
        for (a, b) in zip(self.target_critic_2.variables, self.critic_2.variables):
            a.assign(TAU * b + (1 - TAU) * a)
    
    def save_model(self, path):
        self.actor.save(os.path.join(path, 'actor.h5'))
        self.critic_1.save(os.path.join(path, 'critic_1.h5'))
        self.critic_2.save(os.path.join(path, 'critic_2.h5'))

class SolarPanelEnv:
    def __init__(self):
        self.state_dim = 9  # 4 LDR values + voltage + current + power + 2 servo angles
        self.action_dim = 2  # 2 servo angles
        self.time = 0
        
    def reset(self):
        self.time = 0
        return np.random.rand(self.state_dim)
    
    def step(self, action):
        # Simulate panel movement and new readings
        next_state = np.random.rand(self.state_dim)
        next_state[-2:] = (action + 1) * 90  # Convert [-1, 1] to [0, 180]
        reward = next_state[6]  # Use power as reward
        self.time += 1
        done = self.time >= STEPS_PER_EPISODE
        return next_state, reward, done, {}

def train_sac():
    env = SolarPanelEnv()
    agent = SolarPanelSAC(env.state_dim, env.action_dim)
    replay_buffer = ReplayBuffer(env.state_dim, env.action_dim)

    episode_rewards = []
    for episode in range(EPISODES):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = agent.get_action(tf.expand_dims(state, 0))
            action = action.numpy()[0]
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, reward, next_state, float(done))
            episode_reward += reward
            state = next_state
            
            if replay_buffer.size >= BATCH_SIZE:
                batch = replay_buffer.sample(BATCH_SIZE)
                agent.update(*batch)
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode}, Reward: {episode_reward}")
    
    # Save the trained model
    agent.save_model('trained_model')
    
    # Plot performance
    plt.plot(episode_rewards)
    plt.title('SAC Performance')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig('performance.png')
    plt.close()

    return agent

if __name__ == "__main__":
    trained_agent = train_sac()
