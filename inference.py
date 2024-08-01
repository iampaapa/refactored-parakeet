import numpy as np
import tensorflow as tf
import os
import logging
from datetime import datetime

class SolarPanelSAC:
    def __init__(self, model_path):
        self.actor = tf.keras.models.load_model(os.path.join(model_path, 'actor.h5'))

    def get_action(self, state):
        mean, log_std = self.actor(state)
        action = tf.tanh(mean)
        return action.numpy()[0]

def setup_logger():
    logger = logging.getLogger('SolarPanelSAC')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('solar_panel_sac.log')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

logger = setup_logger()

def inference(state):
    try:
        model_path = 'trained_model'
        agent = SolarPanelSAC(model_path)
        
        # Ensure state is in the correct format (9-dimensional numpy array)
        state = np.array(state, dtype=np.float32)
        if state.shape != (9,):
            raise ValueError("State should be a 9-dimensional array")
        
        # Get action from the model
        action = agent.get_action(tf.expand_dims(state, 0))
        
        # Convert action from [-1, 1] to [0, 180] for servo angles
        new_angles = (action + 1) * 90
        
        # Adjust for the servo angle corruption
        new_angles[0] = min(new_angles[0] + 100, 180)  # Add 100 to first angle, cap at 180
        
        logger.info(f"[DECISION] New angles calculated: {new_angles}")
        return new_angles.tolist()
    except Exception as e:
        logger.error(f"[ERROR] Error in inference: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    current_state = [0.5, 0.6, 0.4, 0.5, 12.0, 1.5, 18.0, 45.0, 90.0]
    new_angles = inference(current_state)
    print(f"New servo angles: {new_angles}")