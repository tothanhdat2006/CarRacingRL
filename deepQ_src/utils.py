import matplotlib.pyplot as plt
import numpy as np

def get_state(state): 
    state = np.ascontiguousarray(state, dtype=np.float32) 
    # return np.expand_dims(state, axis=0)
    return state

def visualize_train(eps_reward, losses, configs):
    plt.figure(figsize=(14, 6))
    plt.subplot(121)
    plt.plot(eps_reward)
    plt.subplot(122)
    plt.plot(losses)
    plt.tight_layout()
    plt.show()
    