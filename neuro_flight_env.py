class JSBSimWrapper:
    def __init__(self):
        self.state_dim = 12 # Coord, Vel, Accel, G-load, Relative Target
        self.action_dim = 4 # Pitch, Roll, Yaw, Throttle
        print("[JSBSim] Flight Dynamics Engine Initialized.")
        
    def reset(self):
        print("[JSBSim] Resetting environment to runway...")
        return [0.0] * self.state_dim
        
    def step(self, action):
        import random
        # Dummy step simulating physics
        next_state = [random.random() for _ in range(self.state_dim)]
        reward = 1.0
        done = False
        surprisal = random.random() # Simulated cognitive surprisal from environment mismatch
        return next_state, reward, done, surprisal
