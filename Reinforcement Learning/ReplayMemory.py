from collections import deque
import numpy as np


# Implementing Experience Replay's Memory
class ReplayMemory:
    def __init__(self, n_steps, capacity=5000):
        self.capacity = capacity
        self.n_steps = n_steps
        self.n_steps_iter = iter(n_steps)
        self.buffer = deque()

    # Create an iterator to return random batches
    def sample_batch(self, batch_size):
        ofs = 0
        values = list(self.buffer)
        np.random.shuffle(values)
        while (ofs + 1) * batch_size <= len(self.buffer):
            yield values[ofs * batch_size:(ofs + 1) * batch_size]
            ofs += 1

    def run_steps(self, samples):
        while samples > 0:
            entry = next(self.n_steps_iter)
            self.buffer.append(entry)
            samples -= 1
        # Don't accumulate samples more than the capacity
        while len(self.buffer) > self.capacity:
            self.buffer.popleft()
