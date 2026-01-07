class PersistenceEMA:
    def __init__(self, beta=0.9):
        self.beta = beta
        self.state = {}

    def update(self, key, signal_strength: float):
        prev = self.state.get(key, 0.0)
        val = self.beta * prev + (1.0 - self.beta) * float(signal_strength)
        self.state[key] = val
        return val

    def get(self, key):
        return self.state.get(key, 0.0)
