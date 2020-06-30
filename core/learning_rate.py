

class StepDecayLearningRate:
    def __init__(self, learning_rate, decay_step, decay_rate, data_size, batch_size):
        self._lr = learning_rate
        self._iteration = data_size // batch_size
        self._decay_step = decay_step
        self._decay_rate = decay_rate
        self._idx = 0

    def __call__(self):
        self._idx += 1
        if self._idx / self._iteration > self._decay_step:
            self._lr *= self._decay_rate
            self._idx = 1
        return self._lr

    def __float__(self):
        return self._lr
