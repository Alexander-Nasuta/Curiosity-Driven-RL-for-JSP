import random

import numpy as np


class Memory:

    def __init__(self, capacity, elem_shape, replacement_strategy: str = "random"):
        if replacement_strategy not in ["random", "fifo"]:
            raise ValueError(f"replacement_strategy={replacement_strategy} is an invalid argument. "
                             f"valid options are 'random' and 'fifo'")
        self.replacement_strategy = replacement_strategy
        self._capacity = capacity
        self._memory = np.zeros(shape=(capacity, *elem_shape))
        self._size = 0

    def add(self, value):
        _len = len(self)
        if _len < self._capacity:
            # fill memory if space is available
            index = self._size
        elif self.replacement_strategy == 'fifo':
            index = self._size % self._capacity
        else:
            index = random.randint(0, _len - 1)  # both values are includes, therefore len -1

        # print(f"strat: {self.replacement_strategy}, {index=}, len={len(self)}, size={self._size}")
        self._memory[index] = value
        self._size += 1

    def get_all_entries(self):
        return self._memory[:len(self)]

    def __len__(self):
        return min(self._size, len(self._memory))


class EpisodicMemory:

    def is_not_empty(self) -> bool:
        return sum([len(sm) for sm in self.ec_sub_memory]) > 0

    def len_of_sub_memory(self, env_index: int):
        return len(self.ec_sub_memory[env_index])

    def __init__(self, capacity, n_envs, obs_shape, replacement_strategy='random'):
        self._capacity = capacity
        self._obs_shape = obs_shape
        self._replacement_strategy = replacement_strategy
        self._n_envs = n_envs
        self.ec_sub_memory = [
            Memory(
                capacity=self._capacity,
                elem_shape=self._obs_shape,
                replacement_strategy=self._replacement_strategy
            ) for _ in range(self._n_envs)]

    def add_all(self, observations: np.ndarray) -> None:
        n_envs, *_ = observations.shape
        print(observations.shape)
        assert n_envs == self._n_envs
        for i, obs in enumerate(observations):
            self.ec_sub_memory[i].add(obs)

    def get_all_entries(self) -> np.ndarray:
        return np.array([sub.get_all_entries() for sub in self.ec_sub_memory], dtype=object)

    def add_to_env_ec_memory(self, env_index: int, observation: np.ndarray) -> None:
        self.ec_sub_memory[env_index].add(observation)

    def reset_ec_memory(self, env_index: int) -> None:
        self.ec_sub_memory[env_index] = Memory(
            capacity=self._capacity,
            elem_shape=self._obs_shape,
            replacement_strategy=self._replacement_strategy
        )

    def reset_all(self) -> None:
        self.ec_sub_memory = [Memory(
            capacity=self._capacity,
            elem_shape=self._obs_shape,
            replacement_strategy=self._replacement_strategy
        ) for _ in range(self._n_envs)]
