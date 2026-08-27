from __future__ import annotations

from openai import AsyncOpenAI


class LoadBalancedClient:
    """Weighted round-robin over AsyncOpenAI clients serving the same model.
    """

    def __init__(self, clients: list[AsyncOpenAI], weights: list[int]) -> None:
        if len(clients) != len(weights):
            raise ValueError("clients and weights must have the same length")
        schedule: list[AsyncOpenAI] = []
        # Interleave by weight (e.g. weights [2,1] -> [a, b, a]) so bursts
        # don't land on a single replica.
        counters = [0] * len(clients)
        total = sum(max(1, w) for w in weights)
        for step in range(total):
            best = max(
                range(len(clients)),
                key=lambda i: max(1, weights[i]) / (counters[i] + 1),
            )
            counters[best] += 1
            schedule.append(clients[best])
        self._schedule = schedule
        self._i = 0

    @property
    def chat(self):
        client = self._schedule[self._i % len(self._schedule)]
        self._i += 1
        return client.chat
