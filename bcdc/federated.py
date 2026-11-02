"""Federated learning simulation: several "hospitals" train one model without sharing images.

* **FedAvg** (McMahan et al., 2017): each round, every client starts from the global model,
  trains locally, and the server averages the client models weighted by their data size.
* **FedProx** (Li et al., 2020): FedAvg plus a proximal term (mu/2)*||w - w_global||^2 in each
  client's loss, which limits client drift when hospitals have different case mixes (non-IID).

This is a single-machine simulation to measure the accuracy cost of federation. It is not
a privacy guarantee on its own: model updates can leak information. Real deployments add
secure aggregation and/or differential privacy, and run on a framework such as Flower.
"""
from __future__ import annotations

import numpy as np


def iid_partition(labels, n_clients, seed=0):
    idx = np.random.default_rng(seed).permutation(len(labels))
    return [np.sort(part) for part in np.array_split(idx, n_clients)]


def dirichlet_partition(labels, n_clients, alpha=0.5, seed=0, min_size=10, max_tries=100):
    """Label-skewed split: each class is divided among clients with proportions drawn from
    Dirichlet(alpha). Smaller alpha = more different case mixes between hospitals."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        parts = [[] for _ in range(n_clients)]
        for c in np.unique(labels):
            idx = rng.permutation(np.where(labels == c)[0])
            props = rng.dirichlet(np.full(n_clients, alpha))
            cuts = (np.cumsum(props)[:-1] * len(idx)).astype(int)
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return [np.sort(np.array(p, dtype=int)) for p in parts]
    raise RuntimeError("Could not make a partition with min_size images per client; raise alpha or lower min_size")


class FederatedTrainer:
    """Runs FedAvg / FedProx on a Keras model whose trainable layers are already set.

    One Adam optimiser is reused and reset at the start of every client update, so the
    training step is compiled once.
    """

    def __init__(self, model, lr=1e-4, mu=0.0):
        import tensorflow as tf
        from tensorflow import keras

        self.model, self.mu = model, float(mu)
        self.opt = keras.optimizers.Adam(lr)
        self.opt.build(model.trainable_variables)
        # Snapshot of the fresh optimiser state. It includes the learning rate (a variable in
        # Keras 3), so resetting must restore these values rather than zero everything.
        self._opt_init = [np.array(v.numpy()) for v in self.opt.variables]
        self.global_vars = [tf.Variable(tf.convert_to_tensor(v), trainable=False) for v in model.trainable_variables]
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy()
        self.rounds_done = 0

        @tf.function
        def step(x, y):
            with tf.GradientTape() as tape:
                loss = self.loss_fn(y, self.model(x, training=True))
                if self.mu > 0:
                    prox = tf.add_n([tf.reduce_sum(tf.square(w - g))
                                     for w, g in zip(self.model.trainable_variables, self.global_vars)])
                    loss = loss + 0.5 * self.mu * prox
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
            return loss

        self._step = step

    @property
    def n_trainable_params(self):
        return int(sum(np.prod(v.shape) for v in self.model.trainable_variables))

    def _reset_optimizer(self):
        for v, init in zip(self.opt.variables, self._opt_init):
            v.assign(init)

    def local_update(self, ds, epochs=1):
        self._reset_optimizer()
        for _ in range(epochs):
            for x, y in ds:
                self._step(x, y)
        return self.model.get_weights()

    def run_round(self, client_datasets, client_sizes, local_epochs=1):
        """One FedAvg/FedProx round. The model holds the new global weights afterwards."""
        import tensorflow as tf

        global_w = self.model.get_weights()
        for g, v in zip(self.global_vars, self.model.trainable_variables):
            g.assign(tf.convert_to_tensor(v))
        sizes = np.asarray(client_sizes, dtype=np.float64)
        avg = None
        for ds, n in zip(client_datasets, sizes):
            self.model.set_weights(global_w)
            w = self.local_update(ds, local_epochs)
            contrib = [wi.astype(np.float64) * (n / sizes.sum()) for wi in w]
            avg = contrib if avg is None else [a + c for a, c in zip(avg, contrib)]
        self.model.set_weights([a.astype(g.dtype) for a, g in zip(avg, global_w)])
        self.rounds_done += 1

    def communication_mb(self, n_clients, rounds):
        """Upload + download of the trainable weights (float32), all clients, all rounds."""
        return 2 * n_clients * rounds * self.n_trainable_params * 4 / 1e6
