"""Two-stage transfer-learning recipe shared by the notebooks (same as Phase 0)."""
from __future__ import annotations

from .backbones import set_finetune_trainable


def train_two_stage(model, base, train_ds, val_ds, head_epochs=15, head_lr=1e-3,
                    finetune_epochs=20, finetune_lr=1e-5, top_fraction=0.35, patience=4, verbose=2):
    """Stage 1: train the head with the backbone frozen. Stage 2: unfreeze the top of the
    backbone (normalisation layers stay frozen) at a low learning rate. Early stopping on
    validation loss, best weights restored. Returns the combined history dict."""
    from tensorflow import keras

    def stop():
        return keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)

    model.compile(optimizer=keras.optimizers.Adam(head_lr), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=head_epochs, callbacks=[stop()], verbose=verbose)
    set_finetune_trainable(base, top_fraction)
    model.compile(optimizer=keras.optimizers.Adam(finetune_lr), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=finetune_epochs, callbacks=[stop()], verbose=verbose)
    history = {k: h1.history[k] + h2.history.get(k, []) for k in h1.history}
    history["stage1_epochs"] = len(h1.history["loss"])
    return history
