"""Recurrent inference engine for nanostate.

Runs a trained S4D model one token at a time using the recurrent view.
No FFT, no sequence dimension. Just a fixed state vector per layer.

This is the SSM advantage: constant memory, constant cost per token.

Usage:
    from engine import load_model, RecurrentState

    model, config = load_model("checkpoints/lm")
    state = RecurrentState(model)
    logits = state.step(65)  # feed one byte, get logits back
"""

import json
import os

import mlx.core as mx

import train as train_module


def load_model(checkpoint_dir):
    """Load a trained model from checkpoint.

    Returns (model, config) where config is the dict from config.json.
    """
    with open(os.path.join(checkpoint_dir, "config.json")) as f:
        config = json.load(f)

    # Set module globals so model classes use correct dimensions
    train_module.D_MODEL = config["d_model"]
    train_module.N_LAYERS = config["n_layers"]
    train_module.STATE_DIM = config["state_dim"]
    train_module.MLP_RATIO = config.get("mlp_ratio", 2)

    model = train_module.NanoSSM(config["task"])
    model.load_weights(os.path.join(checkpoint_dir, "model.npz"))
    mx.eval(model.parameters())
    return model, config


class RecurrentState:
    """Hidden state for step-by-step recurrent inference.

    Pre-computes the discretized SSM parameters (A_d, B_d) once at init,
    then runs the recurrent update: state = A_d * state + B_d * input
    for each token.

    Supports both the old SSMBlock (SSM+MLP with norm1/norm2) and the
    Mamba-style gated block (norm → in_proj → SSM + gate → out_proj).
    """

    def __init__(self, model):
        self.model = model

        # Pre-compute discrete parameters and init zero state per layer
        self._layers = []
        self.states = []
        for block in model.blocks:
            ssm = block.ssm
            dt = mx.exp(ssm.log_dt)  # (d_inner,) or (d_model,)
            A = -mx.exp(ssm.log_A)  # (d_inner, N) or (d_model, N)
            A_d = mx.exp(A * dt[:, None])  # discrete A
            B_d = ssm.B * dt[:, None]  # discrete B
            self._layers.append((A_d, B_d, ssm.C, ssm.D))
            self.states.append(mx.zeros_like(A_d))

        mx.eval(self.states)

        # Detect block type: gated (Mamba-style) vs classic (SSM+MLP)
        self._gated = hasattr(model.blocks[0], 'in_proj')

    def _silu(self, x):
        return x * mx.sigmoid(x)

    def step(self, token):
        """Feed one token, return logits.

        token: int (byte value for LM, nucleotide index for DNA)
        returns: mx.array of shape (vocab_size,)
        """
        x = self.model.embed(mx.array([token]))[0]  # (d_model,)

        for i, block in enumerate(self.model.blocks):
            A_d, B_d, C, D = self._layers[i]

            if self._gated:
                # Mamba-style gated block: norm → in_proj → SSM + gate → out_proj
                residual = x
                x_norm = block.norm(x)
                xz = block.in_proj(x_norm)
                d_inner = xz.shape[0] // 2
                x_ssm, z = xz[:d_inner], xz[d_inner:]

                # Recurrent SSM step on expanded dimension
                self.states[i] = A_d * self.states[i] + B_d * x_ssm[:, None]
                ssm_out = mx.sum(C * self.states[i], axis=1) + D * x_ssm

                # Gating and project back
                y = ssm_out * self._silu(z)
                x = residual + block.out_proj(y)
            else:
                # Classic SSMBlock: SSM → norm1 → MLP → norm2
                self.states[i] = A_d * self.states[i] + B_d * x[:, None]
                ssm_out = mx.sum(C * self.states[i], axis=1) + D * x
                x = block.norm1(x + ssm_out)
                x = block.norm2(x + block.mlp(x))

        x = self.model.norm(x)
        logits = self.model.head(x)
        mx.eval(logits, *self.states)
        return logits

    def reset(self):
        """Reset hidden state to zeros."""
        for i in range(len(self.states)):
            self.states[i] = mx.zeros_like(self.states[i])
        mx.eval(self.states)
