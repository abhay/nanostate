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

    kwargs = {}
    if config.get("vocab_size"):
        kwargs["vocab_size"] = config["vocab_size"]
    if config.get("block_type"):
        kwargs["block_type"] = config["block_type"]
    model = train_module.NanoSSM(config["task"], **kwargs)
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
        self._block_type = getattr(model, "block_type", "s4d")

        if self._block_type == "ssd":
            self._init_ssd()
        else:
            self._init_s4d()

    def _init_s4d(self):
        # Pre-compute discrete parameters and init zero state per layer
        self._layers = []
        self.states = []
        for block in self.model.blocks:
            ssm = block.ssm
            dt = mx.exp(ssm.log_dt)  # (d_inner,) or (d_model,)
            A = -mx.exp(ssm.log_A)  # (d_inner, N) or (d_model, N)
            A_d = mx.exp(A * dt[:, None])  # discrete A
            B_d = ssm.B * dt[:, None]  # discrete B
            self._layers.append((A_d, B_d, ssm.C, ssm.D))
            self.states.append(mx.zeros_like(A_d))

        mx.eval(self.states)

        # Detect block type: gated (Mamba-style) vs classic (SSM+MLP)
        self._gated = hasattr(self.model.blocks[0], "in_proj")

    def _init_ssd(self):
        # SSD state: (n_heads, d_head, d_state) per layer
        self.states = []
        self._conv_bufs = []
        for block in self.model.blocks:
            ssd = block.ssd
            n_heads = ssd.n_heads
            d_head = ssd.d_head
            d_state = ssd.d_state
            self.states.append(mx.zeros((n_heads, d_head, d_state)))
            # Conv1d buffer: last (kernel_size - 1) inputs
            d_inner = block.in_proj.weight.shape[0] // 2
            self._conv_bufs.append(mx.zeros((block.conv_pad, d_inner)))
        mx.eval(self.states, self._conv_bufs)

    def _silu(self, x):
        return x * mx.sigmoid(x)

    def step(self, token):
        """Feed one token, return logits.

        token: int (byte value for LM, nucleotide index for DNA)
        returns: mx.array of shape (vocab_size,)
        """
        x = self.model.embed(mx.array([token]))[0]  # (d_model,)

        if self._block_type == "ssd":
            x = self._step_ssd(x)
        else:
            x = self._step_s4d(x)

        x = self.model.norm(x)
        logits = self.model.head(x)
        mx.eval(logits, *self.states)
        return logits

    def _step_s4d(self, x):
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
        return x

    def _step_ssd(self, x):
        for i, block in enumerate(self.model.blocks):
            ssd = block.ssd
            residual = x
            x_norm = block.norm(x)

            # Parallel projections
            xz = block.in_proj(x_norm)
            d_inner = xz.shape[0] // 2
            x_raw, z = xz[:d_inner], xz[d_inner:]

            # Causal conv1d: shift buffer and apply conv weights
            self._conv_bufs[i] = mx.concatenate([self._conv_bufs[i][1:], x_raw[None, :]], axis=0)
            # Manual conv: concat buffer + current, apply depthwise weights
            conv_input = mx.concatenate([self._conv_bufs[i], x_raw[None, :]], axis=0)  # (kernel_size, d_inner)
            # Conv1d weights: (d_inner, 1, kernel_size) in MLX
            conv_w = block.conv1d.weight[:, 0, :]  # (d_inner, kernel_size)
            x_conv = mx.sum(conv_w * conv_input.T, axis=1)  # (d_inner,)
            if hasattr(block.conv1d, "bias") and block.conv1d.bias is not None:
                x_conv = x_conv + block.conv1d.bias
            x_conv = self._silu(x_conv)

            # Input-dependent A, B, C
            a = -self._softplus(ssd.a_proj(x_conv))  # (n_heads,)
            b = ssd.b_proj(x_conv)  # (d_state,)
            c = ssd.c_proj(x_conv)  # (d_state,)

            # Reshape x for multi-head: (d_inner,) -> (n_heads, d_head)
            x_heads = x_conv.reshape(ssd.n_heads, ssd.d_head)

            # Recurrent SSD step: h = a * h + B * x, y = C^T * h
            # a is scalar per head, h is (n_heads, d_head, d_state)
            decay = mx.exp(a)  # (n_heads,)
            self.states[i] = decay[:, None, None] * self.states[i] + x_heads[:, :, None] * b[None, None, :]
            # Output: y = C^T * h + D * x
            y_heads = mx.sum(self.states[i] * c[None, None, :], axis=2)  # (n_heads, d_head)
            y = y_heads.reshape(-1) + ssd.D * x_conv  # (d_inner,)

            # Gate and project
            y = y * self._silu(z)
            x = residual + block.out_proj(y)
        return x

    def _softplus(self, x):
        return mx.log(1 + mx.exp(x))

    def reset(self):
        """Reset hidden state to zeros."""
        for i in range(len(self.states)):
            self.states[i] = mx.zeros_like(self.states[i])
        if self._block_type == "ssd":
            for i in range(len(self._conv_bufs)):
                self._conv_bufs[i] = mx.zeros_like(self._conv_bufs[i])
            mx.eval(self.states, self._conv_bufs)
        else:
            mx.eval(self.states)
