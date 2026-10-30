from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass


@configclass
class SymmetryConfig:
    joint_names: list[str] = MISSING
    symmetry_joint_names: dict[str, str] = MISSING
    sign_flip_joints: list[str] = MISSING


@configclass
class LayerConfig:
    """Configuration for neural network layer settings."""

    """
    mlp settings
    """

    hidden_dims: list[int] = MISSING
    """List of hidden layer dimensions."""
    activation: str = MISSING
    """Activation function name."""
    dropout_prob: float = MISSING
    """Dropout probability."""
    use_layer_norm: bool = MISSING
    """Whether to use layer normalization."""

    """
    mlp encoder settings
    """

    encoder_activation: str = MISSING
    """Activation function name for encoder layers."""
    encoder_output_dim: int | None = MISSING
    """Output dimension for encoder. Only used for encoder modules."""
    encoder_hidden_dims: list[int] | None = MISSING
    """Hidden dimensions for encoder. Only used for encoder modules."""

    """
    cnn encoder settings
    """

    hidden_channels: tuple[int, ...] | None = MISSING
    """Hidden channel dimensions. Only used for CNN modules."""
    kernel_size: int | tuple[int, ...] = MISSING
    """Kernel size for convolutions. Only used for CNN modules."""
    stride: int | tuple[int, ...] = MISSING
    """Stride for convolutions. Only used for CNN modules."""
    padding: str | int | tuple[str | int, ...] = MISSING
    """Padding mode for convolutions. Only used for CNN modules."""


@configclass
class ModuleConfig:
    """Configuration for neural network modules."""

    module_type: Literal["MLP", "MLPEncoder", "CNNEncoder"] = MISSING
    """Module type (e.g., MLP)."""

    obs_keys: list[str] = MISSING
    """List of observation keys."""

    layer_config: LayerConfig = MISSING
    """Feature extraction layer settings."""

    min_noise_std: float | None = MISSING
    """Minimum noise standard deviation."""

    min_mean_noise_std: float | None = MISSING
    """Minimum mean noise standard deviation."""


@configclass
class PPOModuleDictConfig:
    """Configuration for PPO module dictionary."""

    actor: ModuleConfig = MISSING
    """Actor module configuration."""

    critic: ModuleConfig = MISSING
    """Critic module configuration."""


@configclass
class PPOConfig:
    """Configuration for the PPO algorithm."""

    algorithm: str = "PPO"

    # ── Training loop ─────────────────────────────────────────────────────────
    num_learning_iterations: int = MISSING
    """Total number of training iterations (each = one rollout + update)."""
    save_interval: int = MISSING
    """Save a checkpoint every this many iterations."""
    load_optimizer: bool = MISSING
    """Whether to restore optimizer state when loading a checkpoint."""
    init_at_random_ep_len: bool = MISSING
    """Randomise the initial episode-length counter to de-correlate resets."""

    # ── Network architecture ──────────────────────────────────────────────────
    module_dict: PPOModuleDictConfig = MISSING
    """Module dictionary configuration."""
    init_noise_std: float = MISSING
    """Initial standard deviation of the actor's action distribution."""

    # ── PPO hyperparameters ───────────────────────────────────────────────────
    num_steps_per_env: int = MISSING
    """Rollout length per environment before each update."""
    num_learning_epochs: int = MISSING
    """Number of epochs over the collected rollout per update."""
    num_mini_batches: int = MISSING
    """Number of mini-batches per epoch."""
    clip_param: float = MISSING
    """PPO clipping epsilon."""
    gamma: float = MISSING
    """Discount factor."""
    lam: float = MISSING
    """GAE lambda."""
    value_loss_coef: float = MISSING
    """Weight of the value loss."""
    entropy_coef: float = MISSING
    """Entropy bonus coefficient."""
    max_grad_norm: float = MISSING
    """Gradient clipping max norm."""

    # ── Learning rate ─────────────────────────────────────────────────────────
    actor_learning_rate: float = MISSING
    actor_optimizer_weight_decay: float = MISSING
    critic_learning_rate: float = MISSING
    critic_optimizer_weight_decay: float = MISSING
    schedule: Literal["adaptive", "fixed"] = MISSING
    """LR schedule. ``"adaptive"`` adjusts based on KL, ``"fixed"`` keeps it constant."""
    desired_kl: float = MISSING
    """Target KL divergence for adaptive LR."""
    max_actor_learning_rate: float | None = MISSING
    min_actor_learning_rate: float | None = MISSING
    max_critic_learning_rate: float | None = MISSING
    min_critic_learning_rate: float | None = MISSING

    # ── Symmetry augmentation ─────────────────────────────────────────────────
    use_symmetry: bool = False
    """Whether to apply x-z plane symmetry augmentation during training."""
    symmetry_config: SymmetryConfig = MISSING
    symmetry_actor_coef: float = 0.0
    symmetry_critic_coef: float = 0.0

    # ── Logging ───────────────────────────────────────────────────────────────
    logging_interval: int = 100
    """Log training metrics every this many iterations."""
    experiment_name: str = "isaaclab"
    """Name of the experiment."""
    logger: Literal["tensorboard", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""
    wandb_project: str = "isaaclab"
    """The wandb project name. Default is "isaaclab"."""
    resume: bool = False
    """Whether to resume a previous training. Default is False.
    """
    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).
    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """
    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).
    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """


@configclass
class FastSACConfig:
    """Configuration for FastSAC algorithm."""

    algorithm: str = "FastSAC"

    # ── Training loop ─────────────────────────────────────────────────────────
    num_learning_iterations: int = MISSING
    """Total environment timesteps."""
    learning_starts: int = MISSING
    """Timestep at which gradient updates begin."""
    save_interval: int = MISSING
    """Save a checkpoint every this many iterations."""

    # ── Networks ──────────────────────────────────────────────────────────────
    module_type: Literal["MLP", "MLPEncoder", "CNNEncoder"] = "MLP"
    activation: Literal["ReLU", "ELU", "SiLU"] = "SiLU"  # NOTE: Only SilU works
    actor_hidden_dim: list[int] = MISSING
    critic_hidden_dim: list[int] = MISSING
    use_layer_norm: bool = True
    num_q_networks: int = 2
    """Number of Q-networks in the ensemble."""

    # ── Distributional critic ─────────────────────────────────────────────────
    num_atoms: int = MISSING
    v_min: float = MISSING
    v_max: float = MISSING

    # ── CNN encoder ───────────────────────────────────────────────────────────
    encoder_obs_key: str = MISSING
    encoder_obs_shape: tuple[int, int, int] = MISSING

    # ── Observation keys ──────────────────────────────────────────────────────
    actor_obs_keys: list[str] = ["policy"]
    critic_obs_keys: list[str] = ["critic"]

    # ── Replay buffer ─────────────────────────────────────────────────────────
    buffer_size: int = MISSING
    """Per-environment replay buffer capacity."""
    num_steps: int = 1
    """N-step return horizon."""
    batch_size: int = MISSING

    # ── SAC hyperparameters ───────────────────────────────────────────────────
    gamma: float = MISSING
    tau: float = MISSING
    """Target network soft-update coefficient."""
    policy_frequency: int = MISSING
    """Actor update frequency (every N critic updates)."""
    num_updates: int = MISSING
    """Gradient updates per environment step."""

    # ── Entropy temperature (alpha) ───────────────────────────────────────────
    alpha_init: float = MISSING
    use_autotune: bool = MISSING
    target_entropy_ratio: float = MISSING

    # ── Action ────────────────────────────────────────────────────────────────
    use_tanh: bool = MISSING
    log_std_max: float = MISSING
    log_std_min: float = MISSING

    # ── Optimizers ────────────────────────────────────────────────────────────
    critic_learning_rate: float = MISSING
    actor_learning_rate: float = MISSING
    alpha_learning_rate: float = MISSING
    weight_decay: float = MISSING
    max_grad_norm: float = MISSING
    """Gradient clipping norm (0 = disabled)."""

    # ── augmentation ────────────────────────────────────────────────────────
    obs_normalization: bool = MISSING
    use_symmetry: bool = MISSING
    symmetry_config: SymmetryConfig = MISSING

    # ── training settings ──────────────────────────────────────────────────────────────────
    compile: bool = False
    """Use ``torch.compile`` for the update functions."""
    amp: bool = True
    """Automatic Mixed Precision."""
    amp_dtype: Literal["bf16", "fp16"] = "bf16"
    """AMP dtype: ``"bf16"`` or ``"fp16"``."""

    # ── Logging ───────────────────────────────────────────────────────────────
    logging_interval: int = 100
    """Log training metrics every this many iterations."""
    save_rsl_rl_wrapper: bool = False
    """Whether to save actor as RSL RL wrapper."""
    experiment_name: str = "isaaclab"
    """Name of the experiment."""
    logger: Literal["tensorboard", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""
    wandb_project: str = "isaaclab"
    """The wandb project name. Default is "isaaclab"."""
    resume: bool = False
    """Whether to resume a previous training. Default is False.
    """
    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).
    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """
    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).
    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """
