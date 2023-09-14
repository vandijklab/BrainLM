from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class BrainLMConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BrainLM`]. It is used to instantiate an BrainLM
    model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        decoder_num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the decoder.
        decoder_hidden_size (`int`, *optional*, defaults to 512):
            Dimensionality of the decoder.
        decoder_num_hidden_layers (`int`, *optional*, defaults to 8):
            Number of hidden layers in the decoder.
        decoder_intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the decoder.
        mask_ratio (`float`, *optional*, defaults to 0.75):
            The ratio of the number of masked tokens in the input sequence.
        norm_pix_loss (`bool`, *optional*, defaults to `False`):
            Whether or not to train with normalized pixels (see Table 3 in the paper). Using normalized pixels improved
            representation quality in the experiments of the authors.

    Example:

    ```python
    >>> from transformers import ViTMAEConfig, ViTMAEModel

    >>> # Initializing a ViT MAE vit-mae-base style configuration
    >>> configuration = ViTMAEConfig()

    >>> # Initializing a model (with random weights) from the vit-mae-base style configuration
    >>> model = ViTMAEModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "brainlm_mae"

    def __init__(
        self,
        hidden_size=256,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=1024,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=224,  # Unused, kept for ViT superclass constructors so they don't throw errors
        patch_size=16,  # Unused
        num_channels=3,  # Unused
        qkv_bias=True,
        decoder_num_attention_heads=8,
        decoder_hidden_size=256,
        decoder_num_hidden_layers=4,
        decoder_intermediate_size=1024,
        num_brain_voxels=424,
        num_timepoints_per_voxel=490,
        mask_ratio=0.5,
        timepoint_patching_size=49,
        use_tanh_decoder=False,
        norm_pix_loss=False,  # Unused
        loss_fn="mse",
        segment_means_seq_len=64,
        num_landmarks=64,
        conv_kernel_size=65,
        inv_coeff_init_option=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.decoder_num_attention_heads = decoder_num_attention_heads
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_num_hidden_layers = decoder_num_hidden_layers
        self.decoder_intermediate_size = decoder_intermediate_size
        self.num_brain_voxels = num_brain_voxels
        self.num_timepoints_per_voxel = num_timepoints_per_voxel
        self.mask_ratio = mask_ratio
        self.timepoint_patching_size = timepoint_patching_size
        self.use_tanh_decoder = use_tanh_decoder
        self.norm_pix_loss = norm_pix_loss
        self.loss_fn = loss_fn
        self.segment_means_seq_len = segment_means_seq_len
        self.num_landmarks = num_landmarks
        self.conv_kernel_size = conv_kernel_size
        self.inv_coeff_init_option = inv_coeff_init_option
