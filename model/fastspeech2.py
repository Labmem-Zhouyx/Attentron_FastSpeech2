import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet, MelEncoder
from .modules import VarianceAdaptor
from .reference import ReferenceEncoder, ReferenceAttention
from utils.tools import get_mask_from_lengths
from text.symbols import symbols

class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        # Coarse-grained Encoder for Reference Mel
        self.coarse_grained_encoder = ReferenceEncoder(
            conv_channels=model_config["reference_encoder"]["conv_filters"],
            conv_kernel_size=model_config["reference_encoder"]["conv_kernel_size"],
            conv_stride=model_config["reference_encoder"]["conv_stride"],
            conv_padding=model_config["reference_encoder"]["conv_padding"],
            lstm_layer_num=model_config["reference_encoder"]["lstm_layer_num"],
            lstm_hidden_size=model_config["reference_encoder"]["lstm_hidden_size"],
        )
        self.coarse_linear = nn.Linear(model_config["reference_encoder"]["lstm_hidden_size"], model_config["transformer"]["encoder_hidden"])

        # Fine-grained Encoder for Reference Mel
        self.fine_grained_encoder = ReferenceEncoder(
            conv_channels=model_config["reference_encoder"]["conv_filters"],
            conv_kernel_size=model_config["reference_encoder"]["conv_kernel_size"],
            conv_stride=model_config["reference_encoder"]["conv_stride"],
            conv_padding=model_config["reference_encoder"]["conv_padding"],
            lstm_layer_num=model_config["reference_encoder"]["lstm_layer_num"],
            lstm_hidden_size=model_config["reference_encoder"]["lstm_hidden_size"],
        )
        self.fclayer = nn.Linear(preprocess_config["preprocessing"]["mel"]["n_mel_channels"], model_config["transformer"]["decoder_hidden"])

        # Reference Attention
        self.ref_atten = ReferenceAttention(
            query_dim=model_config["transformer"]["decoder_hidden"],
            key_dim=model_config["reference_encoder"]["lstm_hidden_size"],
            ref_attention_dim=model_config["reference_attention"]["attention_dim"], 
            ref_attention_dropout=model_config["reference_attention"]["attention_dropout"],
        )

    def forward(
        self,
        texts,
        src_lens,
        max_src_len,
        speakers=None,
        ref_coarse_mels=None,
        ref_coarse_mel_lens=None,
        ref_fine_mels=None,
        ref_fine_mel_lens=None,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        output = self.encoder(texts, src_masks)

        ref_coarse_emb = self.coarse_grained_encoder(ref_coarse_mels)
        ref_coarse_mel_masks = (1 - get_mask_from_lengths(ref_coarse_mel_lens, max(ref_coarse_mel_lens)).float()).unsqueeze(-1).expand(-1, -1, 512)
        coarse_grained_emb = torch.sum(ref_coarse_emb * ref_coarse_mel_masks, axis=1) / ref_coarse_mel_lens.unsqueeze(-1).expand(-1, 512)
        coarse_grained_emb = self.coarse_linear(coarse_grained_emb)

        output = output + coarse_grained_emb.unsqueeze(1).expand(-1, max_src_len, -1)

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, mel_masks)

        ref_fine_emb = self.fine_grained_encoder(ref_fine_mels)
        ref_style_emb = self.fclayer(ref_fine_mels)
        fine_graind_emb, ref_alignments = self.ref_atten(
            output, mel_lens, ref_fine_emb, ref_style_emb, ref_fine_mels, ref_fine_mel_lens
        )
        output = output + fine_graind_emb

        output = self.mel_linear(output)
        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            ref_alignments
        )
