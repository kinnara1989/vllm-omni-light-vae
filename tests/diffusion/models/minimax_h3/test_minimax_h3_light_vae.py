# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
    DistributedVaeMixin,
)
from vllm_omni.diffusion.models.minimax_h3.vae import (
    MiniMaxH3LightVideoVAE,
    _light_vae_opt_enabled,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, True),
        ("1", True),
        ("true", True),
        ("yes", True),
        ("0", False),
        ("off", False),
        ("false", False),
        ("no", False),
    ],
)
def test_light_vae_opt_switch(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("MINIMAX_H3_LIGHT_VAE_OPT", raising=False)
    else:
        monkeypatch.setenv("MINIMAX_H3_LIGHT_VAE_OPT", value)
    assert _light_vae_opt_enabled() is expected


def test_light_vae_keeps_decode_contract():
    assert issubclass(MiniMaxH3LightVideoVAE, DistributedVaeMixin)
    assert callable(getattr(MiniMaxH3LightVideoVAE, "decode_latent", None))
    assert callable(getattr(MiniMaxH3LightVideoVAE, "load_to_device", None))
    assert callable(getattr(MiniMaxH3LightVideoVAE, "offload_to_cpu", None))
