# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from mmseg.models.uda.dacs import DACS
from mmseg.models.uda.attention_dacs import AttentionDACS
from mmseg.models.uda.dacs_pseudo_only_no_mix import DACSPseudo
from mmseg.models.uda.dacs_pseudo_only import DACSPseudoMix
from mmseg.models.uda.dacs_consistency import DACSConsistency
from mmseg.models.uda.dacs_cnn import DACSCNN

__all__ = ['DACS', 'AttentionDACS', 'DACSPseudo', 'DACSPseudoMix', 'DACSConsistency', 'AttentionDACS', 'DACSCNN']
