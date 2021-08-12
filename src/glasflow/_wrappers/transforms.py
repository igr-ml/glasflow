# -*- coding: utf-8 -*-
"""
Wrappers for `nflows.transforms`
"""
import nflows
import torch
import wrapt


@wrapt.patch_function_wrapper(nflows.transforms.lu.LULinear, "weight_inverse")
def _weight_inverse(wrapped, instance, args, kwargs):
    lower, upper = instance._create_lower_upper()
    identity = torch.eye(
        instance.features,
        instance.features,
        device=instance.lower_entries.device,
    )
    lower_inverse, _ = torch.triangular_solve(
        identity, lower, upper=False, unitriangular=True
    )
    weight_inverse, _ = torch.triangular_solve(
        lower_inverse, upper, upper=True, unitriangular=False
    )
    return weight_inverse
