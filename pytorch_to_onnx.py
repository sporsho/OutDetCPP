# this is not a standalone code,
# this code should be run from outdet folder from https://github.com/sporsho/3D_OutDet
import pickle
import numpy as np
import torch.nn as nn
import torch.onnx
import torch
from modules import OutDet

if __name__ == "__main__":
    pt_file = '/var/local/home/aburai/3D_OutDet/saved_models/bin_desnow_wads/outdet.pt'
    state_dict = torch.load(pt_file)
    model = OutDet(num_classes=2, kernel_size=3, depth=1, dilate=1)
    model.load_state_dict(state_dict)
    # model.fuse()
    dummy_points = torch.randn((250000, 4), requires_grad=True)
    dummy_dist = torch.randn((250000, 9), requires_grad=True)
    dummy_ind = torch.randint(low=0, high=250000, size=(250000, 9))
    model.eval()
    torch.onnx.export(
        model,
        (dummy_points, dummy_dist, dummy_ind),
        "outdet.onnx",
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["points", "dist", "indices"],
        output_names=["out"],
        dynamic_axes={"points": {0: "batch_size"},
                      "dist": {0: "batch_size"},
                      "indices": {0: "batch_size"},
                      "out": {0: "batch_size"}}
    )