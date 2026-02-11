#!/usr/bin/env python3
"""
Patch the RVM ONNX model to bake downsample_ratio=0.25 as a constant.

TensorRT's ONNX parser cannot handle the dynamic downsample_ratio input
that feeds into Resize nodes (shape tensor type error). This script
replaces the downsample_ratio input with a constant, making all shapes
statically determinable at engine build time.
"""

import sys
import numpy as np
import onnx
import onnx_graphsurgeon as gs

def patch_model(input_path: str, output_path: str, ratio: float = 0.25):
    graph = gs.import_onnx(onnx.load(input_path))

    # Find the downsample_ratio input tensor
    ds_input = None
    for inp in graph.inputs:
        if inp.name == "downsample_ratio":
            ds_input = inp
            break

    if ds_input is None:
        print("ERROR: 'downsample_ratio' input not found in model")
        sys.exit(1)

    # Create a constant tensor with the baked ratio value
    ds_const = gs.Constant(
        name="downsample_ratio_const",
        values=np.array([ratio], dtype=np.float32)
    )

    # Replace all references to the input with our constant
    for node in graph.nodes:
        for i, node_input in enumerate(node.inputs):
            if node_input.name == "downsample_ratio":
                node.inputs[i] = ds_const

    # Remove downsample_ratio from graph inputs
    graph.inputs = [inp for inp in graph.inputs if inp.name != "downsample_ratio"]

    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), output_path)
    print(f"Patched model saved to {output_path}")
    print(f"  downsample_ratio baked as constant = {ratio}")
    print(f"  Inputs: {[inp.name for inp in graph.inputs]}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.onnx> <output.onnx> [ratio=0.25]")
        sys.exit(1)

    ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
    patch_model(sys.argv[1], sys.argv[2], ratio)
