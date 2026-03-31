"""
Realtime copy of TensorRT helpers used by inference (no dependency on shuttle_tracking).
"""

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torch
import numpy as np


def load_engine(engine_file_path):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_file_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    return engine, context


def allocate_buffers(engine, batch_size):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for idx, binding in enumerate(engine):
        size = abs(trt.volume(engine.get_tensor_shape(binding)) * batch_size)
        dtype = trt.nptype(engine.get_tensor_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if idx == 0:
            inputs.append({"host": host_mem, "device": device_mem})
        else:
            outputs.append({"host": host_mem, "device": device_mem})
    return inputs, outputs, bindings, stream


def do_inference(engine, context, bindings, inputs, outputs, stream, shape):
    for inp in inputs:
        host_data = inp["host"].numpy()
        cuda.memcpy_htod_async(inp["device"], host_data, stream)
    context.set_input_shape(engine.get_tensor_name(0), shape)
    context.set_tensor_address(engine.get_tensor_name(0), bindings[0])
    context.set_tensor_address(engine.get_tensor_name(1), bindings[1])
    context.execute_async_v3(stream_handle=stream.handle)
    stream.synchronize()
    for out in outputs:
        cuda.memcpy_dtoh_async(out["host"], out["device"], stream)
    stream.synchronize()
    output_data = outputs[0]["host"].reshape(engine.get_tensor_shape(engine.get_tensor_name(1)))
    return torch.tensor(output_data)
