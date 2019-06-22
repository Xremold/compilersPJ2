import tvm
import logging
from tvm import autotvm
import numpy as np
import sys


def auto_schedule(func, args):
    """Automatic scheduler
    
    Args:
    -----------------
    func: function object
        similar to batch_gemm function mentioned above
    args: tuple
        inputs to func
    -----------------
    Returns:
    s: tvm.schedule.Schedule
    bufs: list of tvm.tensor.Tensor
    """
    ops, bufs = func(*args)
    #################################################
    # do some thing with `ops`, `bufs` and `args`
    # to analyze which schedule is appropriate
    
    s = tvm.create_schedule(ops)

    # GEMM!
    if len(args) == 4:

        # for i in range(0, len(s.stages)):
        #     print(type(s.stages[i]), s.stages[i])

        # # for stage 2

        # for (1, 1024, 1024, 1024): 32, 32, 8 : 3.5
        # for (2, 512, 512, 512) : 32, 32, 8 : 2.8
        # for (8, 1024, 32, 1024) 
        #    32, 8(16), 8 : 4.9
        #    32, 32, 8 : 5.6
        
        xSplitFactor = 2
        ySplitFactor = 4
        kSplitFactor = 8

        gemm_tensor = bufs[len(bufs) - 1]
        print(gemm_tensor.op.input_tensors)
        gemm_op = s[gemm_tensor]

        gemm_op = s.stages[2]

        xo, yo, xi, yi = gemm_op.tile(gemm_op.op.axis[1], gemm_op.op.axis[2], x_factor=xSplitFactor, y_factor=ySplitFactor)
        k, = gemm_op.op.reduce_axis
        ko, ki = gemm_op.split(k, factor=kSplitFactor)
        gemm_op.reorder(xo, ko, yo, xi, ki, yi)
        # com_operation2.unroll(yi)

        print(tvm.lower(s, bufs, simple_mode=True))
        return s,bufs

    
    # CONV_2D!
    else:
        # return s, bufs
        # every stage stands for a part of the computation.


        # # for stage 3
        # for (1, 1024, 7, 7, 1024, 1024, 3, 3, 0, 1, 1, 1, 1): 8,4,4,8
        # for (4, 112, 14, 14, 224, 112, 3, 3, 0, 1, 1, 2, 1): 8,4,4,8
        # for (8, 384, 27, 27, 64, 384, 1, 1, 1, 1, 0, 1, 1): 4,4,4,8
        bias_tensor = None
        conv_tensor = None
        pad_tensor = None

        conv_tensor = bufs[len(bufs) - 1]
        in_tensor2 = conv_tensor.op.input_tensors[1]
        in_tensor1 = conv_tensor.op.input_tensors[0]
        if in_tensor2.op.name == "bias":
            bias_tensor = conv_tensor
            conv_tensor = in_tensor1

        in_tensor1 = conv_tensor.op.input_tensors[0]
        pad_tensor = in_tensor1

        if bias_tensor != None:
            bias_op = s[bias_tensor]
        conv_op = s[conv_tensor]
        pad_op = s[pad_tensor]

        ocSplitFactor = 8
        xSplitFactor = 2
        ySplitFactor = 2
        icSplitFactor = 8

        print(conv_op.op.reduce_axis, conv_op.op.axis, sep="####")

        # oc:out-channel, x:image-height, y:image-width, ic;in-channel, kh:kernel-height, kw:kernel-width
        oco, oci = conv_op.split(conv_op.op.axis[1], factor=ocSplitFactor)
        xo, yo, xi, yi = conv_op.tile(conv_op.op.axis[2], conv_op.op.axis[3], x_factor=xSplitFactor, y_factor=ySplitFactor)
        
        ic, kh, kw = conv_op.op.reduce_axis
        ico, ici = conv_op.split(ic, factor=icSplitFactor)
        conv_op.reorder(oco, ico, xo, yo, oci, ici, kh, kw, xi, yi)

        if bias_tensor != None:
            xo, yo, xi, yi = bias_op.tile(bias_op.op.axis[2], bias_op.op.axis[3], x_factor=8, y_factor=8)
            bias_op.reorder(xo, yo, xi, yi)
        
        pad_op.compute_inline()

        print(tvm.lower(s, bufs, simple_mode=True))
        return s, bufs