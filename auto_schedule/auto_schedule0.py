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
    com_tensor = bufs[len(bufs) - 1]

    # GEMM!
    if len(com_tensor.op.axis) == 3:

        # for i in range(0, len(s.stages)):
        #     print(type(s.stages[i]), s.stages[i])

        # # for stage 2

        # for (1, 1024, 1024, 1024): 32, 32, 8 : 3.5
        # for (2, 512, 512, 512) : 32, 32, 8 : 2.8
        # for (8, 1024, 32, 1024) 
        #    32, 8(16), 8 : 4.9
        #    32, 32, 8 : 5.6
        
        xSplitFactor = 32
        ySplitFactor = 32
        kSplitFactor = 8

        com_operation2 = s.stages[2]

        xo, yo, xi, yi = com_operation2.tile(com_operation2.op.axis[1], com_operation2.op.axis[2], x_factor=xSplitFactor, y_factor=ySplitFactor)
        k, = com_operation2.op.reduce_axis
        ko, ki = com_operation2.split(k, factor=kSplitFactor)
        com_operation2.reorder(xo, ko, yo, xi, ki, yi)
        # com_operation2.unroll(yi)

        # CC = s.cache_write(com_tensor, 'global')
        # xo, yo, xi, yi = s[com_tensor].tile(com_tensor.op.axis[1], com_tensor.op.axis[2], x_factor=xSplitFactor, y_factor=ySplitFactor)

        # s[CC].compute_at(s[com_tensor], yo)
        # # print(s[CC].op.axis)
        # _, xc, yc = s[CC].op.axis

        # k, = s[CC].op.reduce_axis
        # ko, ki = s[CC].split(k, factor=4)
        # s[CC].reorder(ko, xc, ki, yc)
        # s[CC].unroll(ki)

        print(tvm.lower(s, bufs, simple_mode=True))

    
    # CONV_2D!
    elif len(com_tensor.op.axis) == 4:
        # TODO
        # every stage stands for a part of the computation.


        # # for stage 3
        # for (1, 1024, 7, 7, 1024, 1024, 3, 3, 0, 1, 1, 1, 1): 8,4,4,8
        # for (4, 112, 14, 14, 224, 112, 3, 3, 0, 1, 1, 2, 1): 8,4,4,8
        # for (8, 384, 27, 27, 64, 384, 1, 1, 1, 1, 0, 1, 1): 4,4,4,8
        ocSplitFactor = 8
        xSplitFactor = 4
        ySplitFactor = 4
        icSplitFactor = 8

        com_operation = s.stages[3] # buggy
        print(com_operation.op.reduce_axis, com_operation.op.axis, sep="####")

        # oc:out-channel, x:image-height, y:image-width, ic;in-channel, kh:kernel-height, kw:kernel-width
        oco, oci = com_operation.split(com_operation.op.axis[1], factor=ocSplitFactor)
        xo, yo, xi, yi = com_operation.tile(com_operation.op.axis[2], com_operation.op.axis[3], x_factor=xSplitFactor, y_factor=ySplitFactor)
        
        ic, kh, kw = com_operation.op.reduce_axis
        ico, ici = com_operation.split(ic, factor=icSplitFactor)
        com_operation.reorder(oco, ico, xo, yo, oci, ici, xi, yi)

        # print(tvm.lower(s, bufs, simple_mode=True))
        return s, bufs

    
    #################################################
    # perform real schedule according to 
    # decisions made above, using primitives 
    # such as split, reorder, parallel, unroll...
    
    #################################################
    # finally, remember to return these two results
    # we need `bufs` to build function via `tvm.build`
    return s, bufs