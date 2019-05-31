import tvm


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
        # TODO
        print(tvm.lower(s, bufs, simple_mode=True))
        xo, yo, xi, yi = s[com_tensor].tile(com_tensor.op.axis[1], com_tensor.op.axis[2], x_factor=32, y_factor=32)
        k, = s[com_tensor].op.reduce_axis
        ko, ki = s[com_tensor].split(k, factor=8)
        s[com_tensor].reorder(xo, ko, yo, xi, ki, yi)
        # s[com_tensor].vectorize(yi)
        # print(tvm.lower(s, bufs, simple_mode=True))

    
    # CONV_2D!
    elif len(com_tensor.op.axis) == 4:
        # TODO
        # every stage stands for a part of the computation.

        com_operation = s.stages[3]
        print(com_operation.op.reduce_axis, com_operation.op.axis, sep="####")

        # oc:out-channel, x:image-heigh, y:image-width, ic;in-channel, kh:kernel-height, kw:kernel-width
        oco, oci = com_operation.split(com_operation.op.axis[1], factor=8)
        xo, yo, xi, yi = com_operation.tile(com_operation.op.axis[2], com_operation.op.axis[3], x_factor=4, y_factor=4)
        
        ic, kh, kw = com_operation.op.reduce_axis
        ico, ici = com_operation.split(ic, factor=8)
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