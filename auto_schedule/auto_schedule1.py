import tvm
import logging
from tvm import autotvm
import numpy as np
import sys

function = None

@autotvm.template 
def GEMMAutoTVM(*args):
    global function
    
    ops, bufs = function(*args)
    s = tvm.create_schedule(ops)
    com_operation2 = s.stages[2]

    x = com_operation2.op.axis[1]
    y = com_operation2.op.axis[2]
    k = com_operation2.op.reduce_axis[0]

    cfg = autotvm.get_config()
    cfg.define_knob("split_y", [4, 8, 32])
    cfg.define_knob("split_x", [4, 8, 32])
    cfg.define_knob("split_k", [4, 8])

    xo, xi = com_operation2.split(x, cfg["split_x"].val)
    yo, yi = com_operation2.split(y, cfg["split_y"].val)
    ko, ki = com_operation2.split(k, cfg["split_k"].val)
    cfg.define_annotate("yi_unroll", [yi], policy='try_unroll')
    # yio, yii = com_operation2.split(yi, factor=4)
    # com_operation2.unroll(yi)
    com_operation2.reorder(xo, ko, yo, xi, ki, yi)
    
    return s, bufs

@autotvm.template 
def CONVAutoTVM(*args):
    global function
    
    ops, bufs = function(*args)
    s = tvm.create_schedule(ops)

    com_operation3 = s.stages[3]

    oc = com_operation3.op.axis[1]
    x = com_operation3.op.axis[2]
    y = com_operation3.op.axis[3]
    ic = com_operation3.op.reduce_axis[0]
    kh = com_operation3.op.reduce_axis[1]
    kw = com_operation3.op.reduce_axis[2]

    cfg = autotvm.get_config()
    cfg.define_knob("split_oc", [4, 8, 32])
    cfg.define_knob("split_x", [4, 8, 32])
    cfg.define_knob("split_y", [4, 8, 32])
    cfg.define_knob("split_ic", [4, 8, 32])

    oco, oci = com_operation3.split(oc, cfg["split_oc"].val)
    xo, xi = com_operation3.split(x, cfg["split_x"].val)
    yo, yi = com_operation3.split(y, cfg["split_y"].val)
    ico, ici = com_operation3.split(ic, cfg["split_ic"].val)
    # cfg.define_annotate("yi_unroll", [yi], policy='try_unroll')
    # yio, yii = com_operation2.split(yi, factor=4)
    # com_operation2.unroll(yi)
    com_operation3.reorder(oco, ico, xo, yo, oci, ici, xi, yi)

    # for stage 1
    com_operation1 = s.stages[1]
    com_operation1.comput_inline()


    # for stage 5 optional
    return s, bufs

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
    global function
    function = func

    # ops, bufs = func(*args)
    # s = tvm.create_schedule(ops)
    # print(len(s.stages))
    # for item in s.stages:
    #     print(item)
    #     print(dir(item))
    #     print("num_child_stages: ", item.num_child_stages)
    #     print("op: ", type(item.op), item.op)
    #     print("origin_op: ", type(item.origin_op), origin_op)
    #     print()

    
    # return s, bnfs
    autotvmFunc = None
    config_sp_size = 0
    if len(args) == 4:
        config_sp_size = 36
        autotvmFunc = GEMMAutoTVM
    else:
        config_sp_size = 81
        autotvmFunc = CONVAutoTVM


    task = autotvm.task.create(autotvmFunc, args=(args), target='llvm')
    print(task.config_space)
    
    # print log
    # logging.getLogger('autotvm').setLevel(logging.DEBUG)
    # logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))
    
    # There are two steps for measuring a config: build and run.
    # By default, we use all CPU cores to compile program. Then measure them sequentially.
    # We measure 5 times and take average to reduce variance.
    measure_option = autotvm.measure_option(
        builder='local',
        runner=autotvm.LocalRunner(number=3))
    
    # begin tuning, log records to file `matmul.log`
    tuner = autotvm.tuner.GridSearchTuner(task)
    tuner.tune(n_trial=config_sp_size,
               measure_option=measure_option,
               callbacks=[autotvm.callback.log_to_file('matmul.log')])
    
    with autotvm.apply_history_best('matmul.log'):
        with tvm.target.create("llvm"):
            s, arg_bufs = autotvmFunc(*args)
            print(tvm.lower(s, arg_bufs, simple_mode=True))
            return s, arg_bufs
