import tvm
import logging
from tvm import autotvm
import numpy as np
import sys

function = None
global_s = None
global_bufs = None

@autotvm.template 
def GEMMAutoTVM(*args):
    global function
    def getSplit(maxNum):
        splitList = []
        splitList.append(1)
        para = 2
        while (True):
            if para <= maxNum / 2 and para <= 32:
                splitList.append(para)
                para *= 2
            else:
                break
        if 16 in splitList:
            splitList.remove(16)
        return splitList
    
    ops, bufs = function(*args)
    s = tvm.create_schedule(ops)
    gemm_tensor = bufs[len(bufs) - 1]
    gemm_op = s[gemm_tensor]

    x = gemm_op.op.axis[1]
    y = gemm_op.op.axis[2]
    k = gemm_op.op.reduce_axis[0]

    cfg = autotvm.get_config()

    cfg.define_knob("split_y", getSplit(int(x.dom.extent)))
    cfg.define_knob("split_x", getSplit(int(y.dom.extent)))
    cfg.define_knob("split_k", getSplit(int(k.dom.extent)))

    xo, xi = gemm_op.split(x, cfg["split_x"].val)
    yo, yi = gemm_op.split(y, cfg["split_y"].val)
    ko, ki = gemm_op.split(k, cfg["split_k"].val)
    gemm_op.reorder(xo, ko, yo, xi, ki, yi)
    # cfg.define_annotate("yi_unroll", [yi], policy='try_unroll')
    # yio, yii = gemm_op.split(yi, factor=4)
    # gemm_op.unroll(yi)
    
    return s, bufs

@autotvm.template 
def CONVAutoTVM(*args):
    global function
    def getSplit(maxNum):
        splitList = []
        splitList.append(1)
        para = 2
        while (True):
            if para <= maxNum / 2 and para <= 32:
                splitList.append(para)
                para *= 2
            else:
                break
        if 16 in splitList:
            splitList.remove(16)
        return splitList
    
    ops, bufs = function(*args)
    s = tvm.create_schedule(ops)

    # get bias_tensor, conv_tensor, pad_tensor and their ops relatively
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

    # conv!
    oc = conv_op.op.axis[1]
    x = conv_op.op.axis[2]
    y = conv_op.op.axis[3]
    ic = conv_op.op.reduce_axis[0]
    kh = conv_op.op.reduce_axis[1]
    kw = conv_op.op.reduce_axis[2]

    cfg = autotvm.get_config()
    cfg.define_knob("split_oc", getSplit(int(oc.dom.extent)))
    cfg.define_knob("split_x", getSplit(int(x.dom.extent)))
    cfg.define_knob("split_y", getSplit(int(y.dom.extent)))
    cfg.define_knob("split_ic", getSplit(int(ic.dom.extent)))

    oco, oci = conv_op.split(oc, cfg["split_oc"].val)
    xo, xi = conv_op.split(x, cfg["split_x"].val)
    yo, yi = conv_op.split(y, cfg["split_y"].val)
    ico, ici = conv_op.split(ic, cfg["split_ic"].val)
    conv_op.reorder(oco, ico, xo, yo, oci, ici, xi, yi)

    # pad!
    pad_x = pad_op.op.axis[2]
    pad_y = pad_op.op.axis[3]
    pad_c = pad_op.op.axis[1]
    cfg.define_knob("split_pad_c", getSplit(int(pad_c.dom.extent)))
    cfg.define_knob("split_pad_x", getSplit(int(pad_x.dom.extent)))
    cfg.define_knob("split_pad_y", getSplit(int(pad_y.dom.extent)))
    pad_co, pad_ci = pad_op.split(pad_c, cfg["split_pad_c"].val)
    pad_xo, pad_xi = pad_op.split(pad_x, cfg["split_pad_x"].val)
    pad_yo, pad_yi = pad_op.split(pad_y, cfg["split_pad_y"].val)
    pad_op.reorder(pad_co, pad_xo, pad_yo, pad_ci, pad_xi, pad_yi)

    # bias
    if bias_tensor == None:
        return s, bufs
    bias_x = bias_op.op.axis[2]
    bias_y = bias_op.op.axis[3]
    bias_c = bias_op.op.axis[1]
    cfg.define_knob("split_bias_c", getSplit(int(bias_c.dom.extent)))
    cfg.define_knob("split_bias_x", getSplit(int(bias_x.dom.extent)))
    cfg.define_knob("split_bias_y", getSplit(int(bias_y.dom.extent)))
    bias_co, bias_ci = bias_op.split(bias_c, cfg["split_bias_c"].val)
    bias_xo, bias_xi = bias_op.split(bias_x, cfg["split_bias_x"].val)
    bias_yo, bias_yi = bias_op.split(bias_y, cfg["split_bias_y"].val)
    bias_op.reorder(bias_co, bias_xo, bias_yo, bias_ci, bias_xi, bias_yi)
    
    # cfg.define_annotate("yi_unroll", [yi], policy='try_unroll')
    
    # pad_op.compute_inline() # too bad!

    return s, bufs

def auto_schedule_gemm(func, args):
    logFile = open("matmul.log", 'w', encoding="utf-8")
    logFile.truncate()
    logFile.close()
    config_sp_size = 100
    autotvmFunc = GEMMAutoTVM

    task = autotvm.task.create(autotvmFunc, args=(args), target='llvm')
    print(task.config_space)
    
    measure_option = autotvm.measure_option(
        builder='local',
        runner=autotvm.LocalRunner(number=3)) # 3 trials for every sample
    

    tuner = autotvm.tuner.GATuner(task)
    tuner.tune(n_trial=config_sp_size,
               measure_option=measure_option,
               callbacks=[autotvm.callback.log_to_file('matmul.log')])
    
    with autotvm.apply_history_best('matmul.log'):
        with tvm.target.create("llvm"):
            s, arg_bufs = autotvmFunc(*args)
            print(tvm.lower(s, arg_bufs, simple_mode=True))
            return s, arg_bufs

def auto_schedule_conv(func, args):
    config_sp_size = 200
    autotvmFunc = CONVAutoTVM

    task = autotvm.task.create(autotvmFunc, args=(args), target='llvm')
    print(task.config_space)
    
    measure_option = autotvm.measure_option(
        builder='local',
        runner=autotvm.LocalRunner(number=3, timeout=1)) # 3 trials for every sample
    

    tuner = autotvm.tuner.GATuner(task)
    tuner.tune(n_trial=config_sp_size,
               measure_option=measure_option,
               callbacks=[autotvm.callback.log_to_file('matmul.log')])
    
    with autotvm.apply_history_best('matmul.log'):
        with tvm.target.create("llvm"):
            s, arg_bufs = autotvmFunc(*args)
            print(tvm.lower(s, arg_bufs, simple_mode=True))
            return s, arg_bufs

def auto_schedule(func, args):

    global function
    function = func

    if len(args) == 4:
        return auto_schedule_gemm(func, args)
    else:
        return auto_schedule_conv(func, args)
