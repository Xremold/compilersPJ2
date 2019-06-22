import tvm
import logging
from tvm import autotvm
import numpy as np
import sys

function = None

@autotvm.template 
def GEMMAutoTVM(*args):
    global function
    def getSplit(maxNum):
        splitList = []
        para = 2
        while (True):
            if para <= maxNum / 2 and para <= 32:
                splitList.append(para)
                para *= 2
            else:
                break
        if len(splitList) == 0:
            splitList.append(1)
        return splitList
    
    ops, bufs = function(*args)
    s = tvm.create_schedule(ops)
    gemm_tensor = bufs[len(bufs) - 1]
    gemm_op = s[gemm_tensor]

    x = gemm_op.op.axis[1]
    y = gemm_op.op.axis[2]
    k = gemm_op.op.reduce_axis[0]

    cfg = autotvm.get_config()
    # print(alist)
    cfg.define_knob("split_y", getSplit(int(x.dom.extent)))
    cfg.define_knob("split_x", getSplit(int(y.dom.extent)))
    cfg.define_knob("split_k", getSplit(int(k.dom.extent)))
    # print("heiheihei")

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
        para = 2
        while (True):
            if para < maxNum / 4 and para <= 32:
                splitList.append(para)
                para *= 2
            else:
                break
        if len(splitList) == 0:
            splitList.append(1)
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
    conv_op.reorder(oco, ico, xo, yo, oci, ici, kh, kw, xi, yi)
    cfg.define_annotate("yi_unroll", [yi], policy='try_unroll')
    
    pad_op.compute_inline()

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

    global function
    function = func
    logFile = open("matmul.log", 'w', encoding="utf-8")
    logFile.truncate()
    logFile.close()
    
    # return s, bnfs
    autotvmFunc = None
    config_sp_size = 0
    if len(args) == 4:
        config_sp_size = 50
        autotvmFunc = GEMMAutoTVM
    else:
        config_sp_size = 100
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
    tuner = autotvm.tuner.GATuner(task)
    tuner.tune(n_trial=config_sp_size,
               measure_option=measure_option,
               callbacks=[autotvm.callback.log_to_file('matmul.log')])
    
    with autotvm.apply_history_best('matmul.log'):
        with tvm.target.create("llvm"):
            s, arg_bufs = autotvmFunc(*args)
            print(tvm.lower(s, arg_bufs, simple_mode=True))
            return s, arg_bufs
