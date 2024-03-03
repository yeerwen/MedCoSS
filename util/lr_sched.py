# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def adjust_moment_cof(base_value, final_value, epochs, niter_per_ep):
    print("base_value: {}, final_value: {}, epoch: {}, niter: {}".format(base_value, final_value, epochs, niter_per_ep))
    iters = np.arange(epochs * niter_per_ep)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

if __name__ == '__main__':
    schedule = adjust_moment_cof(1.0, 0.996, 300, 695)
    with open("text.csv", "w") as fp:
        for i in schedule:
            fp.write(str(i)+ "\n")
    print(schedule)