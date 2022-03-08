import os
import time
import math
import torch
import torch.nn as nn
import torchvision.utils as utils
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from model_baseline import cancer_classifier
from util import cancer_dataset
import warnings
warnings.filterwarnings('ignore')


class unpacker(object):
    def __init__(self):
        super(unpacker, self).__init__()
        self.l = []

    def _unpacker(self, module):
        for m in module.children():
            if list(m.children()) == []:
                self.l.append(m)
            else:
                self._unpacker(m)
        
    def result(self):
        return self.l

def unpacking(module):
    unp = unpacker()
    unp._unpacker(module)

    return unp.result()

def explore_make_grid(conv_layer, grayscale=True, maxnum=9):
    w_s = conv_layer.weight
    w_s_l = w_s.size(0) * w_s.size(1)
    if grayscale:
        w_s_l = min(w_s_l, maxnum)
        w_s = w_s.reshape([-1, w_s.size(2), w_s.size(3)]).unsqueeze(1)[:w_s_l, ...]
        w_s = w_s.reshape([-1, 1, w_s.size(2), w_s.size(3)])
    else:
        w_s_l = min(w_s_l // 3, maxnum)
        w_s_lt = 3 * w_s_l
        w_s = w_s.reshape([-1, w_s.size(2), w_s.size(3)]).unsqueeze(1)[:w_s_lt, ...]
        w_s = w_s.reshape([-1, 3, w_s.size(2), w_s.size(3)])
    length = w_s.size(0)
    row = int(math.sqrt(length))
    col = (length + row - 1) // row
    grids = utils.make_grid(w_s, nrow=col, padding=1, normalize=True)
    grids /= grids.max()

    return grids


def visualizer(model, suptitle=None, grayscale=True, n_kernel_coefficient=256, skip_1x1=False, plt_save=None, plt_show=True):
    c_l = []
    a = unpacking(model)
    for l in a:
        if isinstance(l, nn.Conv2d):
            if skip_1x1:
                if l.weight.size(2) > 1 and l.weight.size(3) > 1:
                    c_l.append(l)
            else:
                c_l.append(l)

    if len(c_l) < 9:
        tinymode = True
    else:
        tinymode = False

    if tinymode:
        row = int(math.sqrt(len(c_l)))
        col = (len(c_l) + row - 1) // row
        n_kernels_per_layer = n_kernel_coefficient // len(c_l)
    else:
        row = 3
        col = 3
        n_kernels_per_layer = n_kernel_coefficient // 16
    
    fig = plt.figure(figsize=(10, 10))

    i_t = 1
    for i, l in enumerate(c_l):
        fig.add_subplot(row, col, i_t)
        if not tinymode:
            if i <= 5 or i > len(c_l) - 4:
                plt.imshow(explore_make_grid(l, grayscale=grayscale, maxnum=n_kernels_per_layer).permute(1, 2, 0).detach().cpu().numpy())
                plt.title(f"{i}th layer, size : {[l.weight.size(2), l.weight.size(3)]}")
                i_t += 1
        else:
            plt.imshow(explore_make_grid(l, grayscale=grayscale, maxnum=n_kernels_per_layer).permute(1, 2, 0).detach().cpu().numpy())
            plt.title(f"{i}th layer, size : {[l.weight.size(2), l.weight.size(3)]}")
            i_t += 1
    if suptitle is not None:
        plt.suptitle(suptitle)
    if plt_save is not None:
        plt.savefig(plt_save)
    if plt_show:
        plt.show()
    else:
        plt.cla()


def visualizer_firstlayer(model, skip_1x1=False, grayscale=True, memo="", plt_save=None, plt_show=True):
    a = unpacking(model)
    for l in a:
        if isinstance(l, nn.Conv2d):
            if skip_1x1:
                if l.weight.size(2) > 1 and l.weight.size(3) > 1:
                    c_l = l
                    break
            else:
                c_l = l
                break
    _ = plt.figure(figsize=(10, 10))
    plt.imshow(explore_make_grid(c_l, grayscale=grayscale, maxnum=100000).permute(1, 2, 0).detach().cpu().numpy())
    plt.title(f"1st layer visualization, {memo}")
    if plt_save is not None:
        plt.savefig(plt_save)
    if plt_show:
        plt.show()
    else:
        plt.cla()


if __name__ == "__main__":
    gray = True
    model = cancer_classifier().cuda()
    model.eval()
    visualizer(model, suptitle="Cancer_classifier_initial", grayscale=gray, skip_1x1=True, plt_save=f"./visualize/color_{str(gray)}_cancer_init_full.png")
    visualizer_firstlayer(model, grayscale=gray, memo="cancer_classifier initial", plt_save=f"./visualize/color_{str(gray)}_cancer_init_1st.png")

    path = "exp2/model_state_dict.pt"
    experiment = path.split("/")[0]
    model.load_state_dict(torch.load(path))
    visualizer(model, suptitle="Cancer_classifier_After_training", grayscale=gray, skip_1x1=True, plt_save=f"./visualize/color_{str(gray)}_cancer_after_full.png")
    visualizer_firstlayer(model, grayscale=gray, memo="cancer_classifier after training", plt_save=f"./visualize/color_{str(gray)}_cancer_after_1st.png")

    import torchvision
    resnet101 = torchvision.models.resnet101(pretrained=True)
    visualizer(resnet101, suptitle="resnet101_pretrained", grayscale=gray, skip_1x1=True, plt_save=f"./visualize/color_{str(gray)}_resnet101_full.png")
    visualizer_firstlayer(resnet101, grayscale=gray, memo="resnet101_pretrained", plt_save=f"./visualize/color_{str(gray)}_cancer_resnet101_1st.png")
