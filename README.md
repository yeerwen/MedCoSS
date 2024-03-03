# MedCoSS
This is the official Pytorch implementation of our CVPR 2024 paper "[Continual Self-supervised Learning: Towards Universal Multi-modal Medical Data Representation Learning](https://arxiv.org/abs/2311.17597)".

<div align="center">
  <img width="100%" alt="MedCoSS illustration" src="github/Overview.png">
</div>

## Requirements
CUDA 11.5<br />
Python 3.8<br /> 
Pytorch 1.11.0<br />
CuDNN 8.3.2.44

### Data Preparation
Coming soon.

### Pre-processing
Coming soon.

### Pre-training
* Download [uni-perceiver-base-L12-H768-224size-torch-pretrained.pth](https://github.com/fundamentalvision/Uni-Perceiver/blob/main/data/checkpoints.md).
* Run `sh run_ssl.sh` for training (4 GPUs with 24G). (Before running it, you need to modify some addresses.)



## To do
- [ ] Dataset Links
- [ ] Pre-processing Code
- [x] Pre-training Code Release
- [ ] Pre-trained Model
- [ ] Fine-tuning Code Release


## Citation
If this code is helpful for your study, please cite:

```
@article{ye2024medcoss,
  title={Continual Self-supervised Learning: Towards Universal Multi-modal Medical Data Representation Learning},
  author={Ye, Yiwen and Xie, Yutong and Zhang, Jianpeng and Chen, Ziyang and Wu, Qi and Xia, Yong},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={},
  year={2024},
}
```

## Acknowledgements
The whole framework is based on [MAE](https://github.com/facebookresearch/mae), [Uni-Perceiver](https://github.com/fundamentalvision/Uni-Perceiver), and [MGCA](https://github.com/HKU-MedAI/MGCA/tree/main).

## Contact
Yiwen Ye (ywye@mail.nwpu.edu.cn)