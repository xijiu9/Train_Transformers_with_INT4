# Hardware Implemented LSS+HQ

Code for hardware implementation LSS and HQ operator.

## INSTALL

Tested with PyTorch 1.12.1 + CUDA 11.3, on an Tesla A100 GPU. Nvidia RTX 3090 can also work. 

> Note: This cuda program is based on [Nvidia cutlass](https://github.com/NVIDIA/cutlass) version 2.10. You need to pull down the corresponding version library. 

### CutLass

```bash
cd ./cutlass/cutlass
#checkout branch
git checkout feature/2.10/updates_before_tagging
```

### LSSWeight

```bash
cd quantize_grad_weight_LSS+LSQ
python setup.py install
```

### LSSAct

```bash
cd quantize_grad_input_LSS+LSQ
python setup.py install
```

### HQ

```bash
cd quantize_forward_LSQ+HQ
python setup_easy.py install
```



## Speed Test

### Flops

```bash
cd quantize_grad_weight_LSS+LSQ
mkdir image
python test.py
```

The result is shown at quantize_grad_weight_LSS+LSQ/image/plot_flops.svg

### Time proportion

```bash
# Time proportion of LSSAct and LSSWeight
cd quantize_grad_weight_LSS+LSQ
mkdir image
python test.py

# Time proportion of HQ
cd ../quantize_forward_LSQ+HQ
mkdir image
python test.py
```

The result is shown at quantize_grad_weight_LSS+LSQ/image/LSSAct.svg, quantize_grad_weight_LSS+LSQ/image/LSSWeight.svg, quantize_forward_LSQ+HQ/image/HQ.svg, respectively.
