## DAFNet: An Image Restoration Model Based on an Improved SimpleGate
The official pytorch implementation of the paper **DAFNet: An Image Restoration Model Based on an Improved SimpleGate**

### Installation
This implementation based on [BasicSR](https://github.com/xinntao/BasicSR) which is a open source toolbox for image restoration tasks and [NAFNet](https://github.com/megvii-research/NAFNet) 

### Results and Pre-trained Models

| name | Dataset|PSNR|SSIM|
|:----|:----|:----|:----|
|DAFNet-GoPro|GoPro|33.19|0.963|
|NAFNet-SIDD|SIDD|40.03|0.960|
|NAFNet-REDS|REDS|29.12|0.867|
|NAFNet-Rain13K|Test100|30.68|0.904|
|NAFNet-Rain13K|Rain100L|37.90|0.973|
|NAFNet-Rain13K|Rain100H|30.63|0.893|
|NAFNet-Rain13K|Test2800|32.49|0.940|

```python
python 3.7.13
pytorch 1.12.1
cuda 11.7
```

```
git clone https://github.com/pcpcpQAQ/DAFNet
cd DAFNet
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```

### Quick Start 
* Single Image Inference Demo:
    * Image Denoising:
    ```
    python basicsr/demo.py -opt options/test/SIDD/DAFNet-width32.yml --input_path 'degard img path' --output_path 'denoising img path'
    ```
    * Image Deblurring:
    ```
    python basicsr/demo.py -opt options/test/REDS/DAFNet-width64.yml --input_path 'degard img path' --output_path 'deblurring img path'
    ```
    * Image Deraining:
    ```
    python basicsr/demo.py -opt options/test/Derain/DAFNet-width32.yml --input_path 'degard img path' --output_path 'deraining img path'
    ```
    * ```--input_path```: the path of the degraded image
    * ```--output_path```: the path to save the predicted image
 
* Training:
    * Image Denoising:
    ```
    python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/SIDD/DAFNet-width32.yml --launcher pytorch
    ```
    * Image Deblurring:
    ```
    python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/GoPro/DAFNet-width32.yml --launcher pytorch
    ```
    ```
    python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/REDS/DAFNet-width64.yml --launcher pytorch
    ```
    * Image Deraining:
    ```
    python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/Derain/DAFNet-width32.yml --launcher pytorch
    ```

* Evaluation:
    * SIDD Dataset:
    ```
    python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt ./options/test/SIDD/DAFNet-width32.yml --launcher pytorch
    ```
    * GoPro Dataset:
    ```
    python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt ./options/test/GoPro/DAFNet-width32.yml --launcher pytorch
    ```
    * REDS Dataset:
    ```
    python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt ./options/test/REDS/DAFNet-width64.yml --launcher pytorch
    ```
    * Test100 Dataset:
    ```
    python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt ./options/test/Derain/DAFNet-Test100.yml --launcher pytorch
    ```
    * Rain100L Dataset:
    ```
    python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt ./options/test/Derain/DAFNet-Rain100L.yml --launcher pytorch
    ```
    * Rain100H Dataset:
    ```
    python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt ./options/test/Derain/DAFNet-Rain100H.yml --launcher pytorch
    ```
    * Rain2800 Dataset:
    ```
    python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt ./options/test/Derain/DAFNet-Rain2800.yml --launcher pytorch
    ```
    * ```Image Deraining(Test100、Rain100L、Rain100H、Test2800)```: need ```./options/test/Derain/evaluate_PSNR_SSIM.m``` evalute PSNR and SSIM
