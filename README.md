<h1 align="center">ConeGS: Error-Guided Densification Using Pixel Cones for Improved Reconstruction with Fewer Primitives</h1>

<p align="center" dir="auto">
    <a href="https://baranowskibrt.github.io/" rel="nofollow">Bartłomiej Baranowski</a>
    ·
    <a href="https://s-esposito.github.io/" rel="nofollow">Stefano Esposito</a>
    ·
    <a href="https://patriciagschossmann.github.io/" rel="nofollow">Patricia Gschoßmann</a>
    ·
    <a href="https://apchenstu.github.io/" rel="nofollow">Anpei Chen</a>
    ·
    <a href="http://www.cvlibs.net/" rel="nofollow">Andreas Geiger</a>
  </p>

<div align="center">

[![button](https://img.shields.io/badge/Project%20Website-blue?style=for-the-badge)](https://baranowskibrt.github.io/conegs/)
[![button](https://img.shields.io/badge/Paper-green?style=for-the-badge)](https://arxiv.org/abs/2511.06810)
[![button](https://img.shields.io/badge/Video-red?style=for-the-badge)](https://youtu.be/ET0q50QAlbI)
</div>

<p align="center">
  <img src="./assets/conegs_teaser.png" width="99%">
</p>


## 🛠️ Installation


The code was tested on Python 3.11 with PyTorch 2.5.1 with CUDA Toolkit 12.1 and 11.8.

Installed CUDA Toolkit is required. To run the code install the following packages:

```shell
git clone https://github.com/baranowskibrt/conegs.git --recursive

conda create -n conegs python=3.11
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install git+https://github.com/nerfstudio-project/nerfacc.git
pip install submodules/simple-knn
pip install submodules/diff-gaussian-rasterization

pip install -r requirements.txt
```

## 📝 Usage

### Single scene

For a particular scene run (Mip-NeRF360's bicycle scene as an example) with a specified budget run:

```
python train.py --config-name defaults.yaml gaussian_model.source_path=../scenes_mipnerf scene_name=bicycle run_name=benchmark optimization.max_points=100000 gaussian_model.images=images_4 
```

And without budget:

```
python train.py --config-name defaults.yaml gaussian_model.source_path=../scenes_mipnerf scene_name=bicycle run_name=benchmark optimization.max_points=100000 gaussian_model.images=images_4 
```

### Benchmark


For all benchmarks on a specific budget adjust the max points (primitives), and run:

```
python full_eval.py --mipnerf360 ../scenes_mipnerf --tanksandtemples ../scenes_tt --deepblending ../scenes_db --ommo ../scenes --output_path benchmarks --common_args " optimization.max_points=100000"
```

Or for unbudget case:

```
python full_eval.py --mipnerf360 ../scenes/ --tanksandtemples ../scenes/ --deepblending ../scenes --ommo ../scenes --output_path benchmarks --common_args " optimization.max_points=0 optimization.gaussian_percentage_increase=0.02"
```
## 🔄 Pipeline

<p align="center">
  <img src="./assets/pipeline.png" width="99%">
</p>



## 🛠️ Data

### Datasets

You can access the datasets we evaluated on here:

- [**MipNeRF360** ](https://jonbarron.info/mipnerf360/)
- [**OMMO** ](https://drive.google.com/drive/folders/1Nu_xD4CUc_1f2YKEbdZP-nfbzyI3na7m?usp=sharing)
- [**Tanks & Temples and Deep Blending**](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip)

### Your own data

Check out the [3DGS code](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#processing-your-own-scenes) for the preprocessing needed.

## 🎓 Citation
```bibtex
@inproceedings{baranowski2026conegs, 
    title={ConeGS: Error-Guided Densification Using Pixel Cones for Improved Reconstruction with Fewer Primitives}, 
    author={Bartłomiej Baranowski and Stefano Esposito and Patricia Gschoßmann and Anpei Chen and Andreas Geiger},
    year={2026},
    booktitle = {2026 International Conference on 3D Vision (3DV)}, 
}

```