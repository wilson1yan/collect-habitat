# Habitat

## Installation

Note that this requires a GPU, and you make need to install EGL

```
conda create -y -n habitat python=3.7 cmake=3.14.0
conda activate habitat
conda install habitat-sim headless -c conda-forge -c aihabitat

git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e .

cd collect-habitat
pip install -r requirements.txt
```

## Collecting Trajectories

`python collect_parallel.py -d data_path -o output_path`
where `data_path` is an abitrarily recursively deep folder of `*.glb` files of 3D scenes
