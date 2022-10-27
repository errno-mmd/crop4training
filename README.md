# crop4training
Crop images for training of AI, focusing on a person in each image

## Prerequirement

You need to install the following libraries to run crop4training.

- [PyTorch](https://pytorch.org/) :
You can [install PyTorch by conda or pip](https://pytorch.org/get-started/locally/#start-locally)
- [OpenCV](https://opencv.org/)
- [Detectron2](https://github.com/facebookresearch/detectron2) :
You can [install Detectron2 by pip](https://detectron2.readthedocs.io/en/latest/tutorials/install.html#install-pre-built-detectron2-linux-only) on Linux
- [Mediapipe](https://google.github.io/mediapipe/getting_started/python) :
You can [install mediapipe Python Solutions by pip](https://google.github.io/mediapipe/getting_started/python#ready-to-use-python-solutions)

### Installation example (for Ubuntu 20.04 WSL on Windows 11 with Anaconda)
```
conda create -y -n crop python=3.9
conda activate crop
conda install -y pytorch=1.10.2=py3.9_cuda10.2_cudnn7.6.5_0 torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install opencv-python
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.10/index.html
pip install mediapipe
```

## Usage

```
python crop4training.py --width 512 --height 512 input_dir output_dir
```
If you want to use as much pixels as possible, use --no_focus option. It may be suitable for image style learning.
```
python crop4training.py --width 512 --height 512 --no_focus input_dir output_dir
```
for more details :
```
python crop4training.py --help
``` 

## License
MIT License. See LICENSE for more details.
