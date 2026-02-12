## Framework Selection

- I will use Torch/Torch Lightning based frameworks
- Popular - Detectron2 & mmdetection
- Chose mmdetection intially, the build issue are bad & the repo is no longer maintained.

## MMDetection install
```
python3 -m pip install openmmim
mim install mmengine
mim install "mmcv==2.1.0"   # This will take a while as whl doesn't exist
# might have to match the NVCCC vversion to cida as well
```


