Text Detection and Text Recognition inference code using ONNX Runtime

### Installation

```
conda create -n PaddleOCR python=3.8
conda activate PaddleOCR
pip install onnxruntime-gpu==1.12.1
pip install opencv-python==4.5.5.64
pip install shapely==2.0.2
pip install pyclipper==1.3.0.post5
```

### Test

* Run `python main.py a.jpg` for testing

### Note

* This repo supports inference only, see reference for more details

### Results

![title](demo/demo.jpg)

#### Reference

* https://github.com/PaddlePaddle/PaddleOCR
