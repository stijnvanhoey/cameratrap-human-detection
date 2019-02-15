# cameratrap-human-detection
Code for forward loop detection of humans on images

**Note**: The repository has a structure with a `data` folder, but this is excluded from the repo to avoid accurrence of photos with persons on online. Also Jupyter notebooks were excluded apart from 2 example notebooks, as a risk of having them containing photos indirectly is possible. Make sure to avoid adding images to these and commit with the images.

When working on the code in the notebooks, creating this folder structure will help you. Add raw images to work on in the `raw` folder


```
├── data
│   ├── processed
│   │   ├── extractions
│   │   ├── face_detection
│   │   └── yolo_model
│   └── raw
├── LICENSE
├── README.md
└── src
    ├── model
    │   ├── deploy.prototxt
    │   ├── res10_300x300_ssd_iter_140000_fp16.caffemodel
    │   └── yolo.h5
    ├── models.py
    ├── tourist_detection_yolo.ipynb
    ├── tourist_face_detection_dnn.ipynb
    ├── utils.py
    └── yolo_detecion.py
```

## Python dependencies

To run the code, the easiest way is to work with miniconda/anaconda and using the provide `environment.yml` file. To setup the Python environment, use:

```
conda env create -f environment.yml
```

which will setup the required packages to run the yolo based analysis. For the face detection, for the tryouts with other packages, just the link is provided to install it yourself with pip/conda.


## Yolo model

The yolo model file `yolo.h5` is available at the Github [release page of OlafenwaMosel](https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5). To run the code, put the model inside a folder `model` within the `src` folder.

## face detection model

The face detection model can be downloaded from the [opencv repo](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector). Note, the `deploy.prototxt` can be downloaded as such. For the `res10_300x300_ssd_iter_140000_fp16.caffemodel` file, you need to run the `download_weights.py` file together which will require the `weights.meta4` file as well.
