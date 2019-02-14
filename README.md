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


## Yolo model

The yolo model file is available at the Github [release page of OlafenwaMosel](https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5). To run the code, put the model inside a folder `model` within the `src` folder.
