Description
-----------
This repository contains a Python [Dash](https://dash.plotly.com/introduction) application for annotating keypoints 
in images. Which fork from [this](https://github.com/luiscarlosgph/keypoint-annotation-tool), thank you very much.

Install dependencies
--------------------
* OpenCV
```
# Ubuntu/Debian
$ sudo apt update
$ sudo apt install libopencv-dev python3-opencv

# Other platforms
$ python3 -m pip install opencv-python --user
```

Install this package
--------------------
```
# Install pip dependencies
$ python -m pip install numpy dash dash_bootstrap_components opencv-python, tqdm

# Download repo
$ git clone https://github.com/luiscarlosgph/keypoint-annotation-tool.git
$ cd keypoint-annotation-tool

# Install package
$ python setup.py install --user
```

Run
---
``` 
# You should be already inside the repo, where there is a sample 'data' folder to make this command work
# modify the --data-dir "data"  to your own data folder
$ python -m wat.run --data-dir data --port 1234 
```
The ```--data-dir``` parameter should contain two folders: ```input``` and ```output```.
The ```input``` folder should contain the images (```*.jpg``` or ```*.png```) to be annotated.
The ```output``` folder should be empty. The annotations will be stored there.
The ```--maxtips``` parameter sets the maximum number of tooltips that can be annotated per image.
By default this is set to four as there are typically two instruments in the scene with two tooltips 
each (one per clasper).

Deployment
----------
Go to [http://localhost:1234](http://localhost:1234) with your favourite web browser.

Annotations
-----------
When an image is annotated, the image file is moved from the ```input``` to the ```output``` folder.
Each image in the output folder (e.g. ```example.png```) is accompanied by two annotation files:

* ```example.npy```: the gaussion heatmap of the annotated keypoints, save to `gt_density_map` folder

* ```demo.json```:  save to `frames` folder,contains a list of annotations in JSON format, for example, 
                         if four tooltips are clicked, it will contain something like
    ```
    {
    "filename": "base.jpg",
    "density": "base.npy",
    "points": [
        [
            264,
            291
        ]
    ]
}
    ```

Reading the annotations
-----------------------
The script [read_annotations.py](https://github.
com/luiscarlosgph/keypoint-annotation-tool/blob/main/src/wat/read_annotations.py) is provided as a staring point to 
process and view your annotated images. The annotation image will save to data/output/vis
```
# Run this command after annotating the sample image provided with the repo
$ python -m wat.read_annotations --dir data/output
```

Demo image
----------
![alt text](https://github.com/luiscarlosgph/keypoint-annotation-tool/blob/main/demo/demo.jpg?raw=true)

License
-------
This code is distributed under the [MIT license](https://github.com/luiscarlosgph/keypoint-annotation-tool/blob/main/LICENSE).
