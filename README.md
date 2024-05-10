# Yolo v7, v5, v4, v3 and v2 for Windows and Linux

YOLOv7 is more accurate and faster than YOLOv5 by **120%** FPS, than YOLOX by **180%** FPS, than Dual-Swin-T by **1200%** FPS, than ConvNext by **550%** FPS, than SWIN-L by **500%** FPS, than PPYOLOE-X by **150%** FPS.

YOLOv7 surpasses all known object detectors in both speed and accuracy in the range from 5 FPS to 160 FPS and has the highest accuracy 56.8% AP among all known real-time object detectors with 30 FPS or higher on GPU V100, batch=1. 

* YOLOv7-e6 (55.9% AP, 56 FPS V100 b=1) by `+500%` FPS faster than SWIN-L C-M-RCNN (53.9% AP, 9.2 FPS A100 b=1)
* YOLOv7-e6 (55.9% AP, 56 FPS V100 b=1) by `+550%` FPS faster than ConvNeXt-XL C-M-RCNN (55.2% AP, 8.6 FPS A100 b=1)
* YOLOv7-w6 (54.6% AP, 84 FPS V100 b=1) by `+120%` FPS faster than YOLOv5-X6-r6.1 (55.0% AP, 38 FPS V100 b=1)
* YOLOv7-w6 (54.6% AP, 84 FPS V100 b=1) by `+1200%` FPS faster than Dual-Swin-T C-M-RCNN (53.6% AP, 6.5 FPS V100 b=1)
* YOLOv7x (52.9% AP, 114 FPS V100 b=1) by `+150%` FPS faster than PPYOLOE-X (51.9% AP, 45 FPS V100 b=1)
* YOLOv7 (51.2% AP, 161 FPS V100 b=1) by `+180%` FPS faster than YOLOX-X (51.1% AP, 58 FPS V100 b=1)

![modern_gpus](https://user-images.githubusercontent.com/4096485/82835867-f1c62380-9ecd-11ea-9134-1598ed2abc4b.png) AP50:95 / AP50 - FPS (Tesla V100) Paper: https://arxiv.org/abs/2004.10934

tkDNN-TensorRT accelerates YOLOv4 **~2x** times for batch=1 and **3x-4x** times for batch=4.

- tkDNN: https://github.com/ceccocats/tkDNN
- OpenCV: https://gist.github.com/YashasSamaga/48bdb167303e10f4d07b754888ddbdcf

### GeForce RTX 2080 Ti

| Network Size               | Darknet, FPS (avg) | tkDNN TensorRT FP32, FPS | tkDNN TensorRT FP16, FPS | OpenCV FP16, FPS | tkDNN TensorRT FP16 batch=4, FPS | OpenCV FP16 batch=4, FPS | tkDNN Speedup |
|:--------------------------:|:------------------:|-------------------------:|-------------------------:|-----------------:|---------------------------------:|-------------------------:|--------------:|
|320                         | 100                | 116                      | **202**                  | 183              | 423                              | **430**                  | **4.3x**      |
|416                         | 82                 | 103                      | **162**                  | 159              | 284                              | **294**                  | **3.6x**      |
|512                         | 69                 | 91                       | 134                      | **138**          | 206                              | **216**                  | **3.1x**      |
|608                         | 53                 | 62                       | 103                      | **115**          | 150                              | **150**                  | **2.8x**      |
|Tiny 416                    | 443                | 609                      | **790**                  | 773              | **1774**                         | 1353                     | **3.5x**      |
|Tiny 416 CPU Core i7 7700HQ | 3.4                | -                        | -                        | 42               | -                                | 39                       | **12x**       |

- Yolo v4 Full comparison: [map_fps](https://user-images.githubusercontent.com/4096485/80283279-0e303e00-871f-11ea-814c-870967d77fd1.png)
- Yolo v4 tiny comparison: [tiny_fps](https://user-images.githubusercontent.com/4096485/85734112-6e366700-b705-11ea-95d1-fcba0de76d72.png)
- CSPNet: [paper](https://arxiv.org/abs/1911.11929) and [map_fps](https://user-images.githubusercontent.com/4096485/71702416-6645dc00-2de0-11ea-8d65-de7d4b604021.png) comparison: https://github.com/WongKinYiu/CrossStagePartialNetworks
- Yolo v3 on MS COCO: [Speed / Accuracy (mAP@0.5) chart](https://user-images.githubusercontent.com/4096485/52151356-e5d4a380-2683-11e9-9d7d-ac7bc192c477.jpg)
- Yolo v3 on MS COCO (Yolo v3 vs RetinaNet) - Figure 3: https://arxiv.org/pdf/1804.02767v1.pdf
- Yolo v2 on Pascal VOC 2007: https://hsto.org/files/a24/21e/068/a2421e0689fb43f08584de9d44c2215f.jpg
- Yolo v2 on Pascal VOC 2012 (comp4): https://hsto.org/files/3a6/fdf/b53/3a6fdfb533f34cee9b52bdd9bb0b19d9.jpg

#### Youtube video of results

| [![Yolo v4](https://user-images.githubusercontent.com/4096485/101360000-1a33cf00-38ae-11eb-9e5e-b29c5fb0afbe.png)](https://youtu.be/1_SiUOYUoOI "Yolo v4") |  [![Scaled Yolo v4](https://user-images.githubusercontent.com/4096485/101359389-43a02b00-38ad-11eb-866c-f813e96bf61a.png)](https://youtu.be/YDFf-TqJOFE "Scaled Yolo v4") |
|---|---|

Others: https://www.youtube.com/user/pjreddie/videos

#### Datasets

- MS COCO: use `./scripts/get_coco_dataset.sh` to get labeled MS COCO detection dataset
- OpenImages: use `python ./scripts/get_openimages_dataset.py` for labeling train detection dataset
- Pascal VOC: use `python ./scripts/voc_label.py` for labeling Train/Test/Val detection datasets
- ILSVRC2012 (ImageNet classification): use `./scripts/get_imagenet_train.sh` (also `imagenet_label.sh` for labeling valid set)
- German/Belgium/Russian/LISA/MASTIF Traffic Sign Datasets for Detection - use this parser: https://github.com/angeligareta/Datasets2Darknet#detection-task
- List of other datasets: https://github.com/AlexeyAB/darknet/tree/master/scripts#datasets

----------------------------------------------------------------------------------
# How to install on Linux (ubuntu jetson nano)
### Set cuda path
- `export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}`
- `export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}`
### Installing
- `git clone https://github.com/the3rdchild/Yolo_jetson-nano`
- `cd Yolo_jetson-nano` you can rename the folder `Yolo_jetson-nano` located in you `Home` with whatever name you want to make it more easy when you access it using terminal: `cd ~/[folder name]`
- *this step is optional: `wget [link].weights` you can train your own weights or download it using `wget` from internet in case you want another model of weights, there is a model inside `Yolo_jetson-nano` folder, name: `yolov7-tiny.weights`
- `make` I have edited the `Makefile` so it can run perfectly with jetson nano in 800x600 pixel (change inside `~/src/image_opencv.cpp` line 610 width and line 611 height) input webcam, you can also change the `Makefile` by following this instructions: "How to compile on Linux (using make)"

## Run YoloV7-tiny
- example: run with webcam 0 `./Yolo_jetson-nano detector demo cfg/coco.data cfg/yolov7-tiny.cfg yolov7-tiny.weights -c 0`

## Run with docker
- example: `sudo docker run --runtime=nvidia --rm -v $PWD:/workspace -w /workspace daisukekobayashi/darknet:gpu darknet detector test data/coco.data yolov7-tiny.cfg yolov7-tiny.weights -i 0 -thresh 0.25 dog.jpg -ext_output` you need to run inside `./Yolo_jetson-nano/build/darknet/x64`, the process will download image of darknet before, so it can run with docker. the problem there's no images from daisukekobayashi repo for jetson nano [arm64] (daisukekobayashi has amd64 images only), so if you run that command on jetson it will only waste your time by doing nothing, the soluiton is you need to find docker image for darknet from another source.
----------------------------------------------------------------------------------

### How to compile on Linux (using `make`)

Just do `make` in the darknet directory. (You can try to compile and run it on Google Colab in cloud [link](https://colab.research.google.com/drive/12QusaaRj_lUwCGDvQNfICpa7kA7_a2dE) (press «Open in Playground» button at the top-left corner) and watch the video [link](https://www.youtube.com/watch?v=mKAEGSxwOAY) )
Before make, you can set such options in the `Makefile`: [link](https://github.com/AlexeyAB/darknet/blob/9c1b9a2cf6363546c152251be578a21f3c3caec6/Makefile#L1)

- `GPU=1` to build with CUDA to accelerate by using GPU (CUDA should be in `/usr/local/cuda`)
- `CUDNN=1` to build with cuDNN v5-v7 to accelerate training by using GPU (cuDNN should be in `/usr/local/cudnn`)
- `CUDNN_HALF=1` to build for Tensor Cores (on Titan V / Tesla V100 / DGX-2 and later) speedup Detection 3x, Training 2x
- `OPENCV=1` to build with OpenCV 4.x/3.x/2.4.x - allows to detect on video files and video streams from network cameras or web-cams
- `DEBUG=1` to build debug version of Yolo
- `OPENMP=1` to build with OpenMP support to accelerate Yolo by using multi-core CPU
- `LIBSO=1` to build a library `darknet.so` and binary runnable file `uselib` that uses this library. Or you can try to run so `LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH ./uselib test.mp4` How to use this SO-library from your own code - you can look at C++ example: https://github.com/AlexeyAB/darknet/blob/master/src/yolo_console_dll.cpp
    or use in such a way: `LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH ./uselib data/coco.names cfg/yolov4.cfg yolov4.weights test.mp4`
- `ZED_CAMERA=1` to build a library with ZED-3D-camera support (should be ZED SDK installed), then run
    `LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH ./uselib data/coco.names cfg/yolov4.cfg yolov4.weights zed_camera`
- You also need to specify for which graphics card the code is generated. This is done by setting `ARCH=`. If you use a newer version than CUDA 11 you further need to edit line 20 from Makefile and remove `-gencode arch=compute_30,code=sm_30 \` as Kepler GPU support was dropped in CUDA 11. You can also drop the general `ARCH=` and just uncomment `ARCH=` for your graphics card.
### if you encounter error during `make` compiling: something about nvcc: you can try this `export PATH=/usr/local/cuda/bin:$PATH` and do the `make` again

### How to compile on Windows (using `CMake`)

Requires:

- MSVC: https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community
- CMake GUI: `Windows win64-x64 Installer`https://cmake.org/download/
- Download Darknet zip-archive with the latest commit and uncompress it: [master.zip](https://github.com/AlexeyAB/darknet/archive/master.zip)

In Windows:

- Start (button) -> All programs -> CMake -> CMake (gui) ->

- [look at image](https://habrastorage.org/webt/pz/s1/uu/pzs1uu4heb7vflfcjqn-lxy-aqu.jpeg) In CMake: Enter input path to the darknet Source, and output path to the Binaries -> Configure (button) -> Optional platform for generator: `x64`  -> Finish -> Generate -> Open Project ->

- in MS Visual Studio: Select: x64 and Release -> Build -> Build solution

- find the executable file `darknet.exe` in the output path to the binaries you specified

![x64 and Release](https://habrastorage.org/webt/ay/ty/f-/aytyf-8bufe7q-16yoecommlwys.jpeg)

### How to compile on Windows (using `vcpkg`)

This is the recommended approach to build Darknet on Windows.

1. Install Visual Studio 2017 or 2019. In case you need to download it, please go here: [Visual Studio Community](http://visualstudio.com). Remember to install English language pack, this is mandatory for vcpkg!

2. Install CUDA enabling VS Integration during installation.

3. Open Powershell (Start -> All programs -> Windows Powershell) and type these commands:

```PowerShell
Set-ExecutionPolicy unrestricted -Scope CurrentUser -Force
git clone https://github.com/AlexeyAB/darknet
cd darknet
.\build.ps1 -UseVCPKG -EnableOPENCV -EnableCUDA -EnableCUDNN
```

(add option `-EnableOPENCV_CUDA` if you want to build OpenCV with CUDA support - very slow to build! - or remove options like `-EnableCUDA` or `-EnableCUDNN` if you are not interested in them). If you open the `build.ps1` script at the beginning you will find all available switches.

## How to train with multi-GPU

1. Train it first on 1 GPU for like 1000 iterations: `darknet.exe detector train cfg/coco.data cfg/yolov4.cfg yolov4.conv.137`

2. Then stop and by using partially-trained model `/backup/yolov4_1000.weights` run training with multigpu (up to 4 GPUs): `darknet.exe detector train cfg/coco.data cfg/yolov4.cfg /backup/yolov4_1000.weights -gpus 0,1,2,3`

If you get a Nan, then for some datasets better to decrease learning rate, for 4 GPUs set `learning_rate = 0,00065` (i.e. learning_rate = 0.00261 / GPUs). In this case also increase 4x times `burn_in =` in your cfg-file. I.e. use `burn_in = 4000` instead of `1000`.

https://groups.google.com/d/msg/darknet/NbJqonJBTSY/Te5PfIpuCAAJ

## How to train (to detect your custom objects)

(to train old Yolo v2 `yolov2-voc.cfg`, `yolov2-tiny-voc.cfg`, `yolo-voc.cfg`, `yolo-voc.2.0.cfg`, ... [click by the link](https://github.com/AlexeyAB/darknet/tree/47c7af1cea5bbdedf1184963355e6418cb8b1b4f#how-to-train-pascal-voc-data))

#### How to use on the command line

If you use `build.ps1` script or the makefile (Linux only) you will find `darknet` in the root directory.

If you use the deprecated Visual Studio solutions, you will find `darknet` in the directory `\build\darknet\x64`.

If you customize build with CMake GUI, darknet executable will be installed in your preferred folder.

- Yolo v4 COCO - **image**: `./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights -thresh 0.25`
- **Output coordinates** of objects: `./darknet detector test cfg/coco.data yolov4.cfg yolov4.weights -ext_output dog.jpg`
- Yolo v4 COCO - **video**: `./darknet detector demo cfg/coco.data cfg/yolov4.cfg yolov4.weights -ext_output test.mp4`
- Yolo v4 COCO - **WebCam 0**: `./darknet detector demo cfg/coco.data cfg/yolov4.cfg yolov4.weights -c 0`
- Yolo v4 COCO for **net-videocam** - Smart WebCam: `./darknet detector demo cfg/coco.data cfg/yolov4.cfg yolov4.weights http://192.168.0.80:8080/video?dummy=param.mjpg`
- Yolo v4 - **save result videofile res.avi**: `./darknet detector demo cfg/coco.data cfg/yolov4.cfg yolov4.weights test.mp4 -out_filename res.avi`
- Yolo v3 **Tiny** COCO - video: `./darknet detector demo cfg/coco.data cfg/yolov3-tiny.cfg yolov3-tiny.weights test.mp4`
- **JSON and MJPEG server** that allows multiple connections from your soft or Web-browser `ip-address:8070` and 8090: `./darknet detector demo ./cfg/coco.data ./cfg/yolov3.cfg ./yolov3.weights test50.mp4 -json_port 8070 -mjpeg_port 8090 -ext_output`
- Yolo v3 Tiny **on GPU #1**: `./darknet detector demo cfg/coco.data cfg/yolov3-tiny.cfg yolov3-tiny.weights -i 1 test.mp4`
- Alternative method Yolo v3 COCO - image: `./darknet detect cfg/yolov4.cfg yolov4.weights -i 0 -thresh 0.25`
- Train on **Amazon EC2**, to see mAP & Loss-chart using URL like: `http://ec2-35-160-228-91.us-west-2.compute.amazonaws.com:8090` in the Chrome/Firefox (**Darknet should be compiled with OpenCV**):
    `./darknet detector train cfg/coco.data yolov4.cfg yolov4.conv.137 -dont_show -mjpeg_port 8090 -map`
- 186 MB Yolo9000 - image: `./darknet detector test cfg/combine9k.data cfg/yolo9000.cfg yolo9000.weights`
- Remember to put data/9k.tree and data/coco9k.map under the same folder of your app if you use the cpp api to build an app
- To process a list of images `data/train.txt` and save results of detection to `result.json` file use:
    `./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights -ext_output -dont_show -out result.json < data/train.txt`
- To process a list of images `data/train.txt` and save results of detection to `result.txt` use:
    `./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights -dont_show -ext_output < data/train.txt > result.txt`
- To process a video and output results to a json file use: `darknet.exe detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights file.mp4 -dont_show -json_file_output results.json`
- Pseudo-labelling - to process a list of images `data/new_train.txt` and save results of detection in Yolo training format for each image as label `<image_name>.txt` (in this way you can increase the amount of training data) use:
    `./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights -thresh 0.25 -dont_show -save_labels < data/new_train.txt`
- To calculate anchors: `./darknet detector calc_anchors data/obj.data -num_of_clusters 9 -width 416 -height 416`
- To check accuracy mAP@IoU=50: `./darknet detector map data/obj.data yolo-obj.cfg backup\yolo-obj_7000.weights`
- To check accuracy mAP@IoU=75: `./darknet detector map data/obj.data yolo-obj.cfg backup\yolo-obj_7000.weights -iou_thresh 0.75`

##### For using network video-camera mjpeg-stream with any Android smartphone

1. Download for Android phone mjpeg-stream soft: IP Webcam / Smart WebCam

    - Smart WebCam - preferably: https://play.google.com/store/apps/details?id=com.acontech.android.SmartWebCam2
    - IP Webcam: https://play.google.com/store/apps/details?id=com.pas.webcam

2. Connect your Android phone to the computer by WiFi (through a WiFi-router) or USB
3. Start Smart WebCam on your phone
4. Replace the address below, shown in the phone application (Smart WebCam) and launch:

- Yolo v4 COCO-model: `./darknet detector demo data/coco.data yolov4.cfg yolov4.weights http://192.168.0.80:8080/video?dummy=param.mjpg -i 0`
