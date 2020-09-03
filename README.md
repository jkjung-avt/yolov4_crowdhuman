YOLOv4 CrowdHuman Tutorial
==========================

This is a tutorial demonstrating how to train a YOLOv4 people detector using [Darknet](https://github.com/AlexeyAB/darknet) and the [CrowdHuman dataset](https://www.crowdhuman.org/).

NOTE: This is **still a work in progress**...

Table of contents
-----------------

* [Setup](#setup)
* [Preparing training data locally](#preparing)
* [Training on a local PC](#training-locally)
* [Training on Coogle Colab](#training-colab)
* [Testing the custom-trained yolov4 model](#testing)
* [Deploying onto Jetson Nano](#deploying)

<a name="setup"></a>
Setup
-----

If you are going to train the model on [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb), you could skip this Setup section and jump straight to [Training on Coogle Colab](#training-colab).

Otherwise, to run training locally, you need to have a x86_64 PC with a decent GPU.  Just for reference, I mainly test the code in this repository using a desktop PC with:

* NVIDIA GeForce RTX 2080 Ti
* Ubuntu 18.04.5 LTS (x86_64)
    - CUDA 10.2
    - cuDNN 8.0.1

Also make sure python3 "cv2" (opencv) module is installed properly on your local PC since the data preprocessing code would require it.

<a name="preparing"></a>
Preparing training data locally
-------------------------------

For training on a local PC, I use a "608x608" yolov4 model as example.  Note that I use python3 exclusively in this tutorial (python2 might not work).  Follow these steps to prepare CrowdHuman dataset for training the yolov4 model.

1. Clone this repository.

   ```shell
   $ cd ${HOME}/project
   $ git clone https://github.com/jkjung-avt/yolov4_crowdhuman
   ```

2. Run the "prepare_rawdata.sh" script in the "data/" subdirectory.  It would download CrowdHuman dataset files and unzip all train/val image files.  You could refer to [data/README.md](data/README.md) for more information about the dataset.

   ```shell
   $ cd ${HOME}/project/yolov4_crowdhuman/data
   $ ./prepare_rawdata.sh 608x608
   ```

   This step could take quite a while, depending on your internet speed.  When it's done, all train/val image files would be located in "data/crowdhuman-608x608/" and the original annotation files, "annotation_train.odgt" and "annotation_val.odgt", would be in "data/raw/".

3. Convert the annotation files to YOLO txt format.  Please refer to [How to train (to detect your custom objects)](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects) for an explanation of YOLO txt files.

   ```shell
   $ python3 gen_txts.py 608x608
   ```

   The "gen_txts.py" script would output all necessary ".txt" files in the "data/crowdhuman-608x608/" subdirectory.  At this point, you should have all custom files needed to train the "yolov4-crowdhuman-608x608" model.  (I've also created a "verify_txts.py" script for verifying the generated txt files.)

   In this tutorial, you'd be training the yolov4 model to detect 2 classes of object: "head" (0) and "person" (1), where the "person" class corresponds to "full body" (including occluded body portions) in the original CrowdHuman annotations.  Take a look at "data/crowdhuman-608x608.data", "data/crowdhuman.names", and "data/crowdhuman-608x608/" to gain a better understanding of the data files that have been generated/prepared for the training.

   ![A sample jpg from the CrowdHuman dataset](doc/crowdhuman_sample.jpg)

<a name="training-locally"></a>
Training on a local PC
----------------------

Continuing from steps in the previous section, you'd be using the "darknet" framework to train the yolov4 model.

1. Download and build "darknet" code.  (TODO: Consider making "darknet" as a submodule and automate the build process?)

   ```shell
   $ cd ${HOME}/project/yolov4_crowdhuman
   $ git clone https://github.com/AlexeyAB/darknet.git
   $ cd darknet
   $ vim Makefile  # edit Makefile with a editor of your own preference
   ```

   Modify the first few lines of the "Makefile" as follows.  Please refer to [How to compile on Linux (using make)](https://github.com/AlexeyAB/darknet#how-to-compile-on-linux-using-make) for more information about these settings.  Note that CUDA compute "75" is for RTX 2080 Ti and "61" for GTX 1080.  You might need to modify those based on the kind of GPU you are using.

   ```
   GPU=1
   CUDNN=1
   CUDNN_HALF=1
   OPENCV=1
   AVX=1
   OPENMP=1
   LIBSO=1
   ZED_CAMERA=0
   ZED_CAMERA_v2_8=0

   ......

   USE_CPP=0
   DEBUG=0

   ARCH= -gencode arch=compute_61,code=[sm_61,compute_61] \
         -gencode arch=compute_75,code=[sm_75,compute_75]

   ......
   ```

   Then do a `make` to build "darknet".

   ```shell
   $ make
   ```

   When it is done, I would suggest to test the "darknet" executable with the `test` command as follows.

   ```shell
   ### download pre-trained yolov4 coco weights and test with the dog image
   $ wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights \
          -q --show-progress --no-clobber
   $ ./darknet detector test cfg/coco.data cfg/yolov4-416.cfg yolov4.weights \
                             data/dog.jpg
   ```

2. Copy over all files needed for training and download the pre-trained weights ("yolov4.conv.137").

   ```shell
   $ cd ${HOME}/project/yolov4_crowdhuman
   $ ./prepare_training.sh 608x608
   ```

3. Train the model.  Please refer to [How to train with multi-GPU](https://github.com/AlexeyAB/darknet#how-to-train-with-multi-gpu) for how to fine-tune your training process.  For example, you could specify `-gpus 0,1,2,3` in order to use multiple GPUs to speed up training.

   ```shell
   $ cd ${HOME}/project/yolov4_crowdhuman/darknet
   $ ./darknet detector train data/crowdhuma-608x608.data \
                              cfg/yolov4-crowdhuman-608x608.cfg \
                              yolov4.conv.137 -map -gpu 0
   ```

   You could monitor training progress on the loss/mAP chart (since the `-map` option is used).  Alternatively, if you are training on a remote PC via ssh, add the `-dont_show -mjpeg_port 8090` option so that you could monitor the loss/mAP chart on a web browser (http://{IP address}:8090/).

   ```
   ### alternatively, if training on an ssh'ed remote PC
   $ ./darknet detector train data/crowdhuman-608x608.data \
                              cfg/yolov4-crowdhuman-608x608.cfg \
                              yolov4.conv.137 -map -gpu 0 \
                              -dont_show -mjpeg_port 8090
   ```

   Training this "yolov4-crowdhuman-608x608" model with my RTX 2080 Ti GPU takes 17~18 hours.  I'm able to get a model with rather higher mAP (mAP@0.5 = ~76%).  (TODO:  Add the loss/mAP chart.)

   ![My sample loss/mAP chart of the "yolov4-crowdhuman-608x608" model](doc/chart_yolov4-crowdhuman-608x608.png)

<a name="training-colab"></a>
Training on Coogle Colab
------------------------

For training on Google Colab, I use a "416x416" yolov4 model as example.  I have put all data processing and training commands into an IPython Notebook.  So training the "yolov4-crowdhuman-416x416" model on Google Colab is just as simple as: (1) opening the Notebook on Google Colab, (2) mount your Google Drive, (3) run all cells in the Notebook.

A few words of caution before you begin running the Notebook on Google Colab:

* Although GPU runtime is *free of charge* on Google Colab, it is not unlimited nor guaranteed.  Even though "virtual machines that have maximum lifetimes that can be as much as 12 hours" is stated in Colab [FAQ](https://research.google.com/colaboratory/faq.html#resource-limits), I often saw my Colab session got disconnected after 7~8 hours of non-interactive use.

* If you repeatedly connect to GPU instances on Google Colab, you could be temporarily locked out (not able to connect to GPU instances for a couple of days).  So I'd suggest you to connect to a GPU runtime only when needed, and to manually terminate the GPU session when you no longer need it.

* It is strongly advised that you read [Resource Limits](https://research.google.com/colaboratory/faq.html#resource-limits) and use GPU instances on Google Colab wisely.

Due to the 7~8 hour limit of runtime mentioned above, you won't be able to train a large yolov4 model in 1 single session.  That's the reason why I chose to do "416x416" model here.

There are 2 ways for you to open the "yolov4_crowdhuman.ipynb" Notebook.  You could either open [yolov4_crowdhuman.ipynb on my Colab account](https://colab.research.google.com/drive/1YM0btXPz_ECJZg1veb_2CBGJmnLJuswP?usp=sharing) and make a copy of your own, or download [yolov4_crowdhuman.ipynb on GitHub](yolov4_crowdhuman.ipynb) and use "File -> Upload notebook" to open it on [your own Colab acount](https://colab.research.google.com/notebooks/intro.ipynb).

Next, follow the instructions in the Notebook, i.e. mount your Google Drive (for saving training log and weights) and then run all cells.  You should have a good chance of finishing training the "yolov4-crowdhuman-416x416" model before the Colab session gets automatically disconnected (expired).

<a name="testing"></a>
Testing the custom-trained yolov4 model
---------------------------------------

If you have trained the "yolov4-crowdhuman-608x608" model locally, it is very easy to test the custom-trained model with "darknet".

   ```shell
   $ cd ${HOME}/project/yolov4_crowdhuman/darknet
   $ ./darknet detector test data/crowdhuma-608x608.data \
                             cfg/yolov4-crowdhuman-608x608.cfg \
                             backup/yolov4-crowdhuman-608x608_best.weights \
                             data/crowdhuman-608x608/273275,4e9d1000623d182f.jpg \
                             -gpu 0
   ```

Otherwise, for the "yolov4-crowdhuman-416x416" model trained on Google Colab, you'll need to:

* build "darknet" locally,
* generate the 2 files "data/crowdhuma-416x416.data" (modify from [crowdhuman-template.data](https://github.com/jkjung-avt/yolov4_crowdhuman/blob/master/data/crowdhuman-template.data)) and "cfg/yolov4-crowdhuman-416x416.cfg" (copy from [yolov4-crowdhuman-416x416.cfg](https://github.com/jkjung-avt/yolov4_crowdhuman/blob/master/cfg/yolov4-crowdhuman-416x416.cfg),
* download "backup/yolov4-crowdhuman-416x416_best.weights" from the "yolov4_crowdhuman" directory on your Google Drive,
* prepare an image file for testing.

Then go to your local "darknet" directory and do:

   ```shell
   $ ./darknet detector test data/crowdhuma-416x416.data \
                             cfg/yolov4-crowdhuman-416x416.cfg \
                             backup/yolov4-crowdhuman-416x416_best.weights \
                             ${HOME}/Pictures/sample.jpg \
                             -gpu 0
   ```

<a name="deploying"></a>
Deploying onto Jetson Nano
--------------------------

* [yolov4-crowdhuman-416x416.cfg](https://github.com/jkjung-avt/yolov4_crowdhuman/blob/master/cfg/yolov4-crowdhuman-416x416.cfg)
* download "backup/yolov4-crowdhuman-416x416_best.weights" from the "yolov4_crowdhuman" directory on your Google Drive,
* Build TensorRT engine and run inference with [tensorrt_demos](https://github.com/jkjung-avt/tensorrt_demos)
