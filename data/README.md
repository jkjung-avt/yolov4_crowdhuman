# CrowdHuman Dataset by MEGVII

* Official web site: [https://www.crowdhuman.org/](https://www.crowdhuman.org/)

* Reference:
   - [CrowdHuman: A Benchmark for Detecting Human in a Crowd](https://arxiv.org/abs/1805.00123)
   - [CrowdHuman Dataset 介紹](https://chtseng.wordpress.com/2019/12/13/crowdhuman-dataset-%E4%BB%8B%E7%B4%B9/)

* When converting CrowdHuman annotations to YOLO txt files,
   - I discard all "mask" objects.  The "mask" objects in the CrowdHuman dataset are not real humans.  They are usually reflections of humans, or pictures of humans in billboards or advertisement posters.
   - I use "hbox" (head) and "fbox" (full body) annotations of all "person" objects.  Note that the "fbox" annotation might include body parts which are "ocluded" in the scene.
   - In the final YOLO txt files, there are 2 classes of objects.  Class 0 is "head", and class 1 "person".
