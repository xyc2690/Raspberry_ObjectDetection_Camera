# Raspberry_ObjectDetection_Camera
Tensorflow ObjectDetectionAPI on Raspberry3B+, OpenCV, SSDmobile

## 1.Download SSDmobile model and Install Tensorflow  
<pre><code># Install TF1.8.0
pip install tensorflow-1.8.0-cp27-none-linux_armv7l.whl

# Get model
tar -xzvf ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz .
</code></pre>
[Link: TensorFlow on arm](https://github.com/lhelontra/tensorflow-on-arm/releases),

[Link: SSD mobile](http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz),

[Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

## 2.Install OpenCV for python
<p>pip install opencv-python
</p>

## 3.Test
By default, you will open the camera, display the images captured.

Modify the code:
<pre><code>TEST_CAM_ONLY = False</code></pre>

And run the Object Detection Demo.
<pre><code>python opencv_camera.py</code></pre>

Have Fun with Raspi!
