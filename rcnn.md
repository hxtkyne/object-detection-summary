### R-CNN—深度学习物体检测的鼻祖

reference：《Rich feature hierarchies for Accurate Object Detection and Segmentation》

#### 一 简介

图片分类：不需要定位

物体检测：需要定位

**RCNN在物体检测上的解决方案**：

（1）找出图片中可能存在目标的候选区域

（2）通过CNN对候选区域提取特征向量

（3）在候选区域的特征向量上训练分类器，分类器用于判别物体的类别并能得到bbox

（4）修正bbox，回归微调bbox的位置

#### 二 解决方案详解

##### 2-1 确定候选区域

以往的方法是通过滑动窗口对寻找图片中可能存在的物体，例如人脸检测，就是利用人工提取的特征haar或者hog和滑动窗口方案实现。这种方法工作量大，耗时。RCNN采用选择搜索（selective search），即利用颜色，纹理等信息进行区域的提取和合并。需要注意的，提取出来的候选区域的长宽不固定，当输入cnn时，需要做resize。论文对图片做了两个方法的比较：

（1）各向异性缩放，即直接缩放到指定大小，这会造成不必要的图片失真

（2）各向同性缩放，在原图裁出候选区域，在边界用固定的背景颜色填充到指定大小

结论：作者发现采用各向异性缩放的实验精度最高。

##### 2-2 通过cnn对候选区域提取特征向量

* 网络结构设计（迁移学习）

（1）选择经典的Alexnet（论文采用）

（2）选择vgg16

*  微调

AlexNet的卷积部分可以作为一个好的特征提取器，后面的全连接层可以理解为一个分类器，然后进行微调训练。微调的细节：

每个mini-batch取128，分别为：

正样本：32，候选区域和图片物体标注区域的iou>0.5

负样本：96，候选区域和图片物体标注区域的iou<0.5

需要注意的是，在训练的时候，我们加上了一个分类层，训练完成后，我们会移除最后的分类层，直接提取到前面的全连接层，维度为4096维。

**问题**：为什么不直接用这个分类器而要添加一个svm分类器？

cnn在训练的时候，对训练数据做了宽松的标注，因此一个bounding box可能只包含物体的一部分，我们也把它标记为正样本，目的是为了防止cnn过拟合。然而svm使用于少量样本，所以需要对正样本的要求更加严格。

* 在候选区域的特征向量上训练分类器

假设上一步从图片中提取出来了2000个候选区域，那么提取出来的就是2000*4096的特征向量，然后用这些向量同时训练N个二分类的svm。训练svm时使用的样本时：

正样本：候选区域和图片物体标注区域的iou>0.7

负样本：候选区域和图片物体标注区域的iou<0.3

经过svm分类后，会输出一段的候选框得分（2000*20），20表示类别，这时候我们用非极大值抑制得到想要的候选框。

NMS的算法很简单，talk is cheap, show me the code !

```python
#coding:utf-8
import numpy as np

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
	#从大到小排列，取index
    order = scores.argsort()[::-1]
	#keep为最后保留的边框
    keep = []
    while order.size > 0:
		#order[0]是当前分数最大的窗口，之前没有被过滤掉，肯定是要保留的
        i = order[0]
        keep.append(i)
		#计算窗口i与其他所以窗口的交叠部分的面积
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
		#交/并得到iou值
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
		#ind为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
        inds = np.where(ovr <= thresh)[0]
		#下一次计算前要把窗口i去除，所有i对应的在order里的位置是0，所以剩下的加1
        order = order[inds + 1]
    return keep
```

* 回归微调修正bbox

使用一个简单的bounding-box回归用于提高定位的表现。bbox回归认为候选区域和gt之间是线性关系（因为在最后从SVM内确定出来的区域比较接近gt，所以近似认为是线性关系）。

从候选框到预测框是一个平移，放缩的过程。只有当候选框和真实框比较接近时（线性问题），才能将其作为训练样本我们的线性回归模型，否则会导致的回归模型不工作，离得远时，就是复杂的非线性问题了。

后续我们将rcnn的进化版！！！