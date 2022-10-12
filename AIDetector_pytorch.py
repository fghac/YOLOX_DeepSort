from yolox.utils.boxes import postprocess
from yolox.data.data_augment import preproc
import torch
import torch.nn as nn
import numpy as np
from BaseDetector import baseDet
import os
from yolox.utils import fuse_model
from yolox.data.datasets import COCO_CLASSES


def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        # check availability
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        devices = device.split(',') if device else range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'

    return torch.device('cuda:0' if cuda else 'cpu')


class Detector(baseDet):

    def __init__(self):
        super(Detector, self).__init__()       #使用内置super方法调用基类的初始化方法
        
        self.build_config()
        #（yolov5s.yaml）mdepth和mwidth两个参数控制（层数和卷积数量）
        self.mdepth = 0.33#depth_multiple控制网络的深度
        self.mwidth = 0.50#width_multiple控制网络的宽度
        self.confthre=0.01#置信度阈值（conf的作用是删除置信度低的框）
        self.nmsthre=0.65 #nms的iou阈值（nms的作用是删除重复的框）
        self.test_size=(640, 640)#默认输入是640*640
        self.rgb_means = (0.485, 0.456, 0.406) #imagenet数据集RGB均值
        self.std = (0.229, 0.224, 0.225)       #imagenet数据集RGB标准差
        self.init_model()

    def init_model(self):

        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
        # 实现训练参数的初始化，BN的eps设为1e-3，momentum设为0.03
        def init_yolo(M): #函数嵌套定义，提高内聚性
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:#该对象中是否存在model属性，默认值为None，设置成模型
            in_channels = [256, 512, 1024]      #RGB通道数
            backbone = YOLOPAFPN(self.mdepth, self.mwidth, in_channels=in_channels)  #主干网络
            head = YOLOXHead(80, self.mwidth, in_channels=in_channels)               #网络头
            model = YOLOX(backbone, head)

        model.apply(init_yolo)
        model.head.initialize_biases(1e-2)
        self.weights = './weights/yolox_s.pth'
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        ckpt = torch.load(self.weights)
        # load the model state dict
        model.load_state_dict(ckpt["model"])#将内存的权重值加载到模型
        model.to(self.device).eval()
        model = fuse_model(model)
        self.m = model

        self.names =['person'] #COCO_CLASSES
        self.num_classes =1 #len(self.names)

    def preprocess(self, img):
        
        #创建帧信息字典
        img_info = {"id": 0}            #帧id
        img_info["file_name"] = None    #文件名
        height, width = img.shape[:2]   #取彩色图片的长宽
        img_info["height"] = height     #帧高度
        img_info["width"] = width       #帧宽度
        img_info["raw_img"] = img       #原帧
        #改变图像的shape为我们规定的shape(等比例缩放、其他地方填充灰色、并将图像归一化)
        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio       #获取图像的缩放比例（ratio：比例）
        #将数组形式转为张量，并扩充维度以满足运算要求
        img = torch.from_numpy(img).unsqueeze(0) #（将ndarray转tensor;0表示行扩充）
        if torch.cuda.is_available():
            img = img.cuda()

        return img_info, img

    def detect(self, im):

        img_info, img = self.preprocess(im)

        outputs = self.m(img)#outputs一般在几千个左右
        outputs = postprocess(#非极大值抑制，将多余的框去掉
                outputs, self.num_classes, self.confthre, self.nmsthre
            )[0]
        pred_boxes = []
        ratio = img_info["ratio"] #获取缩放比例
        img = img_info["raw_img"] #获取原始图像

        boxes = outputs[:, 0:4]   #获取预测框

        # preprocessing: resize
        boxes /= ratio            #将预测框还原成原图大小

        cls_ids = outputs[:, 6]   #获取类别
        scores = outputs[:, 4] * outputs[:, 5]   #获取置信度

        for i in range(len(boxes)):
            box = boxes[i].cpu()
            lbl = self.names[int(cls_ids[i])]
            conf = scores[i]
            #if lbl not in ['person']:
               # continue
            if conf < self.confthre:
                continue
            x1 = int(box[0])  #
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            pred_boxes.append(
                            (x1, y1, x2, y2, lbl, conf))

        return im, pred_boxes


if __name__ == '__main__':
    
    det = Detector()
    