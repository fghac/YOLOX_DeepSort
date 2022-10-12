from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from pyheatmap.heatmap import HeatMap
from PIL import Image, ImageFont, ImageDraw
import torch
import cv2
import math
import numpy as np

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)

def plot_bboxes(image, bboxes,lines, line_thickness=None):
    #print(bboxes)
   # Plots one bounding box on image img
    tl = line_thickness or round(#image.shape[0]:高度;image.shape[1]:宽度
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    for (x1, y1, x2, y2, cls_id, pos_id,tag) in bboxes:
        print("ID:",pos_id,"位置:","(",(x1+x2)/2,",",(y1+y2)/2,")")
        #print(x1,' ',x2, ' ',y2-y1)
        #if cls_id in ['person']:
        if tag==1:
            txt='risk'
            color = (0, 0, 255)  #BGR
        else:
            color = (0, 255, 0)
            txt='safe'
        c1, c2 = (x1, y1), (x2, y2)
        #调用cv2.rectangle()函数绘制矩形框
        cv2.rectangle(image, c1, c2, color, 2, cv2.LINE_AA)#LINE_AA:抗锯齿线
        tf = max(tl - 1, 1)  # font thickness 
        t_size = cv2.getTextSize(cls_id, 0, fontScale=tl / 3, thickness=tf)[0]#计算文本字符串的宽度和高度
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        #调用cv2.rectangle()函数绘制矩形框（用于写文字）
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, '{} ID-{}'.format(txt, pos_id), (c1[0], c1[1] - 2), 0, tl / 3,#cls_id
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    for u1,u2,v1,v2,min,po1,po2,tag in lines:
        if tag==1:
            ans="risk"
        else:
            ans="safe"
        print("ID1:",po1,"ID2:",po2,"ID1位置:","(",u1,",",u2,")","ID2位置:","(",v1,",",v2,")","距离:",min,ans)
        if tag==1:
            color=(0,0,255)
            '''
            
       `    else:
            color=(0,255,0)
            '''
            cv2.circle(image,(u1,u2),2,(255,0,0),thickness=4)
            cv2.circle(image,(v1,v2),2,(255,0,0),thickness=4)
            cv2.line(image,(u1,u2),(v1,v2),color,thickness=tf,lineType=cv2.LINE_AA)
    print('\n')
    return image
def plot_heatmap(img,boxes):
    sum=len(boxes)
    data=[]
    for i in range(sum):
        x1, y1, x2, y2, cls1, pos1=boxes[i]
        tmp=[(x1+x2)/2,y2,1]
        data.append(tmp)
        heat=HeatMap(data)
        #img=heat.heatmap(img,cv2.COLORMAP_JET)
        #img=cv2.applyColorMap(np.uint8(heat),cv2.COLORMAP_JET)
        cv2.addWeighted(img.copy(),0.7,img,1-0.7,0) # 将热度图覆盖到原图
    return img
def use_heatmap(image,boxes ):
    sum=len(boxes)
    data=[]
    for i in range(sum):
        #x1, y1, x2, y2, cls1, pos1=boxes[i]
        for (x1, y1, x2, y2, cls_id, pos_id,tag) in boxes:
            if tag==1:
                tmp=[int((x1+x2)/2),int(y2),1]
                data.append(tmp)
    background = Image.new("RGB", (image.shape[1], image.shape[0]), color=0)
    # 开始绘制热度图
    hm = HeatMap(data)
    hit_img = hm.heatmap(base=background, r = 200) # background为背景图片，r是半径，默认为10
    # ~ plt.figure()
    # ~ plt.imshow(hit_img)
    # ~ plt.show()
    #hit_img.save('out_' + image_name + '.jpeg')
    hit_img = cv2.cvtColor(np.asarray(hit_img),cv2.COLOR_RGB2BGR)#Image格式转换成cv2格式cv2.COLORMAP_JET
    overlay = image.copy()
    alpha = 0.3 # 设置覆盖图片的透明度
    #cv2.rectangle(overlay, (0, 0), (image.shape[1], image.shape[0]), (255, 0, 0), -1) # 设置蓝色为热度图基本色蓝色
    #image = cv2.addWeighted(overlay, alpha, image, 1-alpha, 0) # 将背景热度图覆盖到原图
    image = cv2.addWeighted(hit_img, alpha, image, 1-alpha, 0) # 将热度图覆盖到原图
    return image
    '''sum=len(boxes)
    box_centers=[]
    for i in range(sum):
        x1, y1, x2, y2, cls1, pos1=boxes[i]
        tmp=[(x1+x2)/2,y2]
        box_centers.append(tmp)
    hm = HeatMap(box_centers)
    box_centers = [(i, image.shape[0] - j) for i, j in box_centers]
    img = hm.heatmap(box_centers, dotsize=200, size=(image.shape[1], image.shape[0]), opacity=128, area=((0, 0), (image.shape[1], image.shape[0])))
    return img'''



def cal_distance(boxes):
    #bboxes2draw:(x1, y1, x2, y2, cls_id, pos_id)
    boxes_draw=[]
    i=0
    j=0
    tag=0
    sum=len(boxes)
    for i in range(sum):
        tag=0
        distance=110
        x1, y1, x2, y2, cls1, pos1=boxes[i]
        for j in range(sum):
            if i==j:
                continue
            m1,n1,m2,n2, cls2, pos2=boxes[j]
            o1=(x1+x2)/2
            o2=(y1+y2)/2
            p1=(m1+m2)/2
            p2=(n1+n2)/2
            distance=math.sqrt(math.pow(o1-p1,2)+math.pow(o2-p2,2))
            if distance<98.63:#####***************不同场景修改/瑶海区：69.2；庐阳区：98.63/***************###############
                tag=1
                break
        boxes_draw.append((x1,y1,x2,y2,cls1,pos1,tag))
        #print(boxes_draw)
    
    #bboxes2draw:(x1, y1, x2, y2, cls_id, pos_id)
    lines_draw=[]
    sum=len(boxes)
    for i in range(sum):
        tag=0
        x1, y1, x2, y2, cls1, pos1=boxes[i]
        min=9999
        TT=0
        for j in range(sum):
            if i==j:
                continue
            m1,n1,m2,n2, cls2, pos2=boxes[j]
            o1=int((x1+x2)/2)
            o2=int((y1+y2)/2)
            p1=int((m1+m2)/2)
            p2=int((n1+n2)/2)
            distance=math.sqrt(math.pow(o1-p1,2)+math.pow(o2-p2,2))
            if(min>distance):
                TT=1
                min=distance
                u1,u2,v1,v2,po1,po2=o1,o2,p1,p2,pos1,pos2
        if(min<98.63):#####***************不同场景修改/瑶海区：69.2；庐阳区：98.63/***************###############
            tag=1
        if(TT==1 ):#and min<100
            la=0
            for x in range(len(lines_draw)):
                a,b,c,d,m,pp1,pp2,t=lines_draw[x]
                if pp1==po2 and pp2==po1:
                    la=1
                    break
            if la==0:
                min=0.35/34.52*min#####***************不同场景修改/瑶海区：min=1.632/112.92*min；庐阳区：/***************###############
                lines_draw.append((u1,u2,v1,v2,min,po1,po2,tag))
    return boxes_draw,lines_draw
def update_tracker(target_detector, image):
    
    new_faces = [] #新检测到的目标
    _, bboxes = target_detector.detect(image)
    
    bbox_xywh = [] #矩形框中心坐标(x,y)、宽、高
    confs = []     #置信度
    clss = []      #类别

    for x1, y1, x2, y2, cls_id, conf in bboxes:

        obj = [
            int((x1+x2)/2), int((y1+y2)/2),
            x2-x1, y2-y1
        ]
        bbox_xywh.append(obj)
        confs.append(conf)
        clss.append(cls_id)

    xywhs = torch.Tensor(bbox_xywh) #单精度浮点数类型张量
    confss = torch.Tensor(confs)   
    #对该跟踪对象进行坐标更新
    outputs = deepsort.update(xywhs, confss, clss, image)

    bboxes2draw = []
    face_bboxes = []
    current_ids = []
    for value in list(outputs):
        x1, y1, x2, y2, cls_, track_id = value #track_id:跟踪对象的id号
        bboxes2draw.append(
            (x1, y1, x2, y2, cls_, track_id)
        )
        current_ids.append(track_id)
        if cls_ == 'face':
            if not track_id in target_detector.faceTracker:
                target_detector.faceTracker[track_id] = 0
                face = image[y1:y2, x1:x2]
                new_faces.append((face, track_id))
            face_bboxes.append(
                (x1, y1, x2, y2)
            )

    ids2delete = []
    for history_id in target_detector.faceTracker:
        if not history_id in current_ids:
            target_detector.faceTracker[history_id] -= 1
        if target_detector.faceTracker[history_id] < -5:
            ids2delete.append(history_id)

    for ids in ids2delete:
        target_detector.faceTracker.pop(ids)
        print('-[INFO] Delete track id:', ids)
    distance=[]
    boxxes,lines=cal_distance(bboxes2draw)
    image = plot_bboxes(image, boxxes,lines)#bboxes2draw:(x1, y1, x2, y2, cls_id, pos_id)
    #imag = plot_heatmap(image,bboxes)
    #image =use_heatmap(image,boxxes)
    return image, new_faces, face_bboxes
    