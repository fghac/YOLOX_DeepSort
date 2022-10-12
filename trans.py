import numpy as np
import cv2
import glob

def imgs2video(out_dir,video_name,fps,img_w,img_h):
    #fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
    out = cv2.VideoWriter(out_dir+video_name, fourcc, fps, (img_w,img_h),True)
    return out


def main():
    img_dir="E:\\trainData\\video\\1\\img\\" #图片保存路径
    video_dir="./" 
    

    i = 0
    for imgs in glob.glob(img_dir+"*.jpg"):
        frame=cv2.imread(imgs)
        if i==0:
            h,w,_ = frame.shape #获取一帧图像的宽高信息
            out = imgs2video(video_dir,"test.avi",20.0,w,h) #按要求创建视频流
        i += 1
        print('i = ',i)
        out.write(frame)#对视频文件写入一帧
     
    #释放视频流
    out.release()
    
if __name__ == "__main__":
    main()
