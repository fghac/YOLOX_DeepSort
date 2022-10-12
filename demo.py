from AIDetector_pytorch import Detector
import imutils
import cv2

def main():

    name = 'demo'

    det = Detector()
    cap = cv2.VideoCapture('./Test1.mp4') #读取已有视频序列
    fps = int(cap.get(5))             #获取帧速率
    print('fps:', fps)
    t = int(1000/fps)                 #设定延迟时间
    
    videoWriter = None

    while True:

        # try:
        _, im = cap.read()            #img返回捕获到的帧
        if im is None:
            break
        
        result = det.feedCap(im)      #***调用接口***其中 im 为 BGR 图像
        result = result['frame']      #返回的 result 是字典，result['frame'] 返回可视化后的图像
        result = imutils.resize(result, height=500)   #缩放图像的高度
        #print(result['faces'])
        #print(result['face_bboxes'] )
        if videoWriter is None:
            #创建视频编解码器，MPEG-4编码.mp4，可指定结果视频的大小
            fourcc = cv2.VideoWriter_fourcc(   
                'm', 'p', '4', 'v')  # opencv3.0
            #创建视频流写入对象，fourcc为视频编解码器，fps为帧播放速率，()元组为视频帧大小
            videoWriter = cv2.VideoWriter(
                'Result1.mp4', fourcc, fps, (result.shape[1], result.shape[0]))
            
        videoWriter.write(result) #写视频帧
        cv2.imshow(name, result)  #显示
   
        cv2.waitKey(t)            #延迟

        if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
            # 点x退出
            break
        # except Exception as e:
        #     print(e)
        #     break

    cap.release()              #释放视频流
    videoWriter.release()      #释放写入流
    cv2.destroyAllWindows()    #关闭所有窗口

if __name__ == '__main__':
    
    main()
    #