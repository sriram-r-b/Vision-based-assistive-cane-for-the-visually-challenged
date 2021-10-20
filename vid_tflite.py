import time
import tflite_runtime.interpreter as tflite
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
#from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np

class Flags1:
    def __init__(self):
        self.framework='tflite'
        self.weights='./checkpoints/yolov4-320.tflite'
        self.size=320
        self.tiny= False#, 'yolo or yolo-tiny')
        self.model='yolov4'#, 'yolov3 or yolov4')
        self.video= "./data/video/video.mp4"#, 'path to input image')
        self.output= './detections/res.avi' # 'path to output folder')
        self.output_format= 'XVID'
        self.iou= 0.45
        self.score= 0.25
        self.dont_show= False
        self.iscam=False # Boolean- denotes if you use cam
    def myfunc(self):        
        print("Hello my func ")
        
        
FLAGS=Flags1()
def rectarea(hmin,hmax,wmin,wmax):
    return ((hmax-hmin)*(wmax-wmin))
def test():
    #config = ConfigProto()
    #config.gpu_options.allow_growth = True
    #session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    if FLAGS.framework == 'tflite':
        interpreter = tflite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    

    # begin video capture
    if FLAGS.iscam:
        vid = cv2.VideoCapture(int(video_path))
    else:
        vid = cv2.VideoCapture(video_path)

    out1 = list()
    count=0

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
    
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        #crop= image.copy()
        #print(crop.shape)
        image_data = image_data / 255.
        image_h, image_w, _ = frame.shape
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

        image = utils.draw_bbox(frame, pred_bbox)
        fps = 1.0 / (time.time() - start_time)
        count=count+1
        print("FPS: %.2f" % fps,"  count  ",count)
        result = np.asarray(image)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #cv2.namedWindow("crop", cv2.WINDOW_AUTOSIZE)
        valid_detections=pred_bbox[3][0]
        out1=list()
        classe=pred_bbox[2][0].astype(int)[: valid_detections]
        for i in range(valid_detections):
            coor = pred_bbox[0][0][i].astype(int)
            scores=( pred_bbox[1][0]*100).astype(int)[i]
            classe=( pred_bbox[2][0]).astype(int)[i]
            hmin= int(coor[0])
            hmax = int(coor[2])
            wmin = int(coor[1])
            wmax = int(coor[3])
            area=rectarea(hmin,hmax,wmin,wmax)
            sc1=[((wmin/image_w)>0.29),((wmax/image_w) < 0.75),(classe == 0),(((hmax-hmin)/image_h)>0.55),(wmax,image_w),((wmin,image_w))]
            #if ( (count in range(137,157)) or (count in range(246,257))or (count in range(291,318)) )and (classe==0):
                # print(sc1)
                # print(scores)
                # print(classe)
                # print(count,"count")
                # print(i,"i")
                
            criteria1=sc1[0] and sc1[1] and sc1[2] and sc1[3]  and (area>20500) 
            criteria2=(((wmax- wmin)/image_w)>0.6) 
            criteria3=((((hmax-hmin)/image_h)>0.6)and ((coor[1]/image_w)>0.35) and ((coor[3]/image_w)<0.75))
            
            #print(criteria1 , criteria2 , criteria3 )
            if  criteria1 or criteria2 or criteria3 or (area>100000):
                out1.append([coor,count])
                #cv2.imwrite( './detections/hit/' + str(count) + '.png', image)
                print( hmin,hmax,wmin,wmax, "hmin:hmax,wmin:wmax",count , i  )
            
                
        #crop1=crop[hmin:hmax,wmin:wmax]
        
        #result = cv2.cvtColor(crop1, cv2.COLOR_RGB2BGR)
        cv2.imwrite( './detections/frames/' + str(count) + '.png', image)
        
        if not FLAGS.dont_show:
            cv2.imshow("result", result)
            #cv2.imshow("crop", crop1)
        
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()
    #session.close()
    return out1

if __name__ == '__main__':
    try:
        a=test()
    except SystemExit:
        pass
