import os
import cv2
import numpy as np
import ast
from timeit import default_timer as timer
from keras import backend as K
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from model import evaluation, yolo_body

right_clicks = list()
polygon_list=[]
Polygon_object_list=[]
polygon_area_list=[]
polygon_color_list=[(153,0,76), (0,204,204), (255,153,153), (102,204,0), (102,0,102)]
poly_index=0

def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.__dict__.update(kwargs)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        start = timer()
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        self.yolo_model = yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
        self.yolo_model.load_weights(self.model_path)
        end = timer()
        print('{} model, anchors, and classes loaded in {:.2f}sec.'.format(model_path, end-start))
        self.colors = ['GreenYellow']
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = evaluation(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image,itr_number,th_mode,car_for_each_polygon_list,polygon_density,pixel_to_dist_ratio,polygon_dist_list_vel_mode,video_fps,velocity_and_view_time,th_low,th_high):
        start = timer()
        num_of_frames_for_mean=10
        number_of_point_in_polygons=np.zeros((1,len(polygon_list)),dtype=int)
        vehicles_area_in_polygon = np.zeros((1, len(polygon_list)), dtype=float)
        number_of_frames=np.zeros((1,len(polygon_list)),dtype=int)
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })


        if(itr_number==1):
            for i in range(len(polygon_list)):
                pts = np.array(right_clicks[np.sum(polygon_list[0:i], dtype=int):np.sum(polygon_list[0:i], dtype=int) + polygon_list[i]], np.int32)
                a1 = np.empty((polygon_list[i],), dtype=object)
                a1[:] = [tuple(j) for j in pts]
                polygon = Polygon(a1.tolist())
                polygon_area_list.append(polygon.area)
                Polygon_object_list.append(polygon)
        for i in range(len(polygon_list)):
             pts = np.array(right_clicks[np.sum(polygon_list[0:i],dtype=int):np.sum(polygon_list[0:i],dtype=int)+polygon_list[i]], np.int32)
             image=cv2.polylines(np.array(image), [pts], True, polygon_color_list[i],thickness=2)

        image = Image.fromarray(image)
        out_prediction = []

        font_path = os.path.join(os.path.dirname(__file__),'font/FiraMono-Medium.otf')
        font = ImageFont.truetype(font=font_path,
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 400

        for i, c in reversed(list(enumerate(out_classes))):
            box = out_boxes[i]
            curr_box_x_center=(box[3]+box[1])/2
            curr_box_y_center=(box[2]+box[0])/2
            point=Point(curr_box_x_center,curr_box_y_center)
            index=np.where([poly.contains(point) for poly in Polygon_object_list])[0]
            if len(index)==0:
                continue

            number_of_point_in_polygons[0,index]+=1
            curr_area=(box[3]-box[1])*(box[2]-box[0])
            if 2400<curr_area<=4500:
                curr_area*=0.8
            elif (4500<curr_area<=9000):
                curr_area *= 0.7
            elif (9000<curr_area<=13000):
                 curr_area *= 0.6
            elif (13000<curr_area<=18000):
                curr_area *= 0.5
            elif (curr_area>18000):
                curr_area *= 0.4
            vehicles_area_in_polygon[0,index]+=curr_area
            score = out_scores[i]

            draw = ImageDraw.Draw(image)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            if top > image.size[1] or right > image.size[0]:
                continue

            out_prediction.append([left, top, right, bottom, c, score])
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=polygon_color_list[index[0]])
            del draw
        car_for_each_polygon_list.insert(0,number_of_point_in_polygons[0])
        polygon_density.insert(0,vehicles_area_in_polygon[0] / polygon_area_list)
        if th_mode=="velocity" and itr_number>1:
            for p in range(len(polygon_list)):
                if car_for_each_polygon_list[0][p]==0 and car_for_each_polygon_list[1][p]==1:
                    number_of_frames[0,p]=np.where(np.array(car_for_each_polygon_list)[1:, p] == 0)[0][0]
                    velocity_and_view_time[0,p]=video_fps
                    velocity_and_view_time[1,p] = (np.array(polygon_dist_list_vel_mode)[p] * np.array(pixel_to_dist_ratio)[p] * 3.6) / (number_of_frames[0][p] / video_fps)


        if(itr_number>=num_of_frames_for_mean):
            mean_of_points_in_polygon=np.round(np.transpose(car_for_each_polygon_list)[:,0:num_of_frames_for_mean].sum(axis=1)/num_of_frames_for_mean).astype(int)
            mean_polygon_density=np.sum(polygon_density[0:10],axis=0)/num_of_frames_for_mean
        else:
            mean_of_points_in_polygon =number_of_point_in_polygons[0]
            mean_polygon_density=polygon_density[-1]
        draw = ImageDraw.Draw(image)
        font_number_of_vehicles = font
        font_number_of_vehicles.size = 40
        rectangle_width=int(image.size[0] / 7)
        space_between_rect=0
        if len(polygon_list)>1:
            space_between_rect = int((image.size[0] - len(polygon_list)*rectangle_width-40)/(len(polygon_list)-1))
        if th_mode == "counting":
            mean_polygon=mean_of_points_in_polygon
        elif th_mode == "density":
            mean_polygon = mean_polygon_density
        else:
            mean_polygon = velocity_and_view_time[1,:]

        for c in range(len(polygon_list)):
            R,G,B=color_result(mean_polygon[c],th_low,th_high)
            draw.rectangle([tuple([10+c*(rectangle_width+space_between_rect), 60]), tuple([10 + c*(rectangle_width+space_between_rect)+rectangle_width, 60 + 40])], fill=(R, G, B))
            draw.rectangle([tuple([10 + c * (rectangle_width + space_between_rect)+rectangle_width, 60]),tuple([10 + c * (rectangle_width + space_between_rect)+rectangle_width+20, 60 + 40])],fill=polygon_color_list[c])
            if th_mode == "counting":
                draw.text([10+c*(rectangle_width+space_between_rect), 65], "vehicles:" + str(mean_polygon[c]), fill=(0, 0, 0),font=font_number_of_vehicles)
            elif th_mode == "density":
                draw.text([10 + c * (rectangle_width + space_between_rect), 65],'density:' + str(int(mean_polygon[c] * 100)) + '%', fill=(0, 0, 0), font=font_number_of_vehicles)
            else:
                if (mean_polygon[c]!=0) or (velocity_and_view_time[0,c]>0):
                    draw.text([10 + c * (rectangle_width + space_between_rect), 65],'velocity:' + str(mean_polygon[c]) + 'kmh', fill=(0, 0, 0),font=font_number_of_vehicles)
                    velocity_and_view_time[0,c]-=1
                    if velocity_and_view_time[0,c] ==0:
                        velocity_and_view_time[1,c]=0
                else:
                    draw.text([10 + c * (rectangle_width + space_between_rect), 65], 'velocity:', fill=(0, 0, 0),font=font_number_of_vehicles)
        del draw
        end = timer()
        fps=round(1/(end - start))
        return fps, out_prediction, image

    def close_session(self):
        self.sess.close()

def detect_video(yolo, video_path,th_mode,th_low,th_high, define_regions, output_path="",input_path=""):
    car_for_each_polygon_list=[]
    polygon_density=[]
    pixel_to_dist_ratio=[]
    polygon_dist_list_vel_mode=[]
    velocity_and_view_time = np.zeros((2, len(polygon_color_list)), dtype=int)
    vid = cv2.VideoCapture(video_path)
    file_name = input_path[:input_path.rfind(".")] + ".txt"

    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = cv2.VideoWriter_fourcc(*'mp4v')
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print('Processing {} with frame size {} '.format(os.path.basename(video_path), video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    n=0
    first_frame=True
    global poly_index
    poly_index=0
    while vid.isOpened():
        n+=1
        return_value, frame = vid.read()

        if first_frame==True:
            if define_regions == 1:
                if th_mode=="velocity":
                    cv2.putText(frame, "Please define velocity regions (BR,BL,TL,TR) order", (int(frame.shape[1] / 4), 40), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 0, 0), 2)
                else:
                    cv2.putText(frame,"Please define regions",(int(frame.shape[1]/3),40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
                cv2.namedWindow('first_frame', cv2.WINDOW_NORMAL)
                cv2.setMouseCallback('first_frame', Mouse_Callback,param=frame)
                while (1):
                    cv2.imshow('first_frame', frame/255)
                    k=cv2.waitKey(10)
                    if k==97:
                        polygon_list.append(int(len(right_clicks)-np.sum(polygon_list)))
                        pts = np.array(right_clicks[ np.sum(polygon_list[0:poly_index], dtype=int):np.sum(polygon_list[0:poly_index],dtype=int) + polygon_list[poly_index]], np.int32)
                        B, G, R = polygon_color_list[poly_index]
                        cv2.polylines(frame, [pts], True, (R,G,B), thickness=2)
                        if th_mode == "velocity":
                            polygon_dist_list_vel_mode.append(np.linalg.norm( ((pts[0,:]+pts[1,:])/2) - ((pts[2,:]+pts[3,:])/2)))
                            cv2.imshow('first_frame', frame / 255)
                            k = cv2.waitKey(10)
                            pixel_to_dist_ratio.append(float(input('Please insert pixel to real distance ratio:')))
                        poly_index+=1
                    if k == 27:
                        polygon_list.append(int(len(right_clicks) - np.sum(polygon_list)))
                        pts = np.array(right_clicks[np.sum(polygon_list[0:poly_index], dtype=int):np.sum(polygon_list[0:poly_index],dtype=int) + polygon_list[poly_index]], np.int32)
                        B, G, R = polygon_color_list[poly_index]
                        cv2.polylines(frame, [pts], True, (R, G, B), thickness=2)
                        cv2.imshow('first_frame', frame / 255)
                        cv2.waitKey(10)
                        if th_mode == "velocity":
                            polygon_dist_list_vel_mode.append(np.linalg.norm(((pts[0, :] + pts[1, :]) / 2) - ((pts[2, :] + pts[3, :]) / 2)))
                            pixel_to_dist_ratio.append(float(input('Please insert pixel to real distance ratio:')))
                        cv2.destroyAllWindows()
                        break

                with open(file_name, "w") as txt_file:
                    txt_file.write(str(right_clicks)+'\n')
                    txt_file.write(str(polygon_list))
                    if th_mode=="velocity":
                        txt_file.write('\n'+str(pixel_to_dist_ratio)+'\n')
                        txt_file.write(str(polygon_dist_list_vel_mode))
                txt_file.close()

            else:
                with open(file_name, "r") as txt_file:
                    right_clicks.extend(ast.literal_eval(txt_file.readline()))
                    polygon_list.extend(ast.literal_eval(txt_file.readline()))
                    if th_mode=="velocity":
                        pixel_to_dist_ratio.extend(ast.literal_eval(txt_file.readline()))
                        polygon_dist_list_vel_mode.extend(ast.literal_eval(txt_file.readline()))

            first_frame = False

        if not return_value:
            break
        frame = frame[:,:,::-1]
        image = Image.fromarray(frame)
        fps_, out_pred, image = yolo.detect_image(image,n,th_mode,car_for_each_polygon_list,polygon_density,pixel_to_dist_ratio,polygon_dist_list_vel_mode,video_fps,velocity_and_view_time,th_low,th_high)
        result = np.asarray(image)
        fps = "FPS: " + str(fps_)
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)

        if isOutput:
            out.write(result[:,:,::-1])

    vid.release()
    out.release()


def Mouse_Callback(event, x, y, flags ,params):
    if event==cv2.EVENT_LBUTTONDBLCLK:
        global right_clicks
        right_clicks.append([x,y])
        B,G,R=polygon_color_list[poly_index]
        if poly_index==0:
            pts=np.array(right_clicks)
        else:
            pts = np.array(right_clicks[np.sum(polygon_list[0:poly_index], dtype=int):np.sum(polygon_list[0:poly_index], dtype=int) + len(right_clicks)],np.int32)

        cv2.polylines(params, [pts], False, (R,G,B), thickness=2)


def color_result(value,th_low,th_high):
    th_mid=(th_low+th_high)/2
    delta=(th_high-th_low)/2
    B=0
    if value<th_mid:
        G=255
    else:
        temp=round((value-th_mid)*255/delta)
        G=int(np.max(255-temp,0))

    if value>=th_mid:
        R=255
    else:
        temp = round((value - th_low) * 255 / delta)
        R=int(np.max(temp,0))

    return R, G, B