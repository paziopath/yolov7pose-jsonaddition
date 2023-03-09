import cv2
import time
import torch
import argparse
import numpy as np
import json #_________________________________paziopath_______________________________________________________#
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt, strip_optimizer, xyxy2xywh
from utils.plots import output_to_keypoint, plot_skeleton_kpts, colors, plot_one_box_kpt
#_________________________________paziopath_______________________________________________________#
from pathlib import Path


NOSE = [0]
LEFT_EYE = [1]
RIGHT_EYE = [2]
LEFT_EAR = [3]
RIGHT_EAR = [4]

RIGHT_ARM = [5, 7, 9]
LEFT_ARM = [6, 8, 10]
RIGHT_LEG = [11, 13, 15]
LEFT_LEG = [12, 14, 16]

key_map = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}
#_________________________________paziopath_______________________________________________________#

@torch.no_grad()
def run(poseweights="yolov7-w6-pose.pt", source="football1.mp4", device='gpu', view_img=False,
        save_conf=True, line_thickness=3, hide_labels=False, hide_conf=True,
        save_dir=Path(r"C:\Users\elham\Documents\GitHub\yolov7-pose-estimation")):
    frame_count = 0  # count no of frames
    total_fps = 0  # count total fps
    time_list = []  # list to store time
    fps_list = []  # list to store fps

    device = select_device(opt.device)  # select device
    half = device.type != 'cpu'

    model = attempt_load(poseweights, map_location=device)  # Load model
    _ = model.eval()
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names

    if source.isnumeric():
        cap = cv2.VideoCapture(int(source))  # pass video to videocapture object
    else:
        cap = cv2.VideoCapture(source)  # pass video to videocapture object

    if (cap.isOpened() == False):  # check if videocapture not opened
        print('Error while trying to read video. Please check path again')
        raise SystemExit()

    else:
        frame_width = int(cap.get(3))  # get video frame width
        frame_height = int(cap.get(4))  # get video frame height

        vid_write_image = letterbox(cap.read()[1], (frame_width), stride=64, auto=True)[0]  # init videowriter
        resize_height, resize_width = vid_write_image.shape[:2]
        out_video_name = f"{source.split('/')[-1].split('.')[0]}"
        out = cv2.VideoWriter(f"{source}_keypoint.mp4",
                              cv2.VideoWriter_fourcc(*'mp4v'), 30,
                              (resize_width, resize_height))

#_________________________________paziopath_______________________________________________________#
        jdict = dict()
        
        count_i = 0
#_________________________________paziopath_______________________________________________________#
        while (cap.isOpened):  # loop until cap opened or video not complete

            print("Frame {} Processing".format(frame_count + 1))
            
            

            ret, frame = cap.read()  # get frame and success from video capture

            if ret:  # if success is true, means frame exist
                orig_image = frame  # store frame
                image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)  # convert frame to RGB
                image = letterbox(image, (frame_width), stride=64, auto=True)[0]
                image_ = image.copy()
                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))

                image = image.to(device)  # convert image data to device
                image = image.float()  # convert image to float precision (cpu)
                start_time = time.time()  # start time for fps calculation

                with torch.no_grad():  # get predictions
                    output_data, _ = model(image)

                output_data = non_max_suppression_kpt(output_data,  # Apply non max suppression
                                                      0.25,  # Conf. Threshold.
                                                      0.65,  # IoU Threshold.
                                                      nc=model.yaml['nc'],  # Number of classes.
                                                      nkpt=model.yaml['nkpt'],  # Number of keypoints.
                                                      kpt_label=True)

                output = output_to_keypoint(output_data)

                im0 = image[0].permute(1, 2,
                                       0) * 255  # Change format [b, c, h, w] to [h, w, c] for displaying the image.
                im0 = im0.cpu().numpy().astype(np.uint8)

                im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)  # reshape image format to (BGR)
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                for i, pose in enumerate(output_data):  # detections per image
#_________________________________paziopath_______________________________________________________#
                    predn = pose.clone()
                    frame_dic = []
#_________________________________paziopath_______________________________________________________#
                    if len(output_data):  # check if no pose
                        for c in pose[:, 5].unique():  # Print results
                            n = (pose[:, 5] == c).sum()  # detections per class
                            print("No of Objects in Current Frame : {}".format(n))

                        for det_index, (*xyxy, conf, cls) in enumerate(
                                reversed(pose[:, :6])):  # loop over poses for drawing on frame
                            c = int(cls)  # integer class
                            kpts = pose[det_index, 6:]
                            label = None if opt.hide_labels else (
                                names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                            plot_one_box_kpt(xyxy, im0, label=label, color=colors(c, True),
                                             line_thickness=opt.line_thickness, kpt_label=True, kpts=kpts, steps=3,
                                             orig_shape=im0.shape[:2])

#_________________________________paziopath_______________________________________________________#
                        # for json
                        image_id = i
                        box = xyxy2xywh(predn[:, :4])  # xywh
                        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                        for p, b in zip(pose.tolist(), box.tolist()):
                            det_dict = ({  # 'frame:': frame_count,
                                # 'image_id': image_id,
                                # 'category_id': coco91class[int(p[5])] if False else int(p[5]),
                                'bbox': [round(x, 3) for x in b],
                                'score': round(p[4], 5)})
                            key_point = p[6:]
                            kdict = dict()
                            klist = list()

                            for i in range(0, len(key_point),
                                           3):  
                                           
                                klist.append(key_point[i:i + 3])

                            for j in range(len(klist)):
                                kdict.update({str(key_map[j]): klist[j]})
                            # det_dict.update({'keypoints': key_point})

                            det_dict.update({'keypoints': kdict})
                            frame_dic.append(det_dict)
                            
#_________________________________paziopath_______________________________________________________#
                            
                if frame_dic == []:
                    det_dict = ({
                        'bbox': 'null',
                        'score': 'null'})
                    det_dict.update({'keypoints': {'null': 'null'}})
                    frame_dic.append(det_dict)
                    jdict.update({"FRAME " + str(count_i): frame_dic})
                else:
                    jdict.update({"FRAME " + str(count_i): frame_dic})

                end_time = time.time()  # Calculatio for FPS
                fps = 1 / (end_time - start_time)
                total_fps += fps
                frame_count += 1

                fps_list.append(total_fps)  # append FPS in list
                time_list.append(end_time - start_time)  # append time in list

                # Stream results
                if view_img:
                    cv2.imshow("YOLOv7 Pose Estimation Demo", im0)
                    cv2.waitKey(1)  # 1 millisecond

                out.write(im0)  # writing the video frame

#_________________________________paziopath_______________________________________________________#
                count_i += 1
                

            else:
                break

        cap.release()
        # cv2.destroyAllWindows()
        avg_fps = total_fps / frame_count
        print(f"Average FPS: {avg_fps:.3f}")
        if len(jdict):
            w = Path('')  # weights
            
            pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
            print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
            with open(pred_json, 'w') as f:
                json.dump(jdict, f)
                
#_________________________________paziopath_______________________________________________________#

        # plot the comparision graph
        plot_fps_time_comparision(time_list=time_list, fps_list=fps_list)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='football1.mp4', help='video/0 for webcam')  # video source
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')  # device arugments
    parser.add_argument('--view-img', action='store_true', help='display results')  # display results
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')  # save confidence in txt writing
    parser.add_argument('--line-thickness', default=3, type=int,
                        help='bounding box thickness (pixels)')  # box linethickness
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')  # box hidelabel
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')  # boxhideconf
    opt = parser.parse_args()
    return opt


# function for plot fps and time comparision graph
def plot_fps_time_comparision(time_list, fps_list):
    plt.figure()
    plt.xlabel('Time (s)')
    plt.ylabel('FPS')
    plt.title('FPS and Time Comparision Graph')
    plt.plot(time_list, fps_list, 'b', label="FPS & Time")
    plt.savefig("FPS_and_Time_Comparision_pose_estimate.png")

#_________________________________paziopath_______________________________________________________#
def vii_tool_format(df, filename):
    import os.path
    import datetime

    recordingID = ""
    filepath = os.path.abspath(filename)
    created_timestamp = os.path.getctime(filepath)
    created_date = datetime.datetime.fromtimestamp(created_timestamp)
    recordingID += created_date.strftime("%Y-%m-%d %H:%M:%S")

    output_json = {"recordingID": recordingID,
                   "applicationName": "YOLOV7"}
    frames = []

    time = datetime.datetime.combine(datetime.date.today(), datetime.time.min)
    df = df.to_dict(orient='index')

    for f, i in df.items():
        time += datetime.timedelta(seconds=1 / 30)
        time_string = time.strftime('%H:%M:%S.%f')
        fr_dict = {"frameStamp": time_string,
                   "frameAttributes": {}}
        for j in range(len(i)):
            if i[j] and i[j]['keypoints'] == {'null': 'null'}:
                frameAttributes = {
                    str(j) + "_Nose_X": 'null',
                    str(j) + "_Nose_Y": 'null',
                    str(j) + "_EyeLeft_X": 'null',
                    str(j) + "_EyeLeft_Y": 'null',
                    str(j) + "_EyeRight_X": 'null',
                    str(j) + "_EyeRight_Y": 'null',
                    str(j) + "_EarLeft_X": 'null',
                    str(j) + "_EarLeft_Y": 'null',
                    str(j) + "_EarRight_X": 'null',
                    str(j) + "_EarRight_Y": 'null',
                    str(j) + "_ShoulderLeft_X": 'null',
                    str(j) + "_ShoulderLeft_Y": 'null',
                    str(j) + "_ShoulderRight_X": 'null',
                    str(j) + "_ShoulderRight_Y": 'null',
                    str(j) + "_ElbowLeft_X": 'null',
                    str(j) + "_ElbowLeft_Y": 'null',
                    str(j) + "_ElbowRight_X": 'null',
                    str(j) + "_ElbowRight_Y": 'null',
                    str(j) + "_WristLeft_X": 'null',
                    str(j) + "_WristLeft_Y": 'null',
                    str(j) + "_WristRight_X": 'null',
                    str(j) + "_WristRight_Y": 'null',
                    str(j) + "_HipLeft_X": 'null',
                    str(j) + "_HipLeft_Y": 'null',
                    str(j) + "_HipRight_X": 'null',
                    str(j) + "_HipRight_Y": 'null',
                    str(j) + "_KneeLeft_X": 'null',
                    str(j) + "_KneeLeft_Y": 'null',
                    str(j) + "_KneeRight_X": 'null',
                    str(j) + "_KneeRight_Y": 'null',
                    str(j) + "_AnkleLeft_X": 'null',
                    str(j) + "_AnkleLeft_Y": 'null',
                    str(j) + "_AnkleRight_X": 'null',
                    str(j) + "_AnkleRight_Y": 'null'
                }
                fr_dict["frameAttributes"].update(frameAttributes)

            elif i[j]:
                frameAttributes = {
                    str(j) + "_Nose_X": str(i[j]['keypoints']['nose'][0]),
                    str(j) + "_Nose_Y": str(i[j]['keypoints']['nose'][1]),
                    str(j) + "_EyeLeft_X": str(i[j]['keypoints']['left_eye'][0]),
                    str(j) + "_EyeLeft_Y": str(i[j]['keypoints']['left_eye'][1]),
                    str(j) + "_EyeRight_X": str(i[j]['keypoints']['right_eye'][0]),
                    str(j) + "_EyeRight_Y": str(i[j]['keypoints']['right_eye'][1]),
                    str(j) + "_EarLeft_X": str(i[j]['keypoints']['left_ear'][0]),
                    str(j) + "_EarLeft_Y": str(i[j]['keypoints']['left_ear'][1]),
                    str(j) + "_EarRight_X": str(i[j]['keypoints']['right_ear'][0]),
                    str(j) + "_EarRight_Y": str(i[j]['keypoints']['right_ear'][1]),
                    str(j) + "_ShoulderLeft_X": str(i[j]['keypoints']['left_shoulder'][0]),
                    str(j) + "_ShoulderLeft_Y": str(i[j]['keypoints']['left_shoulder'][1]),
                    str(j) + "_ShoulderRight_X": str(i[j]['keypoints']['right_shoulder'][0]),
                    str(j) + "_ShoulderRight_Y": str(i[j]['keypoints']['right_shoulder'][1]),
                    str(j) + "_ElbowLeft_X": str(i[j]['keypoints']['left_elbow'][0]),
                    str(j) + "_ElbowLeft_Y": str(i[j]['keypoints']['left_elbow'][1]),
                    str(j) + "_ElbowRight_X": str(i[j]['keypoints']['right_elbow'][0]),
                    str(j) + "_ElbowRight_Y": str(i[j]['keypoints']['right_elbow'][1]),
                    str(j) + "_WristLeft_X": str(i[j]['keypoints']['left_wrist'][0]),
                    str(j) + "_WristLeft_Y": str(i[j]['keypoints']['left_wrist'][1]),
                    str(j) + "_WristRight_X": str(i[j]['keypoints']['right_wrist'][0]),
                    str(j) + "_WristRight_Y": str(i[j]['keypoints']['right_wrist'][1]),
                    str(j) + "_HipLeft_X": str(i[j]['keypoints']['left_hip'][0]),
                    str(j) + "_HipLeft_Y": str(i[j]['keypoints']['left_hip'][1]),
                    str(j) + "_HipRight_X": str(i[j]['keypoints']['right_hip'][0]),
                    str(j) + "_HipRight_Y": str(i[j]['keypoints']['right_hip'][1]),
                    str(j) + "_KneeLeft_X": str(i[j]['keypoints']['left_knee'][0]),
                    str(j) + "_KneeLeft_Y": str(i[j]['keypoints']['left_knee'][1]),
                    str(j) + "_KneeRight_X": str(i[j]['keypoints']['right_knee'][0]),
                    str(j) + "_KneeRight_Y": str(i[j]['keypoints']['right_knee'][1]),
                    str(j) + "_AnkleLeft_X": str(i[j]['keypoints']['left_ankle'][0]),
                    str(j) + "_AnkleLeft_Y": str(i[j]['keypoints']['left_ankle'][1]),
                    str(j) + "_AnkleRight_X": str(i[j]['keypoints']['right_ankle'][0]),
                    str(j) + "_AnkleRight_Y": str(i[j]['keypoints']['right_ankle'][1])
                }
            fr_dict["frameAttributes"].update(frameAttributes)
        frames.append(fr_dict)

    output_json.update({'Frames': frames})

    save_dir = Path(r"C:\Users\elham\Documents\GitHub\yolov7-pose-estimation")
    pred_json = str(save_dir / f"Vii_predictions.json")  # predictions json
    print('\nsaving %s...' % pred_json)
    with open(pred_json, 'w') as f:
        json.dump(output_json, f, indent=4)



# main function
def main(opt):
    run(**vars(opt))
    import pandas as pd
    j = pd.read_json(r"C:\Users\elham\Documents\GitHub\yolov7-pose-estimation\._predictions.json",
                     orient="index")
    vii_tool_format(j, 'football1.mp4')
    x = 0
#_________________________________paziopath_______________________________________________________#

if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device, opt.poseweights)
    main(opt)
