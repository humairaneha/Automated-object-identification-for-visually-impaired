# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 17:38:26 2024

@author: humai
"""


import argparse
import csv
import os
import platform
import sys
from pathlib import Path

import torch
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import torch
import torch.nn.functional as F
import subprocess
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box
from ultralytics import YOLO
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import classify_transforms
import torchvision.transforms as transforms
import numpy as np
from Face.Facemodule import recognize_face
from imageprocessing import apply_clahe_to_channels,is_low_light,improve_image,detect_blur,enhance_image
import cv2
print(cv2.getBuildInformation())
@smart_inference_mode()
def run(
        weights=ROOT / 'best300.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/mydata.yaml', 
        cur_data=ROOT / 'data/coco128.yaml',# dataset.yaml path
        imgsz=(640, 640),
        cur_imgsz=(224,224),# inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_csv=False,  # save results in CSV format
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)  #Convert the source path to a string, and determine if the source is a file, URL, webcam, or screenshot
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS) 
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file) 
    screenshot = source.lower().startswith('screen')
    if is_url and is_file: #If the source is a URL and a file, it checks and downloads the file
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    # Assuming there is a mislabeled class at index `mislabel_index`
    mislabel_index = 5  # Change this to the index of the mislabeled class
    corrected_class_name = 'person'  # Provide the corrected class name

# Modify the class name
    model.names[mislabel_index] = corrected_class_name

    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size


    
    cur_weights= r"C:\Users\humai\Documents\best.pt"
   #model for currency classifier
    cur_model = DetectMultiBackend(cur_weights, device=device, dnn=dnn, data=cur_data, fp16=half)
    cur_stride, cur_names, cur_pt = cur_model.stride, cur_model.names, cur_model.pt
    cur_imgsz = check_img_size(cur_imgsz, s=cur_stride)  # check image size
    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    cur_model.warmup(imgsz=(1 if cur_pt else bs, 3, *cur_imgsz))
      # warmup# warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    k=1
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
          
            test_im=im
            print(im.shape)
            
            #test_im = np.transpose(test_im, (1, 2, 0))
            '''
            if(is_low_light(test_im)):
                print("in low light")
                test_im=improve_image(test_im)
                print(test_im.shape)
                if(detect_blur(test_im)):
                    test_im = enhance_image(test_im)
                im=np.transpose(test_im, (2, 0, 1))
                '''
            im = torch.from_numpy(im).to(model.device) 
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / 'predictions.csv'

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            data = {'Image Name': image_name, 'Prediction': prediction, 'Confidence': confidence}
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)
        
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f'{names[c]}'
                    confidence = float(conf)
                    confidence_str = f'{confidence:.2f}'

                    
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        if names[c] == 'currency':
                            # Crop the region containing the currency
                            currency_crop = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                            print(currency_crop.shape)
                            
                            path=r"E:\yolo v scaled\New folder (2)\currency1_"+str(k)+".png"
                            path=r"E:\yolo v scaled\New folder (2)\currency2_"+str(k)+".png"
                            k+=1
                            cv2.imwrite(path, currency_crop)
                            currency_crop=cv2.resize(currency_crop,(224,224))
                            #cv2.imshow("window",currency_crop)
                           # cur_im=im0
                            #subprocess.run(["python", "predict.py", "--weights", "path_to_classification_model_weights.pt", "--source", str(roi_path), "--other_arguments"])
                            #cur_input = transforms.ToTensor()(cv2.resize(cur_im, (224, 224))).unsqueeze(0)
                            cur_input = classify_transforms(cur_imgsz[0])(currency_crop)
                            #print(cur_input.shape)
                            cur_input = torch.Tensor(cur_input).to(device)
                           
                            cur_input = cur_input.half() if cur_model.fp16 else cur_input.float()  # uint8 to fp16/32
                            cur_input_numpy = cur_input.permute(1, 2, 0).cpu().detach().numpy()
                            # Scale the pixel values to the range [0, 255]
                            cur_input_numpy = (cur_input_numpy * 255).astype(np.uint8)

                            # Display the images
                            #cv2.imwrite(path, cur_input_numpy)
                            #cv2.imshow('Preprocessed Image', cur_input_numpy)
                            #cv2.waitKey(0)
                            #cv2.destroyAllWindows()
                            if len(cur_input.shape) == 3:
                                cur_input = cur_input[None]  
                            #cur_input=cur_input.permute(0,3,1,2)
                            results = cur_model(cur_input)
                            cur_pred = F.softmax(results, dim=1)
                           
                            top2_probs, top2_classes = cur_pred.topk(2)
                            print("Top 2 Probabilities:", top2_probs.tolist())
                            print("Top 2 Classes:", top2_classes.tolist())
                            #print(top2_classes.tolist())
                            print(cur_names[int(top2_classes[0][0])])
                            label = None if hide_labels else (cur_names[int(top2_classes[0][0])] if hide_conf else f'{cur_names[int(top2_classes[0][0])]} {top2_probs[0][0]:.2f}')
                            if(top2_probs[0][0]<0.5):
                                label="currency value not detected"
                                annotator.box_label(xyxy,label, color=colors(c, True))
                            else:
                                annotator.box_label(xyxy,label, color=colors(c, True))
                            #here goes the curreny code
                        
                        elif names[c] == 'person' :
                            # Crop the region containing the currency
                            person_crop = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                            #print(person_crop.shape)
                            
                            path=r"E:\yolo v scaled\New folder (2)\person1_"+str(k)+".png"
                            #path=r"E:\yolo v scaled\New folder (2)\currency2_"+str(k)+".png"
                            k+=1
                            cv2.imwrite(path, person_crop)
                            
                            person_id=recognize_face(person_crop)
                            if(person_id!=-1):
                              annotator.box_label(xyxy,person_id, color=colors(c, True))
                            else:
                                annotator.box_label(xyxy,names[c]+"no face detected", color=colors(c, True))
                            
                        else:    
                          label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                          annotator.box_label(xyxy, label, color=colors(c, True))
                       #new added
                        
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
                    cv2.imwrite(r"D:\Defense Final\myyolov5\runs\image.jpg",im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'best (3).pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default= 0 , help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/mydata.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-csv', action='store_true', help='save results in CSV format')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
