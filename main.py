#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@author: Gong
"""

import math
import random
import warnings
from timeit import time

import cv2
import numpy as np
import pymysql
import tensorflow as tf
from PIL import Image
from mtcnn.mtcnn import MTCNN
from numpy import asarray

from age_gender.model import select_model, get_checkpoint
from age_gender.utils import ImageCoder, make_multi_crop_batch
from deep_sort import nn_matching, preprocessing
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from person import Person
from utils import utils
from yolo3.detector import Detector

warnings.filterwarnings("ignore")


def load_model(ckpt_path, label_list, image_size=160):
    graph = tf.Graph()
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(graph=graph, config=config)
    model = select_model('inception')
    with sess.as_default():
        with graph.as_default():
            images = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
            logits = model(len(label_list), images, 1, False)
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
            sess.run(init_op)
            model_ckpt_path, global_step = get_checkpoint(ckpt_path, None)
            saver = tf.train.Saver()
            saver.restore(sess, model_ckpt_path)
            softmax_output = tf.nn.softmax(logits)
            coder = ImageCoder()
            return sess, images, softmax_output, coder


def classify_one_multi_crop(sess, label_list, softmax_output, coder, images, image_file='./images/face.jpg'):
    image_batch = make_multi_crop_batch(image_file, coder)
    batch_results = sess.run(softmax_output, feed_dict={images: image_batch.eval()})
    output = batch_results[0]
    batch_sz = batch_results.shape[0]
    for i in range(1, batch_sz):
        output = output + batch_results[i]
    output /= batch_sz
    best = np.argmax(output)
    best_choice = (label_list[best], output[best])
    return best_choice


def calc_height(x, y, h):
    # pre-set params
    ori_x = 851.0
    ori_y = 248.0
    elevator_dist = 126.32
    elevator_ratio = 0.89
    base = 160
    scale = 30
    if x == ori_x and y == ori_y:
        return 0.0
    vec = np.array([x - ori_x, y - ori_y])
    dist = math.sqrt(vec.dot(vec))
    ratio = elevator_ratio * math.sqrt(elevator_dist / dist)
    predict_h = h * ratio
    if predict_h < base:
        return random.uniform(base, base + scale / 2)
    elif predict_h > base + scale:
        return random.uniform(base + scale / 2, base + scale)
    else:
        return predict_h


def extract_face(image, detector=MTCNN()):
    input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_image = None
    pixels = asarray(input_image)
    detection = detector.detect_faces(pixels)
    if len(detection) > 0:
        x, y, w, h = detection[0]['box']
        x, y = abs(x), abs(y)
        face = pixels[y: y + h, x: x + w]
        face_image = Image.fromarray(face)
    return face_image


def connect_db():
    return pymysql.connect(
        host='localhost',
        port=3306,
        user='root',
        password='root',
        database='video_detect',
        charset='latin1'
    )


def main(video_file, write_video=False):
    # yolo
    yolo_model = './models/yolo/yolo.h5'
    yolo_anchors = './models/yolo/yolo_anchors.txt'
    yolo_classes = './models/yolo/coco_classes.txt'
    yolo = Detector(
        model_path=yolo_model,
        anchors_path=yolo_anchors,
        classes_path=yolo_classes,
        score=0.5,
        iou=0.5
    )
    print('yolo loaded.')

    # deep_sort
    deep_sort_model = './models/deep_sort/mars-small128.pb'
    max_cosine_dist = 0.3
    nn_budget = None
    max_bbox_overlap = 1.0
    confidence = 1.0
    encoder, image_encoder = utils.create_box_encoder(
        model_filename=deep_sort_model,
        batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric(
        metric="cosine",
        matching_threshold=max_cosine_dist,
        budget=nn_budget
    )
    tracker = Tracker(metric)
    print('deep_sort loaded.')

    # face detection
    face_detector = MTCNN()
    print('mtcnn loaded.')

    # age prediction
    image_size = 160
    age_list = ['0~2', '4~6', '8~12', '15~20', '25~32', '38~43', '48~53', '60+']
    age_ckpt_path = './models/age_gender/age'
    age_sess, age_images, age_softmax_output, age_coder = load_model(
        ckpt_path=age_ckpt_path,
        label_list=age_list,
        image_size=image_size
    )

    # gender prediction
    gender_list = ['F', 'M']
    gender_ckpt_path = './models/age_gender/gender'
    gender_sess, gender_images, gender_softmax_output, gender_coder = load_model(
        ckpt_path=gender_ckpt_path,
        label_list=gender_list,
        image_size=image_size
    )
    print('rude-carnie loaded.')

    # read video
    video_capture = cv2.VideoCapture(video_file)
    w = int(video_capture.get(3))
    h = int(video_capture.get(4))
    video_fps = video_capture.get(cv2.CAP_PROP_FPS)
    # print(w, h, video_fps)

    # write video
    if write_video:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video_writer = cv2.VideoWriter('output.avi', fourcc, 15, (w, h))
    detection_file = open('detection.txt', 'w')

    # frame info
    fps = 0.0
    num_frame = -1
    total_pool = set()
    current_pool = set()
    person_map = {}

    # output setting
    border_thickness = 2
    background_gbr = (18, 153, 255)
    padding = 2
    font_scale = 0.4
    font_face = cv2.FONT_HERSHEY_COMPLEX
    font_thickness = 1
    font_bgr = (255, 255, 255)
    _, text_height = cv2.getTextSize("test", font_face, font_scale, font_thickness)[0]
    # print(text_height)

    # main process
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        num_frame += 1
        t = time.time()
        current_time = float(num_frame) / float(video_fps)

        # RGB image
        image = Image.fromarray(frame[..., ::-1])

        # yolo detection
        bbox_list = yolo.detect_image(image)

        # deep sort
        with image_encoder.session.as_default():
            with image_encoder.session.graph.as_default():
                features = encoder(frame, bbox_list)
                detection_list = [Detection(bbox, confidence=confidence, feature=feature)
                                  for bbox, feature in zip(bbox_list, features)]
                # suppression
                bbox_list = np.array([detection.tlwh for detection in detection_list])
                confidence_list = np.array([detection.confidence for detection in detection_list])
                indices = preprocessing.non_max_suppression(bbox_list, max_bbox_overlap, confidence_list)
                detection_list = [detection_list[i] for i in indices]
                tracker.predict()
                tracker.update(detection_list)

                # deal with track
                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue

                    # read id
                    person_id = int(track.track_id)
                    current_pool.add(person_id)
                    if person_id not in total_pool:
                        person = Person(person_id, current_time)
                        person_map[person_id] = person
                        total_pool.add(person_id)
                    person = person_map[person_id]

                    # cut image
                    bbox = track.to_tlbr()
                    x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    if x_min < 0 or y_min < 0 or x_max > w or y_max > h:
                        continue

                    # height prediction
                    height = calc_height(
                        x=x_min + (x_max - x_min) / 2.0,
                        y=y_max,
                        h=y_max - y_min)
                    person.height_list.append(height)
                    person.latest_height = height

                    # update person info
                    if num_frame % 5 == 0:
                        # face detection
                        face = extract_face(
                            image=frame[y_min:y_max, x_min:x_max],
                            detector=face_detector
                        )

                        if face is not None:
                            # age detection
                            temp_file = './images/face.jpg'
                            face.save(temp_file)
                            with age_sess.as_default():
                                with age_sess.graph.as_default():
                                    with tf.device('/cpu:0'):
                                        age, score = classify_one_multi_crop(
                                            sess=age_sess,
                                            label_list=age_list,
                                            softmax_output=age_softmax_output,
                                            coder=age_coder,
                                            images=age_images,
                                            image_file=temp_file)
                                        person.age_dict[age] = person.age_dict.get(age, 0) + score
                                        person.latest_age = age
                                        person.latest_age_score = score

                            # gender detection
                            with gender_sess.as_default():
                                with gender_sess.graph.as_default():
                                    with tf.device('/cpu:0'):
                                        gender, score = classify_one_multi_crop(
                                            sess=gender_sess,
                                            label_list=gender_list,
                                            softmax_output=gender_softmax_output,
                                            coder=gender_coder,
                                            images=gender_images,
                                            image_file=temp_file)
                                        person.gender_dict[gender] = person.gender_dict.get(gender, 0) + score
                                        person.latest_gender = gender
                                        person.latest_gender_score = score

                    # add info to frame
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), background_gbr, border_thickness)
                    label_list = person.get_label()

                    background_rect = (
                        (x_min - padding, y_min - text_height * 2 - padding * 3),
                        (x_max + padding, y_min + text_height * 2 + padding * 2))
                    cv2.rectangle(frame, background_rect[0], background_rect[1], background_gbr, cv2.FILLED)

                    cv2.putText(frame, "ID: " + label_list[0], (x_min, y_min - text_height - padding * 2),
                                font_face, font_scale, font_bgr, font_thickness)
                    cv2.putText(frame, "G: " + label_list[1], (x_min, y_min - padding),
                                font_face, font_scale, font_bgr, font_thickness)
                    cv2.putText(frame, "A: " + label_list[2], (x_min, y_min + text_height),
                                font_face, font_scale, font_bgr, font_thickness)
                    cv2.putText(frame, "H: " + label_list[3], (x_min, y_min + text_height * 2 + padding),
                                font_face, font_scale, font_bgr, font_thickness)

            for idx in current_pool:
                person = person_map[idx]
                person.end = current_time
                detection_file.write(person.__str__() + "\n")
            current_pool.clear()

            fps = (fps + (1. / (time.time() - t))) / 2
            # print("%d, fps = %.2f" % (num_frame, fps))
            cv2.putText(frame, "%.2f" % fps, (22, 33), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1)

            # show frame
            # cv2.imshow("", frame)

            # save frame
            if write_video:
                video_writer.write(frame)

    # add into database
    con = connect_db()
    cur = con.cursor()
    for person in person_map.values():
        duration = person.end - person.start
        if duration > 1.5:
            person_info = person.get_info()
            try:
                sql_str = ("INSERT INTO t_persons (age, gender, height, appear, disappear)"
                           + " VALUES ('%s', '%s', '%f', '%f', '%f')" %
                           (
                               person_info[3],
                               person_info[4],
                               float(person_info[5]),
                               float(person_info[1]),
                               float(person_info[2])
                           ))
                cur.execute(sql_str)
                con.commit()
            except Exception as e:
                print(person.__str__())
                con.rollback()
                raise e
    cur.close()
    con.close()

    # end process
    video_capture.release()
    if write_video:
        video_writer.release()

    # detection_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video = 'cut_2.mp4'
    main(video_file=video,
         write_video=True)
