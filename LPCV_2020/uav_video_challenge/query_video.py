#!/usr/bin/env python3


import argparse
import glob
import json
import os

import cv2
from ocr_lib import OCRLib


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run OCR model on a image or directory of images"
    )
    parser.add_argument(
        "--input_video", help="Path to the input video", type=str, required=True
    )
    parser.add_argument(
        "--query_file", help="Path of question text file", type=str, required=True
    )
    parser.add_argument(
        "--results_path",
        help="Path to store the results from video",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--sampling_rate", help="Sample every x frames", default=1, type=int
    )
    parser.add_argument(
        "--rcg_scr_th", help="Threshold for kept text", default=0.9, type=float
    )
    parser.add_argument(
        "--config_file", type=str, help="OCRLib config json file", required=True
    )

    args = parser.parse_args()

    if args.results_path is None:
        args.results_path = (
            os.path.splitext(os.path.basename(args.input_video))[0] + "_results"
        )
        print('Using "{}" as results cache path'.format(args.results_path))
        if not os.path.exists(args.results_path):
            os.makedirs(args.results_path)

    return args


def load_result_index(results_path, rcg_scr_th):
    json_files = glob.glob(os.path.join(results_path, "*.json"))
    txt2frm, frm2txt = {}, {}
    for json_file in json_files:
        frame_no = int(os.path.splitext(os.path.basename(json_file))[0])
        with open(json_file, "r") as h_input:
            results = json.load(h_input)

        frm2txt[frame_no] = []
        for result in results:
            if result["rcg_scr"] > rcg_scr_th:
                text = result["rcg_str"].upper()

                frm2txt[frame_no].append(text)
                if text in txt2frm:
                    txt2frm[text].append(frame_no)
                else:
                    txt2frm[text] = [frame_no]

    return txt2frm, frm2txt


def process_video(input_video, cfg_fn, results_path, sampling_rate):
    reader = OCRLib(cfg_fn)
    vidcap = cv2.VideoCapture(input_video)
    assert vidcap.isOpened()

    success = True
    cnt = 0
    while success:
        success, img = vidcap.read()
        cnt += 1

        if not success:
            break

        if (cnt - 1) % sampling_rate != 0:
            continue

        frm_output = os.path.join(results_path, "{}.json".format(cnt))
        if os.path.exists(frm_output):
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = reader.process_rgb_image(img)

        with open(frm_output, "w") as h_out:
            json.dump(results, h_out)


def query_video(
    query_file, input_video, config_file, results_path, sampling_rate, rcg_scr_th=0.9
):

    # Process video
    process_video(input_video, config_file, results_path, sampling_rate)

    # Create database lookup
    txt2frm, frm2txt = load_result_index(results_path, rcg_scr_th)

    # Query results
    with open(query_file, "r") as f_query:
        queries = f_query.read().replace(" ", "").replace("\n", "").split(";")

    for query in queries:
        if query not in txt2frm:
            print("{}: -;".format(query))
        else:
            frms = txt2frm[query]
            outputs = ""
            if len(frms) == 1:
                outputs = " ".join([x for x in frm2txt[frms[0]] if x != query])
            else:
                for frm in frms:
                    outputs += " frame_{} ".format(frm)
                    outputs += " ".join([x for x in frm2txt[frm] if x != query])
            print("{}: {};".format(query, outputs))


if __name__ == "__main__":
    args = parse_args()

    query_video(**vars(args))
