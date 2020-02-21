import os
import argparse
from yolo import YOLO, detect_video
from timeit import default_timer as timer

def get_parent_dir(n=1):
    current_path = os.path.dirname(os.path.abspath(__file__))
    for k in range(n):
        current_path = os.path.dirname(current_path)
    return current_path


def GetFileList(dirName,endings):
    listOfFile = os.listdir(dirName)
    allFiles = list()

    for i,ending in enumerate(endings):
        if ending[0]!='.':
            endings[i] = '.'+ending
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + GetFileList(fullPath,endings)
        else:
            for ending in endings:
                if entry.endswith(ending):
                    allFiles.append(fullPath)
    return allFiles


data_folder = os.path.join(get_parent_dir(n=1), 'Data')
image_folder = os.path.join(data_folder, 'Source_Video')
image_test_folder = os.path.join(image_folder, 'Video_Test')
detection_results_folder = os.path.join(get_parent_dir(n=1), 'Results')
model_folder = os.path.join(data_folder, 'Model_Weights')
model_weights = os.path.join(model_folder, 'trained_weights_final.h5')
model_classes = 'data_classes.txt'
anchors_path =  'yolo_anchors.txt'

FLAGS = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument(
        "--input_path", type=str, dest='input_path', default=image_test_folder,
        help="Absolute path to video directory. Default is " + image_test_folder
    )
    parser.add_argument(
        "--output_path", type=str, dest='output', default=detection_results_folder,
        help="Output path for detection video results. Default is " + detection_results_folder
    )
    parser.add_argument(
        '--weights_path', type=str, dest='weights_path', default=model_weights,
        help='Path to pre-trained weight files. Default is ' + model_weights
    )
    parser.add_argument(
        '--confidence', type=float, dest='score', default=0.1,
        help='Threshold for YOLO object confidence score to show predictions. Default is 0.25.'
    )
    parser.add_argument(
        '--threshold_mode', type=str, dest='th_mode', default='counting',
        help='set the threshold mode :'
             'counting=threshold define by number of the vehicles,'
             'density=threshold define by the density of the vehicles, '
             'velocity=threshold define by the velocity of the vehicles'
    )
    parser.add_argument(
        '--threshold_low', type=float, dest='th_low', default=3,
        help='set the upper threshold density for low congestion'
    )
    parser.add_argument(
        '--threshold_high', type=float, dest='th_high', default=4,
        help='set the lower threshold density for high congestion'
    )
    parser.add_argument(
        '--define_regions', type=int, dest='define_regions', default=0,
        help='Option for the user to choose the regions of the video for the detection'
    )

    FLAGS = parser.parse_args()
    th_low = FLAGS.th_low
    th_high = FLAGS.th_high
    define_regions = FLAGS.define_regions
    th_mode = FLAGS.th_mode
    vid_endings = ('.mp4', '.mpeg', '.mpg', '.avi', '.mov')
    input_path = GetFileList(FLAGS.input_path,vid_endings)
    output_path = FLAGS.output
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    yolo_obj = YOLO(**{"model_path": FLAGS.weights_path,
                       "anchors_path": anchors_path,
                       "classes_path": model_classes,
                       "score": FLAGS.score,
                       "model_image_size": (416, 416),
                       }
                    )

    if input_path[0]:
        start = timer()
        output_path = os.path.join(FLAGS.output, os.path.basename(input_path[0]))
        detect_video(yolo_obj, input_path[0], th_mode, th_low, th_high, define_regions, output_path=output_path,
                     input_path=input_path[0])
        end = timer()
        print('Processed video in {:.1f}sec'.format(end - start))
    yolo_obj.close_session()




