import cv2
import time
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import winsound


cfg = get_cfg()
# Load the model configuration from the Model Zoo (Detectron2's keypoint model, specifically the lowest resource-requiring version.)
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml"))
cfg.MODEL.DEVICE = "cpu"  # Force CPU mode to allow users with no Nvidia GPU to run the program, also avoids making them install cuda
# Set the detection threshold (only detections above this confidence will be used, basically if it's that percent sure about a given bodily feature)
conf_thresh = 0.6
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_thresh
# Load pre-trained weights for the model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml")

predictor = DefaultPredictor(cfg)

# ----------------------------------------
#  Open Video Feed and Wait for Baseline
# ----------------------------------------

cap = cv2.VideoCapture(0)  #sets video device to webcam 0, change if you need it to be different

if not cap.isOpened():
    print("Error: Could not open video stream.",
          "Make sure webcam 0 is connected and PostureCheck is given permissions.")
    time.sleep(5.0)
    exit()

def posturewarning():
    winsound.Beep(1000,500)
    warning_message = "Posture Warning! Press 'C' to continue or 'R' to retake baseline photo."
    print(warning_message)
    warning_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(warning_frame, warning_message, (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.imshow("Posture Warning", warning_frame)
    
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('c') or key == ord('C'):
            cv2.destroyWindow("Posture Warning")
            return "continue"
        elif key == ord('r') or key == ord('R'):
            cv2.destroyWindow("Posture Warning")
            return "retake"



def capturephoto(): # used for capturing baseline photos
    baseline_frame = None
    while baseline_frame is None:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab a frame.")
        else:
            frame = cv2.resize(frame, (1280, 720))
            instruction = ("Press 'P' when sitting straight with good posture.")
            cv2.putText(frame, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("Baseline Capture - Press \"P\" to capture or \"Q\" to quit", frame)

    # Check if user pressed 'P' or 'p'
            key = cv2.waitKey(1) & 0xFF
            if key == ord('p') or key == ord('P'):
                baseline_frame = frame.copy()  # Capture current frame as baseline
                cv2.destroyAllWindows()
            elif key == ord('q') or key == ord('Q'):
                print("Exiting...")
                cap.release()
                cv2.destroyAllWindows()
                exit()
    return baseline_frame
            
# Loop to display the video feed with an instruction overlay
def analyzebaseline():
    baselineframe = capturephoto()
    if baselineframe is None:
        print("Error: No baseline frame captured.")
        return None
    outputs = predictor(baselineframe)
    instances = outputs["instances"].to("cpu")
    if len(instances) > 0:
        keypoints = instances[0].pred_keypoints.numpy().squeeze(0)  # Expected shape: (17, 3)
        nose = keypoints[0] if keypoints[0,2] >= conf_thresh else None 
        left_eye = keypoints[1] if keypoints[1,2] >= conf_thresh - 0.1 else None
        right_eye = keypoints[2] if keypoints[2,2] >= conf_thresh - 0.1 else None
        left_ear = keypoints[3] if keypoints[3,2] >= conf_thresh - 0.1 else None
        right_ear = keypoints[4] if keypoints[4,2] >= conf_thresh - 0.1 else None
        keypoint_list = [nose,left_eye,right_eye,left_ear,right_ear]
        print("Baseline keypoints:", keypoint_list)
        if nose is not None:
            return keypoint_list
    else:
         print("No person detected in the baseline image.")

def posturecheck(keypointlist):
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. There may be some issue with your webcam.")
            return None
        else:
            frame = cv2.resize(frame, (1280, 720))
    
        outputs = predictor(frame)
        visualizer = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.5)
        out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
        result_frame = out.get_image()
        instances = outputs["instances"].to("cpu")
        if len(instances) > 0:
            keypoints = instances[0].pred_keypoints.numpy().squeeze(0) #squeezes each keypoint to 2 variables so multiple people don't need to be specified
            current_nose = keypoints[0] if keypoints[0,2] >= conf_thresh else None
            current_left_eye = keypoints[1] if keypoints[1,2] >= conf_thresh - 0.1 else None
            current_right_eye = keypoints[2] if keypoints[2,2] >= conf_thresh - 0.1 else None
            current_left_ear = keypoints[3] if keypoints[3,2] >= conf_thresh - 0.1 else None
            current_right_ear = keypoints[4] if keypoints[4,2] >= conf_thresh - 0.1 else None
        #current_keypoints = [current_nose, current_left_eye, current_right_eye, current_left_ear, current_right_ear]
        #print("Current keypoints:", current_keypoints, "Baseline keypoints:", baseline_keypoints)  # Debug print statement
            posturethreshold = 50

            if current_nose is not None:
                print(f"Current nose Y",current_nose[1], "Baseline nose Y:" , keypointlist[0][1])

            if (current_nose is not None and (current_nose[1] >= keypointlist[0][1] + posturethreshold)) or ():
                print("Posture warning triggered!")
                if posturewarning() == "retake":
                    keypointlist = analyzebaseline()
                    while keypointlist is None:
                        keypointlist = analyzebaseline()

    # Display the processed frame with keypoints. Will likely remove this in the future for ease of use
        cv2.imshow("Posture Check Window - Press Q to quit application.", result_frame)
        return "continue"

baseline_keypoints = analyzebaseline()
while baseline_keypoints == None:
    baseline_keypoints = analyzebaseline()

while True:
    action = posturecheck(baseline_keypoints)
    if action == "retake":
        baseline_keypoints = analyzebaseline()
        while baseline_keypoints is None:
            baseline_keypoints = analyzebaseline()
    elif action == "continue":
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.waitKey(30000)
