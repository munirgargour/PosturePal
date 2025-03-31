# PosturePal
A Python application designed to keep your posture perfect, implementing Meta's Detectron2 computer vision model
This project uses Detectron2's keypoint rcnn R 50 FPN 1x model to track the facial and skeletal positions of the user,
take a baseline photo and detect slouching by various methods.

Here is a link to the Detectron2 Github page: https://github.com/facebookresearch/detectron2

The file videotest-cpu.py is the executable file, alternatively I have packaged it into an .exe file, which you can find here: 
https://drive.google.com/file/d/1ktLEG_m7R4v1AUNbbE3kwysbVzWQhe-B/view?usp=drivesdk

Videotest-cpu.py will only work in the detectron base folder.
Clone Detectron2's repository into a folder and put the python file in the base "detectron2" folder for it to run properly.
