# Enhancing-toll-gate-management-system-using-computer-vision-and-image-processing

## Introduction:


In the ever-evolving landscape of transportation and technology, Toll Gate Management Systems play a crucial role in streamlining the flow of traffic and optimizing revenue collection. Traditional toll collection methods often result in congestion, delays, and inefficiencies. To address these challenges, our project proposes an innovative solution that leverages cutting-edge technologies in computer vision and image processing. 

Our system is designed to seamlessly detect and track vehicles as they pass through toll gates, eliminating the need for vehicles to come to a complete stop for fee collection. The backbone of our solution is the YOLOv8 (You Only Look Once) model, a state-of-the-art real-time object detection system. This model ensures accurate and efficient identification of vehicles, even in challenging conditions. Once a vehicle is detected, our system utilizes EasyOCR to extract the license plate number from the captured images. EasyOCR, known for its robust text recognition capabilities, ensures reliable extraction of alphanumeric characters from license plates, regardless of variations in font, size, or lighting conditions. The extracted license plate number serves as a unique identifier for each vehicle, allowing for seamless and automated fee collection. With this information, toll gate authorities can associate the vehicle with its corresponding toll fee, facilitating swift and contactless transactions. By enabling fee collection while the vehicle is in motion, our system significantly reduces traffic congestion and travel time for commuters. 

In summary, our Toll Gate Management System represents a paradigm shift in toll collection, embracing advanced technologies to create a more efficient, secure, and user-friendly experience for both commuters and toll gate authorities. Through the integration of YOLOv8 and EasyOCR, we aim to revolutionize the way toll gates operate, ushering in a new era of intelligent and contactless transportation management.


## Features:
This project focuses on the development and implementation of an innovative solution for enhancing the toll gate management system  using state-of-the-art technologies in Optical Character Recognition (OCR) and YOLOv8.

### EasyOCR Technology Integration:
The scope extends to incorporating the EasyOCR technique, a sophisticated optical character recognition tool, for extracting alphanumeric characters from the detected number plates.


### Utilization of the YOLOv8:
The utilization of the YOLOv8 model for real-time vehicle detection is a pivotal aspect of the system's scope. This topic involves understanding and implementing state-of-the-art object detection algorithms to 
accurately identify vehicles in each frame.


   
### Tracking Using SORT:
Tracking the movement of vehicles is crucial for continuous monitoring. A sort algorithm is employed to maintain the integrity of vehicle tracking across successive frames, contributing to the system's ability to manage dynamic traffic scenarios.

## Requirements
### Hardware Requirements
The hardware requirements for the implementation of the proposed text recognition system from handwritten images are outlined below:
#### High-Performance Workstation: 
A workstation with a multicore processor (e.g., Intel Core i7 or AMD Ryzen 7) for parallel processing.
#### Graphics Processing Unit (GPU):
A dedicated GPU (e.g., NVIDIA GeForce RTX series) for accelerated computations, especially for deep learning tasks.
#### Memory (RAM):
Minimum 6GB of RAM to handle the computational demands of OCR and image processing tasks.
#### Storage:
Adequate storage space (preferably SSD) to accommodate large datasets and model files.
#### High-Resolution Display:
A high-resolution 5 for detailed image analysis and visualization.

### Softare Requirements

The software requirements for the successful deployment of the text recognition system are as follows:
Operating System:
A 64-bit operating system, such as Windows 10 or Ubuntu, for compatibility with modern deep learning frameworks.

#### Development Environment:
Python programming language (version 3.6 or later) for coding the OCR system.

#### Deep Learning Frameworks:
Installation of deep learning frameworks, including ultralytics, to leverage pre-trained models and facilitate model training.

#### OCR Libraries:
Integration of OCR libraries, such as EasyOCR, to incorporate advanced text recognition capabilities.

#### Image Processing Libraries:
Usage of image processing libraries like OpenCV for preprocessing tasks and efficient handling of images.

#### Document Management:
Document management tools (e.g., PyMuPDF or PyPDF2) for handling PDF documents and file operations.

#### Version Control:
Implementation of version control using Git for collaborative development and code management.

#### Integrated Development Environment (IDE):
Selection of a suitable IDE, such as VSCode or PyCharm, for code development and debugging.


## Program

#### Main.py:
```python

from ultralytics import YOLO
import cv2
import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv

results = {}
mot_tracker = Sort()
# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./models/license_plate_detector.pt')
# load video
cap = cv2.VideoCapture('./sample.mp4')
vehicles = [2, 3, 5, 7]
         # read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])
        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))
        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
            if car_id != -1:
                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}
# write results
write_csv(results, './test.csv')


```
#### Util.py:
```python
 def read_license_plate(license_plate_crop):

    detections = reader.readtext(license_plate_crop)
    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')
        if license_complies_format(text):
            return format_license(text), score
    return None, None

def get_car(license_plate, vehicle_track_ids):

    x1, y1, x2, y2, score, class_id = license_plate
    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break
    if foundIt:
        return vehicle_track_ids[car_indx]
    return -1, -1, -1, -1, -1

```

## Output
In this section, we present the outcomes and achievements of the implemented system. The project's success is evaluated based on its ability to accurately detect vehicles, also the vehicle tracking is done   and text recognition from the number plate will be done and store the extracted number for further fee processing done in the toll gate .

### Output Video after Detection/tracking and number extraction

#### Visualization of Detection/tracking of vehicle and extracted number

![Screenshot 2023-11-26 091503](https://github.com/TamilVenthanRS/Enhancing-toll-gate-management-system-using-computer-vision-and-image-processing/assets/75235477/9f60e991-7614-4dc5-802f-4e4fde2e2c21)


#### The extracted number from the number plate by EasyOCR processing

![Screenshot 2023-11-16 193307](https://github.com/TamilVenthanRS/Enhancing-toll-gate-management-system-using-computer-vision-and-image-processing/assets/75235477/4af8eb58-e50d-459e-98de-2a5d0a44d4f3)


### Storing Of Extracted Number From the Number Plate
The information of each vehicle in each frame is stored with the vehicle and number plate bounding boxes, and the extracted number and the confidence score for the extraction is stored with respect to its respective vehicles.

![Screenshot 2023-11-16 192745](https://github.com/TamilVenthanRS/Enhancing-toll-gate-management-system-using-computer-vision-and-image-processing/assets/75235477/98a03ac8-61ae-418c-8fa5-0959e7186f92)

####  After filtering out the Best extracted Number
The filtering process for the number takes place due to  false numbers that were extracted by the help of the confidence score given by the EasyOCR.

![Screenshot 2023-11-16 192935](https://github.com/TamilVenthanRS/Enhancing-toll-gate-management-system-using-computer-vision-and-image-processing/assets/75235477/cde7eed9-2c00-4f98-b9c5-1aeb35c2db76)

## Result

In conclusion, the implementation of an advanced toll gate management system, as outlined in this project, represents a significant leap forward in optimizing traditional toll collection processes. By seamlessly integrating cutting-edge technologies such as YOLOv8 for vehicle detection, tracking, and number plate identification, alongside the EasyOCR technique for efficient number extraction, this system eliminates the need for vehicles to halt at toll booths. The real-time nature of the system not only enhances the accuracy of fee collection but also contributes to a substantial reduction in traffic congestion and delays.
#### Key Achievements

Seamless and Efficient Toll Collection: The toll gate management system achieves a key milestone by enabling fee collection without the need for vehicles to come to a complete stop. This innovation not only optimizes traffic flow but also minimizes congestion at toll plazas, providing a more efficient and streamlined toll collection process. 
Real-time Vehicle Tracking and Number Plate Recognition: The integration of YOLOv8 for real-time vehicle detection and tracking, coupled with its application for number plate recognition, represents a significant achievement. This ensures accurate and continuous monitoring of vehicles entering the toll plaza, enhancing the overall precision and reliability of the toll gate management system. 
Scalability and Adaptability: One of the system's key achievements lies in its scalability and adaptability to diverse toll plazas and infrastructural settings. The modular architecture allows for seamless integration, making it a versatile solution that can be implemented across various locations with different traffic patterns, demonstrating its broad applicability and potential for widespread use in toll gate management systems.

#### Future Enhancements

Integration of Advanced Machine Learning Models: Future enhancements could involve integrating even more advanced machine learning models for improved vehicle detection and tracking. Exploring models that incorporate deep learning techniques, such as attention mechanisms or advanced feature representations, could further enhance the system's accuracy and adaptability to diverse traffic scenarios. 
Incorporation of Vehicle Recognition for Automated Toll Classification: Future iterations of the project could explore the integration of vehicle recognition technologies to automatically classify vehicles based on characteristics such as size, weight, or type. This information could be leveraged to implement differential toll rates, encouraging the use of eco-friendly vehicles or optimizing toll collection for various vehicle categories.

#### Final Thoughts

This project utilizing YOLOv8 and EasyOCR, redefines conventional toll collection. Enabling seamless, real-time fee collection without vehicle stops, it optimizes traffic flow and lays the foundation for intelligent transportation systems, helps to avoid the pollution , gives a user satisfaction and showcases the potential of advanced technologies in enhancing urban mobility.
