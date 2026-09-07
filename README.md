# PPE Detection System

A computer vision-based Personal Protective Equipment (PPE) detection
and compliance monitoring system built using YOLO11.

## Features

- PPE object detection using YOLO11
- Person-level PPE association
- Helmet detection
- No-Helmet detection
- Vest detection
- No-Vest detection
- Safety-Shoes detection
- No-Safety-Shoes detection
- Safety-Goggles detection
- Without-Safety-Goggles detection
- BoT-SORT based tracking for video
- Image and video processing modes
- Person-level PPE status reporting

## PPE Classes

The current YOLO11 model contains 9 classes:

1. Helmet
2. No-Helmet
3. No-Vest
4. Person
5. Vest
6. Safety-Shoes
7. No-Safety-Shoes
8. Without-Safety-Goggles
9. With-Safety-Goggles

## Project Structure

```text
PPE_Detection_System_Final/
│
├── configs/
├── models/
├── src/
├── inputs/
├── outputs/
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
