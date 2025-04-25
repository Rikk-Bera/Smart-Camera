# Smart-Camera
This article is a description of a latest AI-powered camera system that can track faces in real time, detect objects and detect fire. This project involves using the most recent technologies like computer vision, machine learning and voice recognition making it an interactive and flexible tool for monitoring as well as security purposes.

Key Features:

Face Tracking:
Utilizes OpenCV and Haar Cascades for detecting and tracking faces in real-time.
Displays bounding boxes around detected faces, making it suitable for security and surveillance applications.
Object Tracking:
Employs a pre-trained MobileNet SSD model to detect and track multiple objects.
Provides real-time object identification and tracking with bounding boxes and labels.
Fire Detection:
Integrates color-based segmentation techniques to detect the presence of fire.
Automatically sends email alerts when a fire is detected, ensuring timely notifications for emergency response.
Voice-Controlled Interface:
Uses the SpeechRecognition and Pyttsx3 libraries to create an intuitive voice-controlled user interface.
Allows users to start, stop, and switch between different tracking modes using simple voice commands.

Technical Stack:

Programming Language: Python
Libraries: OpenCV, NumPy, Threading, SMTP, SpeechRecognition, Pyttsx3
Models: Haar Cascades for face detection, MobileNet SSD for object detection
Integration: Real-time video feed from IP cameras

How It Works:

Voice Command: Users can issue voice commands to initiate face tracking, object tracking, or fire detection.
Real-Time Processing: The system captures video frames from the camera and processes them to detect faces, objects, or fire.
Multi-Mode Operation: Users can switch between different tracking modes seamlessly, by using the multi-threaded architecture.
Applications:
Security and Surveillance: Real-time monitoring of environments for enhanced security.
Industrial Safety: Early detection of fire hazards in industrial settings.
Smart Home Systems: Intelligent home monitoring and automation based on detected activities.
