#!/usr/bin/env python3
import os
import datetime
import sys
import cv2
import time
import numpy as np
import torch
import threading
import csv
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QLabel, QPushButton, QComboBox, QFileDialog, QSlider, QCheckBox,
                           QTabWidget, QGroupBox, QGridLayout, QProgressBar, QSpinBox,
                           QDoubleSpinBox, QSplitter, QFrame, QStatusBar, QMessageBox, QLineEdit)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont, QColor, QPalette
import torch
torch.backends.cudnn.benchmark = True
# Import our modules
from detection_model import ObjectDetector
from depth_model import DepthEstimator
from bbox3d_utils import BBox3DEstimator, BirdEyeView
try:
    from segmentation_model import SegmentationModel
    SEGMENTATION_AVAILABLE = True
except ImportError:
    SEGMENTATION_AVAILABLE = False
    print("Segmentation module not available")
from detection_history import DetectionHistory  # Make sure this import is present

# Worker thread for video processing
class VideoProcessingThread(QThread):
    # Define signals
    frame_ready = pyqtSignal(dict)
    processing_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)
    status_update = pyqtSignal(str)
    fps_update = pyqtSignal(float)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.running = False
        self.cap = None
        self.frame_count = 0
        self.start_time = None
        self.detector = None
        self.depth_estimator = None
        self.segmenter = None
        self.bbox3d_estimator = None
        self.bev = None

    def initialize_models(self):
        try:
            # Initialize detector
            self.status_update.emit("Initializing object detector...")
            self.detector = ObjectDetector(
                model_size=self.config['yolo_model_size'],
                conf_thres=self.config['conf_threshold'],
                iou_thres=self.config['iou_threshold'],
                classes=self.config['classes'],
                device=self.config['device']
            )
            
            # Initialize depth estimator
            self.status_update.emit("Initializing depth estimator...")
            self.depth_estimator = DepthEstimator(
                model_size=self.config['depth_model_size'],
                device=self.config['device']
            )
            
            # Initialize segmentation if enabled
            if self.config['enable_segmentation'] and SEGMENTATION_AVAILABLE:
                self.status_update.emit("Initializing segmentation model...")
                self.segmenter = SegmentationModel(
                    model_name=self.config['sam_model_name'],
                    device=self.config['device']
                )
            
            # Initialize 3D bbox estimator
            self.bbox3d_estimator = BBox3DEstimator()
            
            # Initialize BEV if enabled
            if self.config['enable_bev']:
                self.bev = BirdEyeView(scale=60, size=(300, 300))
                
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"Error initializing models: {str(e)}")
            return False

    def run(self):
        self.running = True
        self.frame_count = 0
        self.start_time = time.time()
        
        # Initialize models
        if not self.initialize_models():
            self.running = False
            self.processing_finished.emit()
            return
        
        # Open video source
        try:
            source = self.config['source']
            if isinstance(source, str) and source.isdigit():
                source = int(source)
                
            self.status_update.emit(f"Opening video source: {source}")
            self.cap = cv2.VideoCapture(source)
            
            if not self.cap.isOpened():
                self.error_occurred.emit(f"Error: Could not open video source {source}")
                self.running = False
                self.processing_finished.emit()
                return
                
            # Get video properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            if fps == 0:  # Sometimes happens with webcams
                fps = 30
                
            self.status_update.emit(f"Video source opened: {width}x{height} @ {fps}fps")
            
            # Main processing loop
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    if isinstance(source, str):  # File ended
                        self.status_update.emit("End of video file reached")
                        break
                    continue  # Skip frame if camera has issues
                
                # Process frame
                result = self.process_frame(frame)
                
                # Emit processed frame
                self.frame_ready.emit(result)
                
                # Calculate FPS
                self.frame_count += 1
                if self.frame_count % 10 == 0:
                    elapsed_time = time.time() - self.start_time
                    fps_value = self.frame_count / elapsed_time
                    self.fps_update.emit(fps_value)
                
            # Clean up
            self.cap.release()
            self.status_update.emit("Processing completed")
            
        except Exception as e:
            self.error_occurred.emit(f"Error during processing: {str(e)}")
        
        finally:
            self.running = False
            self.processing_finished.emit()

    def process_frame(self, frame):
        """Process a single frame and return all visualization results"""
        try:
            # Create copies for different visualizations
            original_frame = frame.copy()
            detection_frame = frame.copy()
            depth_frame = frame.copy()
            result_frame = frame.copy()
            segmentation_frame = frame.copy() if self.config['enable_segmentation'] and SEGMENTATION_AVAILABLE else None
            
            # Step 1: Object Detection
            try:
                detection_frame, detections = self.detector.detect(
                    detection_frame, 
                    track=self.config['enable_tracking']
                )
            except Exception as e:
                print(f"Error during object detection: {e}")
                detections = []
                cv2.putText(detection_frame, "Detection Error", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Step 1.5: Instance Segmentation (if enabled)
            segmentation_results = []
            if self.config['enable_segmentation'] and SEGMENTATION_AVAILABLE and self.segmenter and detections:
                try:
                    # Extract bounding boxes for segmentation
                    boxes = [detection[0] for detection in detections]
                    
                    # Generate segmentation masks
                    segmentation_results = self.segmenter.segment_with_boxes(original_frame, boxes)
                    
                    # Combine with detection information
                    segmentation_frame, _ = self.segmenter.combine_with_detection(
                        original_frame, detections
                    )
                except Exception as e:
                    print(f"Error during segmentation: {e}")
                    cv2.putText(segmentation_frame, "Segmentation Error", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Step 2: Depth Estimation
            try:
                depth_map = self.depth_estimator.estimate_depth(original_frame)
                depth_colored = self.depth_estimator.colorize_depth(depth_map)
            except Exception as e:
                print(f"Error during depth estimation: {e}")
                # Create a dummy depth map
                h, w = original_frame.shape[:2]
                depth_map = np.zeros((h, w), dtype=np.float32)
                depth_colored = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(depth_colored, "Depth Error", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Step 3: 3D Bounding Box Estimation
            boxes_3d = []
            active_ids = []
            
            for detection in detections:
                try:
                    bbox, score, class_id, obj_id = detection
                    
                    # Get class name
                    class_name = self.detector.get_class_names()[class_id]
                    
                    # Get depth in the region of the bounding box
                    if class_name.lower() in ['person', 'cat', 'dog']:
                        # For people and animals, use the center point depth
                        center_x = int((bbox[0] + bbox[2]) / 2)
                        center_y = int((bbox[1] + bbox[3]) / 2)
                        depth_value = self.depth_estimator.get_depth_at_point(depth_map, center_x, center_y)
                        depth_method = 'center'
                    else:
                        # For other objects, use the median depth in the region
                        depth_value = self.depth_estimator.get_depth_in_region(depth_map, bbox, method='median')
                        depth_method = 'median'
                    
                    # Create a simplified 3D box representation
                    box_3d = {
                        'bbox_2d': bbox,
                        'depth_value': depth_value,
                        'depth_method': depth_method,
                        'class_name': class_name,
                        'object_id': obj_id,
                        'score': score
                    }
                    
                    # Add segmentation mask if available
                    if self.config['enable_segmentation'] and SEGMENTATION_AVAILABLE:
                        # Find matching segmentation result
                        for seg_result in segmentation_results:
                            if np.array_equal(seg_result.get('bbox', None), bbox):
                                box_3d['mask'] = seg_result['mask']
                                break
                    
                    boxes_3d.append(box_3d)
                    
                    # Keep track of active IDs for tracker cleanup
                    if obj_id is not None:
                        active_ids.append(obj_id)
                except Exception as e:
                    print(f"Error processing detection: {e}")
                    continue
            
            # Clean up trackers for objects that are no longer detected
            self.bbox3d_estimator.cleanup_trackers(active_ids)
            
            # Step 4: Visualization
            # Determine which frame to use as the base for result
            if self.config['enable_segmentation'] and SEGMENTATION_AVAILABLE and segmentation_frame is not None:
                # Use segmentation frame if available
                result_frame = segmentation_frame.copy()
            else:
                # Otherwise use detection frame
                result_frame = detection_frame.copy()
            
            # Draw boxes on the result frame
            for box_3d in boxes_3d:
                try:
                    # Determine color based on class
                    class_name = box_3d['class_name'].lower()
                    if 'car' in class_name or 'vehicle' in class_name:
                        color = (0, 0, 255)  # Red
                    elif 'person' in class_name:
                        color = (0, 255, 0)  # Green
                    elif 'bicycle' in class_name or 'motorcycle' in class_name:
                        color = (255, 0, 0)  # Blue
                    elif 'potted plant' in class_name or 'plant' in class_name:
                        color = (0, 255, 255)  # Yellow
                    else:
                        color = (255, 255, 255)  # White
                    
                    # Draw box with depth information
                    result_frame = self.bbox3d_estimator.draw_box_3d(result_frame, box_3d, color=color)
                except Exception as e:
                    print(f"Error drawing box: {e}")
                    continue
            
            # Draw Bird's Eye View if enabled
            bev_image = None
            if self.config['enable_bev'] and self.bev:
                try:
                    # Reset BEV and draw objects
                    self.bev.reset()
                    for box_3d in boxes_3d:
                        self.bev.draw_box(box_3d)
                    bev_image = self.bev.get_image()
                except Exception as e:
                    print(f"Error drawing BEV: {e}")
            
            # Return all results
            return {
                'original': original_frame,
                'detection': detection_frame,
                'depth_map': depth_map,
                'depth_colored': depth_colored,
                'segmentation': segmentation_frame,
                'result': result_frame,
                'bev': bev_image,
                'boxes_3d': boxes_3d,
                'detections': detections
            }
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return {
                'original': frame,
                'error': str(e)
            }

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()


class YOLO3DGui(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Set window properties
        self.setWindowTitle("YOLO-3D: Advanced 3D Object Detection")
        self.setMinimumSize(1280, 800)
        
        # Initialize variables
        self.video_thread = None
        self.current_fps = 0
        self.processing_video = False
        self.output_video = None
        self.current_frame = None
        
        # Default configuration
        self.config = {
            'source': None,
            'output_path': "output.mp4",
            'yolo_model_size': "nano",
            'depth_model_size': "small",
            'sam_model_name': "sam2_b.pt",
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'conf_threshold': 0.25,
            'iou_threshold': 0.45,
            'classes': None,
            'enable_tracking': True,
            'enable_bev': True,
            'enable_pseudo_3d': True,
            'enable_segmentation': SEGMENTATION_AVAILABLE,
            'record_output': False
        }
        
        # Create central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create splitter for resizable panels
        self.splitter = QSplitter(Qt.Horizontal)
        
        # Create left panel (controls)
        self.control_panel = self.create_control_panel()
        self.splitter.addWidget(self.control_panel)
        
        # Create right panel (visualization)
        self.visualization_panel = self.create_visualization_panel()
        self.splitter.addWidget(self.visualization_panel)
        
        # Set initial splitter sizes
        self.splitter.setSizes([300, 980])
        
        # Add splitter to main layout
        self.main_layout.addWidget(self.splitter)
        
        # Create status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        
        # Status bar elements
        self.status_label = QLabel("Ready")
        self.fps_label = QLabel("FPS: --")
        self.device_label = QLabel(f"Device: {self.config['device']}")
        
        # Add elements to status bar
        self.statusBar.addWidget(self.status_label, 3)
        self.statusBar.addPermanentWidget(self.fps_label, 1)
        self.statusBar.addPermanentWidget(self.device_label, 1)
        
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2D2D30;
                color: #E0E0E0;
            }
            QWidget {
                background-color: #2D2D30;
                color: #E0E0E0;
            }
            QPushButton {
                background-color: #0078D7;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1C97EA;
            }
            QPushButton:pressed {
                background-color: #00559E;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
            QComboBox, QSlider, QSpinBox, QDoubleSpinBox {
                background-color: #3E3E42;
                color: #E0E0E0;
                border: 1px solid #555555;
                padding: 4px;
                border-radius: 3px;
            }
            QLabel {
                color: #E0E0E0;
            }
            QCheckBox {
                color: #E0E0E0;
            }
            QGroupBox {
                border: 1px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                color: #E0E0E0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
            }
            QTabWidget::pane {
                border: 1px solid #555555;
                border-radius: 5px;
            }
            QTabBar::tab {
                background-color: #3E3E42;
                color: #E0E0E0;
                padding: 8px 20px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #0078D7;
            }
            QTabBar::tab:hover:!selected {
                background-color: #505054;
            }
            QStatusBar {
                background-color: #1E1E1E;
                color: #E0E0E0;
            }
            QScrollArea {
                border: none;
            }
            QSplitter::handle {
                background-color: #3E3E42;
            }
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 3px;
                text-align: center;
                color: #E0E0E0;
                background-color: #3E3E42;
            }
            QProgressBar::chunk {
                background-color: #0078D7;
                width: 1px;
            }
        """)
        
        # Initialize timers
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_display)
        
        # Show the window
        self.show()
        
        self.detection_history = DetectionHistory()  # <-- Add this line
        self.detection_csv_path = "detection_history.csv"  # <-- Add this line
        self.last_logged_class_id = None  # <-- Add this line
        
    def create_control_panel(self):
        """Create the left control panel"""
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        # Input/Output group
        io_group = QGroupBox("Input/Output")
        io_layout = QVBoxLayout(io_group)
        
        # Input file selection
        input_layout = QHBoxLayout()
        self.input_label = QLabel("Input:")
        self.input_path = QLabel("No file selected")
        self.input_path.setWordWrap(True)
        self.input_browse = QPushButton("Browse")
        self.input_browse.clicked.connect(self.browse_input)
        self.camera_button = QPushButton("Use Camera")
        self.camera_button.clicked.connect(self.use_camera)
        self.http_label = QLabel("HTTP URL:")
        self.http_input = QLineEdit()
        self.http_input.setPlaceholderText("http://<ip>:<port>/video")
        self.http_button = QPushButton("Use HTTP Camera")
        self.http_button.clicked.connect(self.use_http_camera)

        input_layout.addWidget(self.input_label)
        input_layout.addWidget(self.input_path, 1)

        input_buttons = QHBoxLayout()
        input_buttons.addWidget(self.input_browse)
        input_buttons.addWidget(self.camera_button)
        input_buttons.addWidget(self.http_label)
        input_buttons.addWidget(self.http_input)
        input_buttons.addWidget(self.http_button)

        io_layout.addLayout(input_layout)
        io_layout.addLayout(input_buttons)
        
        # Output file selection
        output_layout = QHBoxLayout()
        self.output_checkbox = QCheckBox("Record Output:")
        self.output_checkbox.stateChanged.connect(self.toggle_output_recording)
        self.output_path = QLabel("output.mp4")
        self.output_path.setEnabled(False)
        self.output_browse = QPushButton("Browse")
        self.output_browse.clicked.connect(self.browse_output)
        self.output_browse.setEnabled(False)
        
        output_layout.addWidget(self.output_checkbox)
        output_layout.addWidget(self.output_path, 1)
        output_layout.addWidget(self.output_browse)
        
        io_layout.addLayout(output_layout)
        
        # Add to control layout
        control_layout.addWidget(io_group)
        
        # Models group
        models_group = QGroupBox("Models")
        models_layout = QGridLayout(models_group)
        
        # YOLO model selection
        models_layout.addWidget(QLabel("YOLO Model:"), 0, 0)
        self.yolo_combo = QComboBox()
        self.yolo_combo.addItems(["nano", "small", "medium", "large", "extra"])
        self.yolo_combo.setCurrentText(self.config['yolo_model_size'])
        self.yolo_combo.currentTextChanged.connect(self.update_config)
        models_layout.addWidget(self.yolo_combo, 0, 1)
        
        # Depth model selection
        models_layout.addWidget(QLabel("Depth Model:"), 1, 0)
        self.depth_combo = QComboBox()
        self.depth_combo.addItems(["small", "base", "large"])
        self.depth_combo.setCurrentText(self.config['depth_model_size'])
        self.depth_combo.currentTextChanged.connect(self.update_config)
        models_layout.addWidget(self.depth_combo, 1, 1)
        
        # SAM model selection (if available)
        if SEGMENTATION_AVAILABLE:
            models_layout.addWidget(QLabel("SAM Model:"), 2, 0)
            self.sam_combo = QComboBox()
            self.sam_combo.addItems(["sam2_s.pt","sam2_b.pt", "sam2_l.pt", "sam2_h.pt"])
            self.sam_combo.setCurrentText(self.config['sam_model_name'])
            self.sam_combo.currentTextChanged.connect(self.update_config)
            models_layout.addWidget(self.sam_combo, 2, 1)
        
        # Device selection
        models_layout.addWidget(QLabel("Device:"), 3, 0)
        self.device_combo = QComboBox()
        available_devices = ["cpu", "NVIDIA GeForce RTX 3050 (CUDA)"]
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            available_devices.append("mps")
        self.device_combo.addItems(available_devices)
        # Set current device based on config
        if self.config['device'] == 'cuda':
            self.device_combo.setCurrentText("NVIDIA GeForce RTX 3050 (CUDA)")
        else:
            self.device_combo.setCurrentText(self.config['device'])
        self.device_combo.currentTextChanged.connect(self.update_config)
        models_layout.addWidget(self.device_combo, 3, 1)
        
        # Add to control layout
        control_layout.addWidget(models_group)
        
        # Detection Settings group
        detection_group = QGroupBox("Detection Settings")
        detection_layout = QGridLayout(detection_group)
        
        # Confidence threshold
        detection_layout.addWidget(QLabel("Confidence:"), 0, 0)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(1)
        self.conf_slider.setMaximum(99)
        self.conf_slider.setValue(int(self.config['conf_threshold'] * 100))
        self.conf_slider.valueChanged.connect(self.update_config)
        self.conf_value = QLabel(f"{self.config['conf_threshold']:.2f}")
        detection_layout.addWidget(self.conf_slider, 0, 1)
        detection_layout.addWidget(self.conf_value, 0, 2)
        
        # IoU threshold
        detection_layout.addWidget(QLabel("IoU:"), 1, 0)
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setMinimum(1)
        self.iou_slider.setMaximum(99)
        self.iou_slider.setValue(int(self.config['iou_threshold'] * 100))
        self.iou_slider.valueChanged.connect(self.update_config)
        self.iou_value = QLabel(f"{self.config['iou_threshold']:.2f}")
        detection_layout.addWidget(self.iou_slider, 1, 1)
        detection_layout.addWidget(self.iou_value, 1, 2)
        
        # Add to control layout
        control_layout.addWidget(detection_group)
        
        # Features group
        features_group = QGroupBox("Features")
        features_layout = QVBoxLayout(features_group)
        
        # Tracking
        self.tracking_check = QCheckBox("Enable Tracking")
        self.tracking_check.setChecked(self.config['enable_tracking'])
        self.tracking_check.stateChanged.connect(self.update_config)
        features_layout.addWidget(self.tracking_check)
        
        # Bird's Eye View
        self.bev_check = QCheckBox("Enable Bird's Eye View")
        self.bev_check.setChecked(self.config['enable_bev'])
        self.bev_check.stateChanged.connect(self.update_config)
        features_layout.addWidget(self.bev_check)
        
        # 3D Visualization
        self.pseudo_3d_check = QCheckBox("Enable 3D Visualization")
        self.pseudo_3d_check.setChecked(self.config['enable_pseudo_3d'])
        self.pseudo_3d_check.stateChanged.connect(self.update_config)
        features_layout.addWidget(self.pseudo_3d_check)
        
        # Segmentation (if available)
        if SEGMENTATION_AVAILABLE:
            self.segmentation_check = QCheckBox("Enable Segmentation")
            self.segmentation_check.setChecked(self.config['enable_segmentation'])
            self.segmentation_check.stateChanged.connect(self.update_config)
            features_layout.addWidget(self.segmentation_check)
        
        # Add to control layout
        control_layout.addWidget(features_group)
        
        # Actions group
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)
        
        # Start/Stop buttons
        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self.start_processing)
        self.start_button.setEnabled(False)
        
        self.stop_button = QPushButton("Stop Processing")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        
        self.save_frame_button = QPushButton("Save Current Frame")
        self.save_frame_button.clicked.connect(self.save_current_frame)
        self.save_frame_button.setEnabled(False)
        
        actions_layout.addWidget(self.start_button)
        actions_layout.addWidget(self.stop_button)
        actions_layout.addWidget(self.save_frame_button)
        
        # Add to control layout
        control_layout.addWidget(actions_group)
        
        # Add a stretch to push everything up
        control_layout.addStretch()
        
        return control_panel
        
    def create_visualization_panel(self):
        """Create the right visualization panel"""
        visualization_panel = QWidget()
        vis_layout = QVBoxLayout(visualization_panel)
        
        # Create tab widget for different views
        self.tab_widget = QTabWidget()
        
        # Main visualization tab
        main_tab = QWidget()
        main_layout = QVBoxLayout(main_tab)
        
        # Main visualization area
        self.main_view = QLabel("No video loaded")
        self.main_view.setAlignment(Qt.AlignCenter)
        self.main_view.setStyleSheet("background-color: #1E1E1E; color: white;")
        self.main_view.setMinimumSize(800, 450)
        main_layout.addWidget(self.main_view)
        
        # Add row of smaller views
        small_views_layout = QHBoxLayout()
        
        # Original frame
        original_box = QGroupBox("Original")
        original_layout = QVBoxLayout(original_box)
        self.original_view = QLabel()
        self.original_view.setAlignment(Qt.AlignCenter)
        self.original_view.setStyleSheet("background-color: #1E1E1E;")
        self.original_view.setMinimumSize(200, 150)
        original_layout.addWidget(self.original_view)
        small_views_layout.addWidget(original_box)
        
        # Depth map
        depth_box = QGroupBox("Depth")
        depth_layout = QVBoxLayout(depth_box)
        self.depth_view = QLabel()
        self.depth_view.setAlignment(Qt.AlignCenter)
        self.depth_view.setStyleSheet("background-color: #1E1E1E;")
        self.depth_view.setMinimumSize(200, 150)
        depth_layout.addWidget(self.depth_view)
        small_views_layout.addWidget(depth_box)
        
        # Detection view
        detection_box = QGroupBox("Detection")
        detection_layout = QVBoxLayout(detection_box)
        self.detection_view = QLabel()
        self.detection_view.setAlignment(Qt.AlignCenter)
        self.detection_view.setStyleSheet("background-color: #1E1E1E;")
        self.detection_view.setMinimumSize(200, 150)
        detection_layout.addWidget(self.detection_view)
        small_views_layout.addWidget(detection_box)
        
        # Bird's Eye View
        bev_box = QGroupBox("Bird's Eye View")
        bev_layout = QVBoxLayout(bev_box)
        self.bev_view = QLabel()
        self.bev_view.setAlignment(Qt.AlignCenter)
        self.bev_view.setStyleSheet("background-color: #1E1E1E;")
        self.bev_view.setMinimumSize(200, 150)
        bev_layout.addWidget(self.bev_view)
        small_views_layout.addWidget(bev_box)
        
        main_layout.addLayout(small_views_layout)
        
        # Add the main tab
        self.tab_widget.addTab(main_tab, "Main View")
        
        # All Views tab for side-by-side comparison
        all_views_tab = QWidget()
        all_views_layout = QGridLayout(all_views_tab)
        
        # Create additional view labels for the all views tab
        self.all_original_view = QLabel()
        self.all_original_view.setAlignment(Qt.AlignCenter)
        self.all_original_view.setStyleSheet("background-color: #1E1E1E;")
        all_views_layout.addWidget(QLabel("Original"), 0, 0)
        all_views_layout.addWidget(self.all_original_view, 1, 0)
        
        self.all_detection_view = QLabel()
        self.all_detection_view.setAlignment(Qt.AlignCenter)
        self.all_detection_view.setStyleSheet("background-color: #1E1E1E;")
        all_views_layout.addWidget(QLabel("Detection"), 0, 1)
        all_views_layout.addWidget(self.all_detection_view, 1, 1)
        
        self.all_depth_view = QLabel()
        self.all_depth_view.setAlignment(Qt.AlignCenter)
        self.all_depth_view.setStyleSheet("background-color: #1E1E1E;")
        all_views_layout.addWidget(QLabel("Depth Map"), 2, 0)
        all_views_layout.addWidget(self.all_depth_view, 3, 0)
        
        self.all_result_view = QLabel()
        self.all_result_view.setAlignment(Qt.AlignCenter)
        self.all_result_view.setStyleSheet("background-color: #1E1E1E;")
        all_views_layout.addWidget(QLabel("Result"), 2, 1)
        all_views_layout.addWidget(self.all_result_view, 3, 1)
        
        # Add the all views tab
        self.tab_widget.addTab(all_views_tab, "All Views")
        
        # Add tabs for more views if segmentation is available
        if SEGMENTATION_AVAILABLE:
            # Segmentation tab
            seg_tab = QWidget()
            seg_layout = QVBoxLayout(seg_tab)
            
            self.segmentation_view = QLabel()
            self.segmentation_view.setAlignment(Qt.AlignCenter)
            self.segmentation_view.setStyleSheet("background-color: #1E1E1E;")
            self.segmentation_view.setMinimumSize(800, 600)
            seg_layout.addWidget(self.segmentation_view)
            
            self.tab_widget.addTab(seg_tab, "Segmentation")
        
        # Add the tab widget to the layout
        vis_layout.addWidget(self.tab_widget)
        
        return visualization_panel
    
    def update_config(self):
        """Update config from UI controls"""
        # Update model selections
        self.config['yolo_model_size'] = self.yolo_combo.currentText()
        self.config['depth_model_size'] = self.depth_combo.currentText()
        if SEGMENTATION_AVAILABLE and hasattr(self, 'sam_combo'):
            self.config['sam_model_name'] = self.sam_combo.currentText()
        
        # Update device
        selected_device = self.device_combo.currentText()
        if selected_device == "NVIDIA GeForce RTX 3050 (CUDA)":
            self.config['device'] = 'cuda'
        else:
            self.config['device'] = selected_device
        self.device_label.setText(f"Device: {self.device_combo.currentText()}")
        
        # Update thresholds
        self.config['conf_threshold'] = self.conf_slider.value() / 100
        self.conf_value.setText(f"{self.config['conf_threshold']:.2f}")
        
        self.config['iou_threshold'] = self.iou_slider.value() / 100
        self.iou_value.setText(f"{self.config['iou_threshold']:.2f}")
        
        # Update feature toggles
        self.config['enable_tracking'] = self.tracking_check.isChecked()
        self.config['enable_bev'] = self.bev_check.isChecked()
        self.config['enable_pseudo_3d'] = self.pseudo_3d_check.isChecked()
        
        if SEGMENTATION_AVAILABLE and hasattr(self, 'segmentation_check'):
            self.config['enable_segmentation'] = self.segmentation_check.isChecked()
    
    def toggle_output_recording(self, state):
        """Toggle output recording setting"""
        self.config['record_output'] = (state == Qt.Checked)
        self.output_path.setEnabled(self.config['record_output'])
        self.output_browse.setEnabled(self.config['record_output'])
    
    def browse_input(self):
        """Open file dialog to select input video"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        
        if file_path:
            self.config['source'] = file_path
            self.input_path.setText(file_path)
            self.status_label.setText(f"Loaded video: {file_path}")
            self.start_button.setEnabled(True)
    
    def use_camera(self):
        """Set webcam as input source"""
        self.config['source'] = 0  # Default camera
        self.input_path.setText("Camera (0)")
        self.status_label.setText("Using default camera")
        self.start_button.setEnabled(True)
    
    def use_http_camera(self):
        """Set HTTP camera as input source"""
        url = self.http_input.text().strip()
        if url.startswith("http"):
            self.config['source'] = url
            self.input_path.setText(f"HTTP Camera: {url}")
            self.status_label.setText(f"Using HTTP camera: {url}")
            self.start_button.setEnabled(True)
        else:
            QMessageBox.warning(self, "Invalid URL", "Please enter a valid HTTP camera URL.")
    
    def browse_output(self):
        """Open file dialog to select output path"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Output Video", "", 
            "Video Files (*.mp4 *.avi);;All Files (*)"
        )
        
        if file_path:
            self.config['output_path'] = file_path
            self.output_path.setText(file_path)
    
    def start_processing(self):
        """Start video processing"""
        if self.processing_video:
            return
        
        if self.config['source'] is None:
            QMessageBox.warning(self, "No Input", "Please select an input video or camera first.")
            return
        
        # Update UI
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.input_browse.setEnabled(False)
        self.camera_button.setEnabled(False)
        self.processing_video = True
        
        # Initialize video writer if recording is enabled
        if self.config['record_output']:
            self.statusBar.showMessage("Initializing output video...")
            # Video writer will be initialized after we get the first frame
        
        # Start processing thread
        self.video_thread = VideoProcessingThread(self.config)
        self.video_thread.frame_ready.connect(self.process_frame)
        self.video_thread.processing_finished.connect(self.on_processing_finished)
        self.video_thread.error_occurred.connect(self.on_error)
        self.video_thread.status_update.connect(self.update_status)
        self.video_thread.fps_update.connect(self.update_fps)
        
        # Start the thread
        self.video_thread.start()
        
        # Start update timer
        self.update_timer.start(30)  # Update at ~30 FPS
        
        # Update status
        self.status_label.setText("Processing started")
    
    def stop_processing(self):
        """Stop video processing"""
        if not self.processing_video:
            return
        
        # Stop thread
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait()
            self.video_thread = None
        
        # Stop timer
        self.update_timer.stop()
        
        # Clean up video writer
        if self.output_video is not None:
            self.output_video.release()
            self.output_video = None
        
        # Update UI
        self.processing_video = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.input_browse.setEnabled(True)
        self.camera_button.setEnabled(True)
        
        # Update status
        self.status_label.setText("Processing stopped")
    
    def on_processing_finished(self):
        """Called when processing is finished"""
        self.stop_processing()
    
    def on_error(self, error_message):
        """Called when an error occurs"""
        QMessageBox.critical(self, "Error", error_message)
        self.stop_processing()
    
    def update_status(self, message):
        """Update status label"""
        self.status_label.setText(message)
    
    def update_fps(self, fps):
        """Update FPS label"""
        self.current_fps = fps
        self.fps_label.setText(f"FPS: {fps:.1f}")
    
    def process_frame(self, result):
        """Process a frame from the video thread"""
        self.current_frame = result
        
        # Initialize video writer if needed
        if self.config['record_output'] and self.output_video is None and 'result' in result:
            h, w = result['result'].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.output_video = cv2.VideoWriter(
                self.config['output_path'], fourcc, 30, (w, h)
            )
        
        # Write frame to output video if recording
        if self.config['record_output'] and self.output_video is not None and 'result' in result:
            self.output_video.write(result['result'])
        
        # Enable frame saving
        self.save_frame_button.setEnabled(True)
    
        # --- Detection history logging ---
        if 'detections' in result and result['detections']:
            for det in result['detections']:
                bbox, score, class_id, obj_id = det
                if class_id in [0, 1, 2]:
                    # Only log if class_id is different from the last logged one
                    if class_id != self.last_logged_class_id:
                        x_center = int((bbox[0] + bbox[2]) / 2)
                        y_center = int((bbox[1] + bbox[3]) / 2)
                        detection_record = {
                            "timestamp": datetime.datetime.now().isoformat(),
                            "class_id": class_id,
                            "x": x_center,
                            "y": y_center
                        }
                        self.detection_history.add(detection_record)
                        file_exists = os.path.isfile(self.detection_csv_path)
                        with open(self.detection_csv_path, 'a', newline='') as csvfile:
                            writer = csv.DictWriter(csvfile, fieldnames=detection_record.keys())
                            if not file_exists:
                                writer.writeheader()
                            writer.writerow(detection_record)
                        self.last_logged_class_id = class_id
                break  # Only check the first detection per frame
        # --- End detection history logging ---

    def update_display(self):
        """Update display with current frame"""
        if self.current_frame is None:
            return
        
        # Convert frames to QImage and display them
        if 'result' in self.current_frame:
            self.update_image_display(self.main_view, self.current_frame['result'], size=(800, 450))
            self.update_image_display(self.all_result_view, self.current_frame['result'], size=(400, 300))
        
        if 'original' in self.current_frame:
            self.update_image_display(self.original_view, self.current_frame['original'], size=(200, 150))
            self.update_image_display(self.all_original_view, self.current_frame['original'], size=(400, 300))
        
        if 'detection' in self.current_frame:
            self.update_image_display(self.detection_view, self.current_frame['detection'], size=(200, 150))
            self.update_image_display(self.all_detection_view, self.current_frame['detection'], size=(400, 300))
        
        if 'depth_colored' in self.current_frame:
            self.update_image_display(self.depth_view, self.current_frame['depth_colored'], size=(200, 150))
            self.update_image_display(self.all_depth_view, self.current_frame['depth_colored'], size=(400, 300))
        
        if 'bev' in self.current_frame and self.current_frame['bev'] is not None:
            self.update_image_display(self.bev_view, self.current_frame['bev'], size=(200, 150))
        
        # Update segmentation view if available
        if SEGMENTATION_AVAILABLE and 'segmentation' in self.current_frame and self.current_frame['segmentation'] is not None:
            self.update_image_display(self.segmentation_view, self.current_frame['segmentation'], size=(800, 600))
    
    def update_image_display(self, label, image, size=None):
        """Update an image display label with a CV2 image"""
        if image is None:
            return
        
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to QImage
        if len(image.shape) == 2:  # Grayscale
            h, w = image.shape
            qimg = QImage(image.data, w, h, w, QImage.Format_Grayscale8)
        else:  # Color
            h, w, ch = image.shape
            bytes_per_line = ch * w
            qimg = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale if needed
        if size:
            pixmap = QPixmap.fromImage(qimg).scaled(
                size[0], size[1], Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        else:
            pixmap = QPixmap.fromImage(qimg)
        
        # Set pixmap
        label.setPixmap(pixmap)
    
    def save_current_frame(self):
        """Save the current frame to disk"""
        if self.current_frame is None or 'result' not in self.current_frame:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Frame", "", 
            "Image Files (*.png *.jpg);;All Files (*)"
        )
        
        if file_path:
            cv2.imwrite(file_path, self.current_frame['result'])
            self.status_label.setText(f"Frame saved to {file_path}")
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Stop processing if active
        if self.processing_video:
            self.stop_processing()
        
        # Accept the close event
        event.accept()


if __name__ == "__main__":
    # Set high DPI scaling
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # Create application
    app = QApplication(sys.argv)
    window = YOLO3DGui()
    sys.exit(app.exec_())