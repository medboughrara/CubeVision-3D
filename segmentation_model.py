import os
import torch
import numpy as np
import cv2
from ultralytics import SAM
import matplotlib.pyplot as plt

class SegmentationModel:
    """
    Instance segmentation using Segment Anything Model via Ultralytics
    """
    def __init__(self, model_name="sam2_b.pt", device=None):
        """
        Initialize the segmentation model
        
        Args:
            model_name (str): Model name/path 
                SAM v1: 'sam_b.pt', 'sam_l.pt', 'sam_h.pt', 'mobile_sam.pt'
                SAM v2: 'sam2_t.pt', 'sam2_s.pt', 'sam2_b.pt', 'sam2_l.pt'
            device (str): Device to run inference on ('cuda', 'cpu', 'mps')
        """
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = device
        print(f"Using device: {self.device} for segmentation")
        
        # Load SAM model from Ultralytics
        try:
            self.model = SAM(model_name)
            self.model.to(device)
            print(f"Loaded SAM model: {model_name} on {self.device}")
            
            # Display model information
            self.model.info()
        except Exception as e:
            print(f"Error loading SAM model: {e}")
            raise
        
        # Initialize parameters
        self.mask_colors = self._generate_colors(100)  # Pre-generate colors for masks
    
    def _generate_colors(self, n):
        """
        Generate distinct colors for visualization
        
        Args:
            n (int): Number of colors to generate
            
        Returns:
            list: List of RGB colors
        """
        colors = []
        for i in range(n):
            # Use HSV color space to generate evenly distributed colors
            h = i / n
            s = 0.8 + (i % 3) * 0.1  # Slight variation in saturation
            v = 0.8 + (i % 5) * 0.05  # Slight variation in value
            
            # Convert HSV to RGB
            h_i = int(h * 6)
            f = h * 6 - h_i
            p = v * (1 - s)
            q = v * (1 - f * s)
            t = v * (1 - (1 - f) * s)
            
            if h_i == 0:
                r, g, b = v, t, p
            elif h_i == 1:
                r, g, b = q, v, p
            elif h_i == 2:
                r, g, b = p, v, t
            elif h_i == 3:
                r, g, b = p, q, v
            elif h_i == 4:
                r, g, b = t, p, v
            else:
                r, g, b = v, p, q
            
            # Convert to 0-255 range and add to list
            colors.append((int(r * 255), int(g * 255), int(b * 255)))
        
        return colors
    
    def segment_with_boxes(self, image, boxes):
        """
        Generate segmentation masks for the given boxes
        
        Args:
            image (numpy.ndarray): Input image (BGR format)
            boxes (list): List of bounding boxes [[x1, y1, x2, y2], ...]
            
        Returns:
            list: List of masks for each box
        """
        if len(boxes) == 0:
            return []
        
        results = []
        
        try:
            # Convert boxes to the correct format if needed
            processed_boxes = []
            for box in boxes:
                # Make sure box is a list of float values (not tensors)
                processed_box = [float(coord) if isinstance(coord, (int, float)) else float(coord.item()) 
                            for coord in box]
                processed_boxes.append(processed_box)
            
            # Run inference with bounding boxes prompt
            model_results = self.model(image, bboxes=processed_boxes, verbose=False)
            
            # Process the results
            for i, result in enumerate(model_results):
                # Extract masks
                masks = result.masks
                if masks is None or len(masks.data) == 0:
                    continue
                
                # Process each mask
                for j, mask_data in enumerate(masks.data):
                    # Convert mask tensor to numpy array
                    mask = mask_data.cpu().numpy()
                    
                    # Set default confidence since Masks object doesn't have a conf attribute
                    # The SAM implementation in Ultralytics doesn't provide confidence scores
                    confidence = 1.0
                    
                    # Store mask with metadata
                    results.append({
                        'mask': mask,
                        'score': confidence,
                        'bbox': processed_boxes[i] if i < len(processed_boxes) else None
                    })
        except Exception as e:
            print(f"Error in segmentation with boxes: {e}")
        
        return results
    
    def segment_with_points(self, image, points, labels=None):
        """
        Generate segmentation masks using point prompts
        
        Args:
            image (numpy.ndarray): Input image (BGR format)
            points (list): List of points [[x, y], ...] or nested list for multi-point prompts
            labels (list): List of labels (1 for foreground, 0 for background)
            
        Returns:
            list: List of masks
        """
        if not points:
            return []
        
        results = []
        
        try:
            # Convert points and labels to correct format if needed
            processed_points = []
            for point in points:
                # Make sure point is a list of float values (not tensors)
                processed_point = [float(coord) if isinstance(coord, (int, float)) else float(coord.item()) 
                                 for coord in point]
                processed_points.append(processed_point)
            
            processed_labels = None
            if labels is not None:
                processed_labels = [int(label) if isinstance(label, (int, float)) else int(label.item()) 
                                  for label in labels]
            
            # Run inference with point prompts
            model_results = self.model(image, points=processed_points, labels=processed_labels, verbose=False)
            
            # Process the results
            for result in model_results:
                # Extract masks
                masks = result.masks
                if masks is None or len(masks.data) == 0:
                    continue
                
                # Process each mask
                for j, mask_data in enumerate(masks.data):
                    # Convert mask tensor to numpy array
                    mask = mask_data.cpu().numpy()
                    
                    # Get confidence if available
                    confidence = float(masks.conf[j].item()) if masks.conf is not None else 1.0
                    
                    # Store mask with metadata
                    results.append({
                        'mask': mask,
                        'score': confidence,
                        'bbox': None  # No bbox for point-based segmentation
                    })
        except Exception as e:
            print(f"Error in segmentation with points: {e}")
        
        return results
    
    def overlay_masks(self, image, masks, alpha=0.3):
        """
        Overlay segmentation masks on image with improved visibility
        
        Args:
            image (numpy.ndarray): Input image (BGR format)
            masks (list): List of mask dictionaries from segment_with_boxes
            alpha (float): Transparency factor (lower value = more transparent masks)
            
        Returns:
            numpy.ndarray: Image with overlaid masks
        """
        # Make a copy of the image to avoid modifying the original
        result = image.copy()
        
        # Overlay each mask with appropriate transparency
        for i, mask_dict in enumerate(masks):
            mask = mask_dict['mask']
            color = self.mask_colors[i % len(self.mask_colors)]
            
            # Create a colored mask image
            colored_mask = np.zeros_like(image)
            colored_mask[mask] = color
            
            # Apply the mask with transparency
            mask_area = np.where(mask)
            if len(mask_area[0]) > 0:  # Only proceed if mask is not empty
                # Apply mask with transparency
                cv2.addWeighted(
                    colored_mask, alpha, 
                    result, 1.0, 
                    0, result
                )
            
            # Draw mask contour for better edge definition
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(result, contours, -1, color, 2)
        
        return result
    
    def process_video(self, source, output_path=None):
        """
        Process a video with segmentation
        
        Args:
            source (str): Path to input video
            output_path (str): Path to output video (None for no output)
            
        Returns:
            None
        """
        # Run inference on video
        results = self.model(source, verbose=True)
        
        # If output_path is specified, results are automatically saved
        if output_path:
            results.save(output_path)
    
    def combine_with_detection(self, image, detections):
        """
        Segment objects from detection results
        
        Args:
            image (numpy.ndarray): Input image (BGR format)
            detections (list): List of detections [bbox, score, class_id, object_id]
            
        Returns:
            tuple: (annotated_image, segmentation_results)
                - annotated_image (numpy.ndarray): Image with masks and detection info
                - segmentation_results (list): List of segmentation results with metadata
        """
        # Extract bounding boxes from detections
        boxes = [detection[0] for detection in detections]
        
        # Generate segmentation masks
        segmentation_results = self.segment_with_boxes(image, boxes)
        
        # Add detection metadata to segmentation results
        for i, result in enumerate(segmentation_results):
            if i < len(detections):
                _, score, class_id, object_id = detections[i]
                result['detection_score'] = float(score) if isinstance(score, (int, float)) else float(score.item())
                result['class_id'] = int(class_id) if isinstance(class_id, (int, float)) else int(class_id.item())
                result['object_id'] = int(object_id) if object_id is not None else None
        
        # Overlay masks on image
        annotated_image = self.overlay_masks(image, segmentation_results)
        
        # Add detection labels
        for i, detection in enumerate(detections):
            if i < len(segmentation_results):
                box, score, class_id, object_id = detection
                x1, y1, x2, y2 = map(int, box)
                
                # Add label
                label = f"ID: {int(object_id) if object_id is not None else 'N/A'} | Class: {class_id} | {float(score):.2f}"
                cv2.putText(annotated_image, label, 
                          (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_image, segmentation_results


# Example usage
if __name__ == "__main__":
    # Initialize model
    segmenter = SegmentationModel(model_name="sam2_b.pt")
    
    # Test on an image
    image = cv2.imread("test.jpg")
    
    # Example 1: Segment with bounding boxes
    boxes = [[100, 100, 300, 300], [400, 200, 600, 400]]
    results = segmenter.segment_with_boxes(image, boxes)
    annotated = segmenter.overlay_masks(image, results)
    cv2.imshow("Segmentation with Boxes", annotated)
    
    # Example 2: Segment with points
    points = [[250, 250], [500, 300]]
    labels = [1, 1]  # 1 for foreground points
    results = segmenter.segment_with_points(image, points, labels)
    annotated = segmenter.overlay_masks(image, results)
    cv2.imshow("Segmentation with Points", annotated)
    
    # Example 3: Process video
    # segmenter.process_video("test.mp4", "output.mp4")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()