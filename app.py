import streamlit as st
import cv2
import torch
import numpy as np
import time
import tempfile
from pathlib import Path

# Import detection utilities
from detection_utils import load_model, detect_objects, draw_boxes, ObjectTracker

def initialize_video_capture(input_source, video_file=None, url=None):
    """Initialize video capture and writer"""
    cap = None
    out = None
    output_path = None
    
    if input_source == "Video File" and video_file is not None:
        # Save uploaded file to temp location
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.read())
        tfile.flush()
        video_path = tfile.name
        
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        
        if cap.isOpened():
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Ensure valid FPS
            if fps <= 0:
                fps = 30
            
            # Create output path in a temporary directory
            temp_dir = tempfile.gettempdir()
            output_path = str(Path(temp_dir) / 'detected_output.mp4')
            
            # Try different codecs in order of preference
            codecs = [
                ('avc1', '.mp4'),
                ('mp4v', '.mp4'),
                ('XVID', '.avi')
            ]
            
            for codec, ext in codecs:
                try:
                    output_path = str(Path(temp_dir) / f'detected_output{ext}')
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    out = cv2.VideoWriter(
                        output_path,
                        fourcc,
                        fps,
                        (width, height),
                        isColor=True
                    )
                    
                    # Test if writer is working
                    if out.isOpened():
                        break
                except Exception:
                    continue
            
            if out is None or not out.isOpened():
                st.error("Failed to create video writer")
                return None, None, None
    
    elif input_source == "Live Stream URL" and url:
        cap = cv2.VideoCapture(url)
    
    return cap, out, output_path

def get_model_info():
    """Return information about available YOLO models"""
    return {
        'yolov8n.pt': {
            'name': 'YOLOv8 Nano',
            'description': 'Smallest and fastest model. Best for CPU or low-power devices.',
            'speed': '⚡⚡⚡⚡⚡',
            'accuracy': '⭐⭐',
            'size': '6.7 MB',
            'details': 'Ideal for real-time applications with limited computing power.'
        },
        'yolov8s.pt': {
            'name': 'YOLOv8 Small',
            'description': 'Small model balancing speed and accuracy.',
            'speed': '⚡⚡⚡⚡',
            'accuracy': '⭐⭐⭐',
            'size': '22.4 MB',
            'details': 'Good for general purpose detection with decent performance.'
        },
        'yolov8m.pt': {
            'name': 'YOLOv8 Medium',
            'description': 'Medium-sized model with good balance.',
            'speed': '⚡⚡⚡',
            'accuracy': '⭐⭐⭐⭐',
            'size': '52.2 MB',
            'details': 'Recommended for standard detection tasks with good GPU.'
        },
        'yolov8l.pt': {
            'name': 'YOLOv8 Large',
            'description': 'Large model with high accuracy.',
            'speed': '⚡⚡',
            'accuracy': '⭐⭐⭐⭐⭐',
            'size': '87.7 MB',
            'details': 'Best for high-accuracy requirements with good computing power.'
        },
        'yolov8x.pt': {
            'name': 'YOLOv8 XLarge',
            'description': 'Extra large model with highest accuracy.',
            'speed': '⚡',
            'accuracy': '⭐⭐⭐⭐⭐⭐',
            'size': '131.7 MB',
            'details': 'Best for tasks requiring maximum accuracy, requires powerful GPU.'
        }
    }

def main():
    st.title("Real-Time Object Detection")
    
    # Initialize session state
    if 'tracker' not in st.session_state:
        st.session_state.tracker = ObjectTracker()
    if 'cap' not in st.session_state:
        st.session_state.cap = None
    if 'out' not in st.session_state:
        st.session_state.out = None
    if 'output_path' not in st.session_state:
        st.session_state.output_path = None
    if 'processed_frames' not in st.session_state:
        st.session_state.processed_frames = 0
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = 'yolov8x.pt'
    if 'model' not in st.session_state:
        st.session_state.model = None
    
    # Sidebar settings
    st.sidebar.title("Settings")
    
    # Model selection
    st.sidebar.subheader("Model Selection")
    model_info = get_model_info()
    selected_model = st.sidebar.selectbox(
        "Choose YOLO Model",
        options=list(model_info.keys()),
        format_func=lambda x: model_info[x]['name'],
        index=list(model_info.keys()).index(st.session_state.selected_model)
    )
    
    # Display model information
    with st.sidebar.expander("Model Details", expanded=True):
        st.markdown(f"**{model_info[selected_model]['name']}**")
        st.write(model_info[selected_model]['description'])
        st.write(f"Speed: {model_info[selected_model]['speed']}")
        st.write(f"Accuracy: {model_info[selected_model]['accuracy']}")
        st.write(f"Size: {model_info[selected_model]['size']}")
        st.write(f"Details: {model_info[selected_model]['details']}")
    
    # Add Load Model button
    if st.sidebar.button("Load Selected Model"):
        with st.spinner(f"Loading {model_info[selected_model]['name']}..."):
            st.session_state.model = load_model(selected_model)
            st.session_state.selected_model = selected_model
            st.sidebar.success("Model loaded successfully!")
    
    # Detection confidence
    detection_confidence = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.5)
    
    # Input selection
    input_source = st.radio("Select Input Source", ["Video File", "Live Stream URL"])
    
    try:
        # Handle video input
        if input_source == "Video File":
            video_file = st.file_uploader("Upload Video", type=['mp4', 'avi'])
            if video_file is not None:
                st.session_state.cap, st.session_state.out, st.session_state.output_path = initialize_video_capture(input_source, video_file=video_file)
        else:
            url = st.text_input("Enter Stream URL")
            if url:
                st.session_state.cap, st.session_state.out, st.session_state.output_path = initialize_video_capture(input_source, url=url)
        
        if st.session_state.cap is not None and not st.session_state.cap.isOpened():
            st.error("Error: Could not open video source")
            st.stop()
        
        # Create placeholder for video display
        video_placeholder = st.empty()
        
        # Initialize frame buffer in session state
        if 'frame_buffer' not in st.session_state:
            st.session_state.frame_buffer = []
        
        # Control buttons - Move them to sidebar to avoid duplication
        st.sidebar.markdown("---")
        st.sidebar.subheader("Controls")
        start_button = st.sidebar.button("Start Detection")
        stop_button = st.sidebar.button("Stop Detection")
        
        if start_button:
            if st.session_state.model is None:
                st.error("Please load a model first using the 'Load Selected Model' button")
                st.stop()
            if st.session_state.cap is None:
                st.error("Please upload a video or provide a stream URL first")
                st.stop()
            st.session_state.run_detection = True
            st.session_state.processed_frames = 0
            st.session_state.frame_buffer = []  # Clear buffer on start
        if stop_button:
            st.session_state.run_detection = False
        
        # Detection loop
        while (hasattr(st.session_state, 'run_detection') and 
               st.session_state.run_detection and 
               st.session_state.cap is not None):
            
            ret, frame = st.session_state.cap.read()
            if not ret:
                break
            
            # Perform detection
            detections = detect_objects(st.session_state.model, frame, detection_confidence)
            
            # Draw boxes on frame
            annotated_frame = draw_boxes(frame, detections, st.session_state.tracker)
            
            # Add frame to buffer
            st.session_state.frame_buffer.append(annotated_frame)
            
            # Write frames to video periodically
            if len(st.session_state.frame_buffer) >= 30:  # Write every 30 frames
                for buffered_frame in st.session_state.frame_buffer:
                    if st.session_state.out is not None:
                        st.session_state.out.write(buffered_frame)
                        st.session_state.processed_frames += 1
                st.session_state.frame_buffer.clear()
            
            # Update display every 3rd frame
            if st.session_state.processed_frames % 3 == 0:
                video_placeholder.image(annotated_frame, channels="BGR")
            
            # Minimal sleep to prevent UI freezing
            time.sleep(0.001)
        
        # Write remaining frames in buffer
        if st.session_state.frame_buffer and st.session_state.out is not None:
            for buffered_frame in st.session_state.frame_buffer:
                st.session_state.out.write(buffered_frame)
                st.session_state.processed_frames += 1
            st.session_state.frame_buffer.clear()
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        raise e
    
    finally:
        # Ensure proper cleanup and save remaining frames
        if hasattr(st.session_state, 'frame_buffer') and st.session_state.frame_buffer and hasattr(st.session_state, 'out') and st.session_state.out is not None:
            for buffered_frame in st.session_state.frame_buffer:
                st.session_state.out.write(buffered_frame)
                st.session_state.processed_frames += 1
            st.session_state.frame_buffer.clear()
        
        # Release resources
        if hasattr(st.session_state, 'cap') and st.session_state.cap is not None:
            st.session_state.cap.release()
        
        if hasattr(st.session_state, 'out') and st.session_state.out is not None:
            st.session_state.out.release()
            cv2.destroyAllWindows()
        
        # Add a separator
        st.markdown("---")
        
        # Download section
        if st.session_state.processed_frames > 0:
            st.subheader("Download Processed Video")
            
            # Force flush and wait
            time.sleep(3)  # Increased wait time
            
            if (st.session_state.output_path and 
                Path(st.session_state.output_path).exists()):
                
                try:
                    with open(st.session_state.output_path, 'rb') as f:
                        video_data = f.read()
                        if len(video_data) > 1000:
                            st.success(f"Successfully processed {st.session_state.processed_frames} frames")
                            # Make download button more prominent
                            st.download_button(
                                label="📥 Download Processed Video",
                                data=video_data,
                                file_name=f"detected_video_{time.strftime('%Y%m%d_%H%M%S')}.mp4",
                                mime="video/mp4",
                                key="download_button"
                            )
                        else:
                            st.error("Error: Video file is empty or corrupted")
                            st.info("Try processing the video again with different settings")
                except Exception as e:
                    st.error(f"Error preparing download: {str(e)}")
                    st.info("Please try processing the video again")
            else:
                st.error("Output video file not found")
                st.info("Make sure to complete the video processing before downloading")

if __name__ == "__main__":
    main()