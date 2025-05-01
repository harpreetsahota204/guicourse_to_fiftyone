import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


import fiftyone as fo
import numpy as np
import pandas as pd
from PIL import Image

def process_json_data(json_data: List[Dict]) -> List[Dict]:
    """
    Process raw GUI interaction data to add structured action history and metadata.
    
    Args:
        json_data (List[Dict]): List of dictionaries containing GUI interaction data
            with 'uid' and 'actions_label' keys.
            
    Returns:
        List[Dict]: Processed data with episode_id, step, current_action, and 
        structured_history fields added.
    """
    # Helper dictionary to map action types to their string representations
    action_formats = {
        'tap': lambda x: f"tap: {x['point']['related']}",
        'swipe': lambda x: f"swipe: {x['dual_point']['related']['from']} {x['dual_point']['related']['to']}",
        'input': lambda x: f"input: {x['text']}",
        'enter': lambda _: "enter",
        'answer': lambda x: f"answer: {x.get('text', '')}"
    }
    
    # Pre-process to group by episode
    episodes = {}
    for item in json_data:
        # Extract episode and step from uid
        uid_parts = item['uid'].split('_')
        episode_id = f"episode_{uid_parts[2]}"
        step = int(uid_parts[-1])
        
        # Format the current action
        action_type = item['actions_label']['name']
        current_action = action_formats.get(action_type, lambda _: "")(item['actions_label'])
        
        # Store processed info
        if episode_id not in episodes:
            episodes[episode_id] = []
        episodes[episode_id].append((step, current_action, item))

    # Process each item with its history
    processed_data = []
    for episode_id, items in episodes.items():
        # Sort items by step
        items.sort(key=lambda x: x[0])
        
        # Process each item
        for i, (step, current_action, item) in enumerate(items):
            processed_item = item.copy()
            processed_item.update({
                'episode_id': episode_id,
                'step': step,
                'current_action': current_action,
                'structured_history': [x[1] for x in items[:i]]  # All previous actions
            })
            processed_data.append(processed_item)

    return processed_data

def save_base64_image(base64_str: str, save_dir: str, image_name: str) -> str:
    """
    Convert a base64-encoded image string to an image file and save it to disk.
    
    Args:
        base64_str: Base64 encoded image string, may include data URI prefix
        save_dir: Directory path where image should be saved
        image_name: Name to give the saved image file (without extension)
        
    Returns:
        str: Full filepath of the saved image
    """
    # Create save directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Remove data URI prefix if present
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    
    # Decode base64 and convert to PIL Image
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data))
    
    # Save image as JPG
    filepath = os.path.join(save_dir, f"{image_name}.jpg")
    img.save(filepath, 'JPEG')
    
    return filepath

def parse_point_string(point_str: str) -> Tuple[float, float]:
    """
    Parse a point string in format '<point>x,y</point>' into x,y coordinates.
    
    Args:
        point_str: String containing point coordinates in XML-like format
        
    Returns:
        tuple: (x, y) coordinates as floats
        
    Raises:
        ValueError: If point string format is invalid
    """
    match = re.match(r'<point>([\d.]+),\s*([\d.]+)</point>', point_str)
    if match:
        x, y = match.groups()
        return float(x), float(y)
    raise ValueError(f"Invalid point string format: {point_str}")

def process_action_label(action: Dict) -> Tuple[Dict, Optional[fo.Keypoints]]:
    """
    Process an action label to create FiftyOne Keypoints object for visualization.
    
    Args:
        action: Dictionary containing action data with either 'point' or 'dual_point'
        
    Returns:
        tuple: (Original action dict, FiftyOne Keypoints object or None)
    """
    # Handle single point actions (e.g. clicks)
    if 'point' in action:
        x, y = parse_point_string(action['point']['related'])
        keypoint = fo.Keypoint(
            points=[[x, y]],
            label=action['name']
        )
        return action, fo.Keypoints(keypoints=[keypoint])
    
    # Handle dual point actions (e.g. swipes)
    elif 'dual_point' in action:
        from_x, from_y = parse_point_string(action['dual_point']['related']['from'])
        to_x, to_y = parse_point_string(action['dual_point']['related']['to'])
        keypoint = fo.Keypoint(
            points=[[from_x, from_y], [to_x, to_y]],
            label=action['name']
        )
        return action, fo.Keypoints(keypoints=[keypoint])
    
    return action, None

def normalize_ui_element(element: Dict, image_width: int, image_height: int) -> Dict:
    """
    Convert UI element coordinates from pixel values to normalized [0,1] range.
    
    Args:
        element: Dictionary containing UI element data with position info
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
        
    Returns:
        dict: Normalized element data with bounding box in [0,1] range
    """
    pos = element['position']
    # Convert pixel coordinates to normalized [0,1] range
    x = pos['x'] / image_width
    y = pos['y'] / image_height
    width = pos['width'] / image_width
    height = pos['height'] / image_height
    
    return {
        'bounding_box': [x, y, width, height],
        'label': element['ui_type'],
        'text': element['text']
    }

def parse_elements(elements_str):
    """
    Parse UI elements string into a list of dictionaries.
    
    Args:
        elements_str: String or list containing UI elements data
        
    Returns:
        list: List of UI element dictionaries
    """
    # Convert string representation of numpy array to Python list
    if isinstance(elements_str, str):
        elements_str = elements_str.replace("array(", "").replace(", dtype=object)", "")
        return eval(elements_str)
    return elements_str

def create_dataset(df, processed_json: List[Dict],  dataset_name: str = "guiact_smartphone_test") -> fo.Dataset:
    """
    Create a FiftyOne dataset from DataFrame and preprocessed JSON data.
    
    Args:
        df: DataFrame containing image data and UI elements
        processed_json: List of preprocessed JSON items with action data
        images_dir: Directory to save extracted images
        dataset_name: Name for the FiftyOne dataset
        
    Returns:
        fo.Dataset: FiftyOne dataset containing all samples with annotations
    """
    # Create new dataset, overwriting if it exists
    dataset = fo.Dataset(name=dataset_name, overwrite=True)
    
    # Create lookup dictionary for faster JSON data access
    json_map = {item['uid']: item for item in processed_json}
    
    samples = []
    
    for _, row in df.iterrows():
        uid = row['index']
        json_item = json_map.get(uid)
        
        if not json_item:
            continue
            
        # Extract and save image from base64, folder name is same as dataset name
        filepath = save_base64_image(row['base64'], dataset_name, uid)
        
        # Create base sample with metadata
        sample = fo.Sample(
            filepath=filepath,
            tags=[],
            episode=json_item['episode_id'],
            step=json_item['step'],
            question=json_item['question'],
            current_action=json_item['current_action'],
            structured_history=json_item['structured_history']
        )
        
        # Process UI elements
        image_size = json_item['image_size']
        ui_elements = []
        
        elements = parse_elements(row['elements'])
        
        # Convert each UI element to a Detection object
        for element in elements:
            normalized = normalize_ui_element(
                element, 
                image_size['width'], 
                image_size['height']
            )
            
            detection = fo.Detection(
                bounding_box=normalized['bounding_box'],
                label=normalized['label'],
                text=normalized['text']
            )
            ui_elements.append(detection)
        
        sample['ui_elements'] = fo.Detections(detections=ui_elements)
        
        # Add action label points for visualization if present
        if 'actions_label' in json_item:
            _, keypoints = process_action_label(json_item['actions_label'])
            if keypoints is not None:
                sample['action_label'] = keypoints
        
        samples.append(sample)
    
    dataset.add_samples(samples, dynamic=True)
    dataset.compute_metadata()
    # Create and save the sequences view
    view = dataset.group_by(
        "episode",
        order_by="step"
    )
    dataset.save_view("sequences", view)
    
    # Save the dataset
    dataset.save()

    return dataset

