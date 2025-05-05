import fiftyone as fo
import base64
import os
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional
from PIL import Image
import io


def save_base64_image(base64_str: str, save_dir: str, image_name: str) -> Tuple[str, Dict[str, int]]:
    """
    Convert and save a base64-encoded image string to a JPEG file on disk.
    
    Args:
        base64_str: Base64-encoded string representation of the image
        save_dir: Directory path where the image should be saved
        image_name: Name to use for the saved image file (without extension)
        
    Returns:
        Tuple[str, Dict[str, int]]: Filepath where image was saved and dictionary with image dimensions
    """
    # Create directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Remove data URL prefix if present
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    
    # Decode base64 to image and save as JPEG
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data))
    filepath = os.path.join(save_dir, f"{image_name}.jpg")
    img.save(filepath, 'JPEG')
    
    # Get actual image dimensions
    image_size = {'width': img.width, 'height': img.height}
    
    return filepath, image_size

    
def normalize_ui_element(element: Dict, image_width: int, image_height: int) -> Dict:
    """
    Normalize UI element coordinates from pixel values to [0,1] range and format for FiftyOne.
    
    Args:
        element: Dictionary containing UI element data with rect coordinates in pixels
        image_width: Width of the original image in pixels 
        image_height: Height of the original image in pixels

    Returns:
        Dict: Normalized element with bounding box in FiftyOne format
    """
    rect = element['rect']
    
    # Convert pixel coordinates to normalized [0,1] range
    x = rect['x'] / image_width
    y = rect['y'] / image_height
    width = rect['width'] / image_width
    height = rect['height'] / image_height
    
    return {
        'bounding_box': [x, y, width, height],
        'label': str(element.get('uid', '')),
        'text': element.get('text', ''),
    }

def parse_normalized_box_string(box_str: str) -> List[float]:
    """
    Parse box string in format '<box>x1,y1,x2,y2</box>' to FiftyOne format [x,y,width,height].
    
    Args:
        box_str: String containing normalized coordinates in <box> tags
        
    Returns:
        List[float]: Coordinates in [x, y, width, height] format
        
    Raises:
        ValueError: If box string format is invalid
    """
    match = re.match(r'<box>([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)</box>', box_str)
    if match:
        x1, y1, x2, y2 = map(float, match.groups())
        return [x1, y1, x2-x1, y2-y1]  # Convert to width/height format
    raise ValueError(f"Invalid box string format: {box_str}")

def parse_elements(elements_str):
    """
    Parse UI elements string from DataFrame into Python list.
    
    Args:
        elements_str: Raw elements string or list from DataFrame
        
    Returns:
        list: Parsed list of UI element dictionaries
    """
    if isinstance(elements_str, str):
        # Clean numpy array string formatting
        elements_str = elements_str.replace("array(", "").replace(", dtype=object)", "")
        return eval(elements_str)
    return elements_str

def process_ui_elements(elements_str, image_size: Dict) -> fo.Detections:
    """
    Convert UI elements into FiftyOne Detections format.
    
    Args:
        elements_str: Raw elements string/list from DataFrame
        image_size: Dictionary containing image width and height
        
    Returns:
        fo.Detections: UI elements formatted as FiftyOne detections
    """
    elements = parse_elements(elements_str)
    ui_elements = []
    
    for element in elements:
        normalized = normalize_ui_element(
            element, 
            image_size['width'], 
            image_size['height']
        )
        
        detection = fo.Detection(
            bounding_box=normalized['bounding_box'],
            label=normalized['label'],
            text=normalized['text'],
        )
        ui_elements.append(detection)
    
    return fo.Detections(detections=ui_elements)

def process_action_labels(action_labels):
    """
    Process a list of action labels into appropriate FiftyOne objects based on action type.
    
    Args:
        action_labels: List of action dictionaries, each containing a 'name' key
        
    Returns:
        dict: Dictionary with possible keys 'detections', 'keypoints', and 'classifications'
        containing the respective FiftyOne objects
    """
    # Initialize lists for different types of actions
    detections = []  # for click, input, select
    keypoints = []   # for scroll
    classifications = []  # for answer, enter
    
    for idx, action in enumerate(action_labels):
        # Safely get the action name
        name = action.get('name')
        
        # Handle actions that need bounding boxes (click, input, select)
        if name in ['click', 'input', 'select']:
            detection = fo.Detection(
                label=name,
                order=idx,
                bounding_box=parse_normalized_box_string(action.get('element', {}).get('related')),
                text=action.get('text')  # Will be None if not present
            )
            detections.append(detection)
            
        # Handle scroll actions
        elif name == 'scroll':
            scroll_data = action.get('scroll', {})
            y = float(scroll_data.get('related', {}).get('down', 0))
            
            keypoint = fo.Keypoint(
                label='scroll',
                order=idx,
                points=[[1.0, abs(y)]],  # Place at right edge of image
                direction= 'up' if y < 0 else 'down',
            )
            keypoints.append(keypoint)
            
        # Handle classification actions (answer)
        elif name in ['answer']:
            classification = fo.Classification(
                label=action.get('text')
            )
            classifications.append(classification)
    
    # Create return dictionary, only including non-empty lists
    result = {}
    if detections:
        result['detections'] = fo.Detections(detections=detections)
    if keypoints:
        result['keypoints'] = fo.Keypoints(keypoints=keypoints)
    if classifications:
        result['classifications'] = fo.Classifications(classifications=classifications)
        
    return result

def create_structured_history(actions_label):
    """
    Convert actions_label list into a structured_history list of strings.
    
    Args:
        actions_label: List of action dictionaries
        
    Returns:
        list: Structured history as list of strings
    """
    history = []
    
    for action in actions_label:
        name = action.get('name')
        
        if name == 'scroll':
            y = float(action.get('scroll', {}).get('related', {}).get('down', 0))
            direction = 'down' if y > 0 else 'up'
            history.append(f'scroll: {direction}')
            
        elif name == 'answer':
            text = action.get('text', '')
            history.append(f'answer: {text}')
            
        elif name in ['click', 'select']:
            box = action.get('element', {}).get('related', '')
            history.append(f'{name}: {box}')
            
        elif name == 'input':
            text = action.get('text', '')
            history.append(f'input: {text}')
            
        elif name == 'enter':
            history.append('enter')
            
    return history

def create_dataset(df, json_data: List[Dict], dataset_name: str = "guiact_websingle_test") -> fo.Dataset:
    """
    Create a FiftyOne dataset from DataFrame and JSON data containing UI interactions.
    Each image from the DataFrame may have multiple associated JSON entries (identified by unique UIDs).
    The function creates separate samples for each unique UID, saving duplicate images when necessary.
    
    Args:
        df: DataFrame containing image data (base64) and UI elements
        json_data: List of dictionaries containing metadata, questions, and action labels
        dataset_name: Name for the FiftyOne dataset
        
    Returns:
        fo.Dataset: Created FiftyOne dataset containing all samples, with images saved using UIDs
    """
    dataset = fo.Dataset(name=dataset_name, overwrite=True)
    
    # Create mapping of image_id to list of JSON items since one image may have multiple questions/actions
    json_map = {}
    for item in json_data:
        image_id = item['image_id']
        if image_id not in json_map:
            json_map[image_id] = []
        json_map[image_id].append(item)
    
    samples = []
    for _, row in df.iterrows():
        image_id = row['index']
        json_items = json_map.get(image_id, [])
        
        # Skip if no JSON entries found for this image
        if not json_items:
            continue
        
        # Create a separate sample for each JSON entry (question/action) associated with this image
        for json_item in json_items:
            # Use UID as filename to ensure unique identification of each question/action instance
            uid = json_item['uid']
            filepath, image_size = save_base64_image(row['base64'], dataset_name, uid)
            
            # Create sample with basic metadata
            sample = fo.Sample(
                filepath=filepath,
                image_id=json_item['image_id'],
                uid=uid,
                question=json_item['question']
            )
            
            # Add UI element detections from the DataFrame
            sample['ui_elements'] = process_ui_elements(row['elements'], image_size)
            
            # Process and add action labels if present in JSON
            if 'actions_label' in json_item:
                action_objects = process_action_labels(json_item['actions_label'])
                for field_name, objects in action_objects.items():
                    sample[f'action_{field_name}'] = objects
                sample['structured_history'] = create_structured_history(json_item['actions_label'])
            
            samples.append(sample)
    
    dataset.add_samples(samples, dynamic=True)
    dataset.compute_metadata()
    return dataset