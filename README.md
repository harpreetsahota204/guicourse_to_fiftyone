# Parsing the GUIAct datasets from GUICourse to FiftyOne format

<img src="guiact_websingle.gif">

⬆️ Sample of the GUIAct Web-Single Dataset

GUIAct is a multi-scenario dataset designed to enhance GUI knowledge in vision language models. It contains GUI navigation tasks across website and smartphone environments with accurately annotated action sequences.

- **Curated by:** Wentong Chen, Junbo Cui, Jinyi Hu and other authors from Tsinghua University, Renmin University of China, and other institutions
- **Funded by:** Not explicitly mentioned in the paper
- **Shared by:** The authors as part of the GUICourse dataset suite
- **License:** CC BY 4.0

# You can download these parsed datasets from the Hugging Face Hub:

## Installation


  If you haven''t already, install FiftyOne:


  ```bash

  pip install -U fiftyone

  ```


  ## Usage


  ```python

  import fiftyone as fo

  from fiftyone.utils.huggingface import load_from_hub

  # load the smartphone dataset
  smartphone_dataset = load_from_hub("Voxel51/guiact_smartphone_test")

# load the web-single dataset
  websinge_dataset = load_from_hub("Voxel51/guiact_websingle_test")

# load the web-multi dataset
  websinge_dataset = load_from_hub("Voxel51/guiact_webmulti_test")

  # Launch the App

  session = fo.launch_app(smartphone_dataset)

  ```


### Dataset Sources

- **Repository:** https://github.com/yiye3/GUICourse
- **Paper:** "GUICourse: From General Vision Language Model to Versatile GUI Agent" (arXiv:2406.11317v1)

## Uses

### Direct Use

GUIAct is intended for:
1. Training GUI agents to perform navigation tasks in website and smartphone environments
2. Enhancing vision language models' knowledge of GUI components and interaction methods
3. Evaluating models on their ability to execute single-step and multi-step GUI tasks
4. Improving position-dependent action execution in visual GUI environments

### Out-of-Scope Use

The dataset is not designed for:
1. Training agents to execute potentially harmful GUI operations
2. Creating autonomous systems without human oversight
3. Training on non-GUI interfaces or text-based interfaces

## Dataset Structure

GUIAct consists of three primary partitions:

1. **GUIAct (web-single)**: 67k single-step action instructions with their corresponding website screenshots
2. **GUIAct (web-multi)**: 5,696 multi-step action instructions (approximately 44k training samples)
3. **GUIAct (smartphone)**: 9,157 multi-step action instructions adapted from AITW (approximately 67k training samples)

Each sample contains:
- A screenshot of a GUI environment
- Task instructions
- Corresponding actions with parameters (positions and text)

The unified action space includes 11 types of actions across 7 categories, including pointing actions (click, tap), inputting actions (input), browsing actions (scroll, swipe), and others.

## FiftyOne Dataset Structure

The following is an example sample structure from the Web-Single dataset:

**Core Fields:**
- `id`: ObjectIdField - Unique identifier
- `filepath`: StringField - Image path
- `tags`: ListField(StringField) - Sample categories
- `metadata`: EmbeddedDocumentField - Image properties (size, dimensions)
- `image_id`: StringField - Unique identifier for the screenshot
- `uid`: StringField - Unique identifier for the task instance
- `question`: StringField - Natural language task description
- `ui_elements`: EmbeddedDocumentField(Detections) containing multiple Detection objects:
  - `label`: Sequential numeric ID for element (e.g., "1", "2")
  - `bounding_box`: Coordinates as [x, y, width, height] in normalized format (0-1)
  - `text`: Text content of element if present
- `action_detections`: EmbeddedDocumentField(Detections) containing target interaction elements:
  - `label`: Action type (e.g., "click") 
  - `bounding_box`: Element coordinates as [x, y, width, height] in normalized format
  - `order`: Sequential order of action
- `structured_history`: ListField(StringField) - Previous actions in structured text format
- `action_keypoints`: EmbeddedDocumentField(Keypoints) - Point-based interaction coordinates (if used)
- `action_classifications`: EmbeddedDocumentField(Classifications) - Action classification information (if used)


## Dataset Creation

### Curation Rationale

GUIAct was created to address three primary limitations in existing GUI datasets:
1. Many existing environments are too simple and don't represent real-world scenarios
2. Existing datasets focus on narrow domains or scenarios
3. Many existing datasets are too small to effectively train GUI agents

### Source Data

#### Data Collection and Processing

**For GUIAct (web-single)**:
1. Website selection: Used GPT-4 to identify diverse scenarios, resulting in 50 domains and 13k websites
2. Screenshot capture: Used web snapshot tools to capture website HTML, interactive elements, and screenshots
3. LLM-Auto Annotation: Used GPT-4V with two images (original and element-identified screenshots) to generate instruction-action pairs
4. Human verification: Annotators reviewed and corrected data, improving accuracy from 55% to 92%

**For GUIAct (web-multi)**:
1. Selected 8 top-level web scenarios across 121 websites
2. Generated 8k high-level instructions using GPT-3.5 and Claude2
3. Used a custom browser plugin for human annotators to execute and verify tasks

**For GUIAct (smartphone)**:
1. Selected a subset of AITW dataset with the "General" tag
2. Filtered screenshots without bottom navigation bars
3. Converted original actions to their unified action space

#### Who are the source data producers?

- Web screenshots from diverse websites across 50 domains
- Smartphone screenshots from the AITW dataset
- Instructions generated by LLMs and refined by human annotators

### Annotations

#### Annotation process

The paper described detailed annotation processes:
- For web-single: GPT-4V generated initial annotations, followed by human verification
- For web-multi: Custom browser plugin allowed annotators to record their interactions
- For smartphone data: Conversion rules applied to adapt AITW annotations

#### Personal and Sensitive Information

The paper doesn't explicitly address personal information in GUIAct, though they mention in the ethical considerations that there's no personally identifiable information in their datasets.

## Bias, Risks, and Limitations

The dataset may contain biases from:
- Website selection process (driven by GPT-4 suggestions)
- Types of tasks represented (focused on common navigation patterns)
- Complexity levels of included tasks
- Cultural specificity of websites chosen

Technical limitations include:
- Limited to website and smartphone scenarios
- Possible inconsistencies in human annotations
- Simplified action space that doesn't cover all possible GUI interactions

### Recommendations

Users should:
- Evaluate model performance across diverse websites not included in the training data
- Be aware of potential biases in the types of tasks represented
- Consider supplementing with additional data for specialized GUI environments
- Implement appropriate safeguards when deploying GUI agents trained on this data
- Ensure human oversight for autonomous GUI interactions

## More Information

The data is part of a larger GUICourse suite that includes GUIEnv (for OCR and grounding abilities) and GUIChat (for interaction abilities). The paper demonstrates that the GUIAct dataset can significantly improve GUI navigation performance of vision language models.

# Citation
```bibtex
@misc{,
  title={GUICourse: From General Vision Language Models to Versatile GUI Agents},
  author={Wentong Chen and Junbo Cui and Jinyi Hu and Yujia Qin and Junjie Fang and Yue Zhao and Chongyi Wang and Jun Liu and Guirong Chen and Yupeng Huo and Yuan Yao and Yankai Lin and Zhiyuan Liu and Maosong Sun},
  year={2024},
  journal={arXiv preprint arXiv:2406.11317},
}
```