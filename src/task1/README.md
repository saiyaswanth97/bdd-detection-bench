# Task 1: Class Distribution Analysis

Analysis of class distribution in the BDD100K dataset, including occurrence patterns, train-val ratios, and class co-occurrence matrices.

## Key Insights

- Most objects are cars (600K in 100K images). The they are followed by traffic signs & lights
- Cars also appear in most images. 
  - 12% images contain just cars.
  - 85% images contain ontl (cars + traffic light + traffic sign) 
  - Each image contains ~15 objects. But a lot of image contain >20 objects.
- Train class object is very rare - 42 occurances in 100K image
- Split seems good based on multiple metrics
  - RoI; Frequnecy distribution
- All classe frequnecy distribution seems to poission distribution
- Causal classes
  - (car; traffic light; traffic sign); (bike; rider); (rider; motor)
- Bbox Area distribution
  - Vehices seems to in range of [0, 0.1-0.2]
  - Rider and bikes are in [0, 0.05]
  - Traffic stuff are in [0, 0.05]. Traffic light is even worse where 90% are below 0.1% (32^2)

## 🔧 Requirements

```bash
# Required packages
pip install tqdm matplotlib numpy
```

## 📋 Dataset Structure Expected

```
src/
└── task1/
    ├── class_distribution.py
    └── images/          # Output folder for generated plots
data/
└── bdd100k_labels/
    └── 100k/
        ├── train/          # 70,000 JSON files
        │   └── *.json
        └── val/            # 10,000 JSON files
            └── *.json
```

## 🎯 Class Categories

The script analyzes these 10 BDD100K categories (in-order):

1. **car**
2. **traffic sign**
3. **traffic light**
4. **person**
5. **truck**
6. **bus**
7. **bike**
8. **rider**
9. **motor**
10. **train**

---


## 🚀 Quick Start

### Run Class Distribution Analysis

```bash
# Run the main distribution analysis
python3 -m src.task1.class_distribution
```

This will:
- Process 70,000 training images & 10,000 validation images
- Generate 6 visualization plots & save them to `src/task1/images/`

### Run Anomaly Detection & Sample Extraction

```bash
# Run anomaly detection and extract sample images
python3 -m src.task1.anamoly_detection
```

This will:
- Compute class bitmasks for presence-based queries
- Generate class presence distribution plots
- Extract sample images with specific class patterns (e.g., images with 40 cars, 30 signs, etc.)
- Create frequency distribution histograms for all classes

## 📊 Generated Visualizations

### 1. Class Cumulative Presence

Shows how many images contain each class at least once. This represents the "presence" of each class across the dataset.

![Class Cumulative Presence](images/class_cumulative_presence.png)

---

### 2. Class Cumulative Instances

Shows the total count of all object instances across the dataset. Multiple instances per image are counted separately.

![Class Cumulative Instances](images/class_cumulative_frequency.png)

---

### 3. Average Occurrences Per Image

Shows the average number of times each class appears in images where it's present.

![Average Occurrences Per Image](images/class_occurrence_per_image.png)

---

### 4. Train/Test Ratio

Compares the total number of object instances between train and validation sets. Ignore this. From the plots looks like the split is done based on "presence" of classes rather than total instances.

![Train Validation Ratio Frequency](images/class_occurrence_ratio_all.png)

---

### 5. Train/Validation Ratio (Presence)
Compares how many images contain each class between train and test sets.

![Train Validation Ratio](images/class_occurrence_ratio_per_image.png)


---

### 6. Class Co-occurrence Matrix

Shows which classes tend to appear together in the same images. Uses log scale for better visualization.

![Class Co-occurrence Matrix](images/class_cooccurrence.png)

---

### 7. Bboxes distribution

Here we have how the boxes area & aspect ratio are spread across each class

![Bbox area distribution](images/bbox_area_distribution_train.png)

| Small | Medium | Large |
|-------|--------|-------|
| ![Small](images/bbox_area_distribution_small.png) | ![Medium](images/bbox_area_distribution_medium.png) | ![Large](images/bbox_area_distribution_large.png) |

![Bbox aspect ratio distribution](images/bbox_aspect_distribution_train.png)

Closer look for the traffic sign - It's distribution is heavily skewed towards small area (0.001)

![Traffic Light Bbox area distribution](images/bbox_area_distribution_traffic_light.png)

We also are plotting where each object will apear on the image. The split across the train and validation seems good.

![Bbox RoI distribution](images/bbox_region_distribution_train.png)
![Val Bbox RoI distribution](images/bbox_region_distribution_validation.png)

---

## 🔍 Anomaly Detection & Sample Extraction

The `anamoly_detection.py` script provides advanced querying and sample extraction capabilities.

### Generated Analysis

#### All Class Frequency Distribution

Subplots showing instance count distributions for each of the 10 classes:

![All Class Frequency Distribution](images/all_class_frequency_distribution.png)

#### Class Presence Distributions

Multiple histogram plots showing the distribution of class presence patterns:

##### Full Distribution Across All Classes

![Class Presence Distribution](images/class_presence_distribution.png)

##### Distribution for Cars, Signs, and Lights Only

![Class Presence Distribution Cars Signs Lights](images/class_presence_distribution_cars_signs_lights.png)
`[none, only car, only sign. car&sign, only light, car&light, sign&light, all]`

##### Distribution Excluding Cars, Signs, and Lights

![Class Presence Distribution No Cars Signs Lights](images/class_presence_distribution_no_cars_signs_lights.png)

#### Sample Images

Extracted sample images demonstrating specific patterns:

##### Sample: Image with Train

![Sample Train](images/sample_train.jpg)

##### Sample: No Cars, Signs, or Lights

![Sample No Car Sign Light](images/sample_no_car_sign_light.jpg)

##### Image with 40 Cars

![Image 40 Cars](images/image_40_cars.jpg)

##### Image with 30 Traffic Signs

![Image 30 Signs](images/image_30_signs.jpg)

##### Image with 40 Persons

![Image 40 Persons](images/image_40_persons.jpg)

##### Image with 15 Bikes

![Image 15 Bikes](images/image_15_bikes.jpg)

##### Image with 10 Motorcycles

![Image 10 Motorcycles](images/image_10_motorcycles.jpg)

##### Image with 4 Trains

![Image 4 Trains](images/image_4_trains.jpg)

### Query Examples

```python
# Presence-based query: Find images with trains present
query_bitmask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # Only train (10th class)
sample_ids = query_presence_sample_id(image_names, class_bitmask, query_bitmask)

# Presence-based query: Find images WITHOUT cars, signs, or lights
query_bitmask = [-1, -1, -1, 0, 0, 0, 0, 0, 0, 0]  # No car, sign, light
sample_ids = query_presence_sample_id(image_names, class_bitmask, query_bitmask)

# Frequency-based query: Find images with exactly 40 cars
sample_ids = query_frequency_sample_id(image_names, instance_counts, query=(0, 40))
```

---

### Action items:
- [] Add occulsion and truncation to the mix
- [] Add Anomoly detection using the bounding boxes
- [] Redo the readme file with more explanation
- [] Containerization
- [] Add `uv`
- [] Explore & add daskboard

---