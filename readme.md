# Shelf Product Identifier

A comprehensive computer vision system for identifying and analyzing products in retail shelf images using deep learning and object detection.

## 🎯 Overview

This system combines **YOLO object detection** and **ResNet18 feature extraction** to:

- Detect individual products in shelf images
- Compare them with a query product using deep learning features
- Highlight matching products with grid coordinates
- Generate detailed analysis reports

## ✨ Key Features

### 🔍 **Intelligent Product Detection**

- **YOLO Object Detection**: Automatically detects products in shelf images
- **Confidence-based Filtering**: Only processes high-confidence detections
- **Robust Alignment**: Ensures perfect correspondence between detections and analysis

### 🧠 **Deep Learning Similarity Matching**

- **ResNet18 Feature Extraction**: 512-dimensional feature vectors for each product
- **Cosine Similarity**: Measures visual similarity between query and shelf products
- **Configurable Threshold**: Adjustable similarity threshold (0.0-1.0)

### 📊 **Grid Organization & Analysis**

- **Automatic Row/Column Assignment**: Organizes products into logical grid
- **Row-by-Row Statistics**: Detailed analysis of each shelf row
- **Match Percentage Calculation**: Per-row and overall match rates

### 🎨 **Visual Results**

- **Highlighted Matches**: Green boxes with similarity scores
- **Grid Coordinates**: R1C1, R2C5, etc. for easy product identification
- **Comprehensive Legend**: Clear visual explanation of results

### 📈 **Detailed Reporting**

- **CSV Predictions**: Machine-readable results
- **Detailed Analysis**: Row-by-row breakdown
- **Summary Statistics**: Overall performance metrics

## 🚀 Quick Start

### Prerequisites

```bash
# Install required packages
pip install ultralytics opencv-python pillow scikit-learn numpy
```

### Basic Usage

```bash
python main.py --query input/cola.jpg --shelf input/retail.jpg --threshold 0.85
```

### Parameters

- `--query`: Path to query product image
- `--shelf`: Path to shelf image to analyze
- `--threshold`: Similarity threshold (default: 0.80)

## 📁 Project Structure

```
shelf-product-identifier-main/
├── main.py                    # Main analysis script
├── models/
│   └── best.pt               # Trained YOLO model
├── src/
│   └── img2vec_resnet18.py   # ResNet18 feature extractor
├── input/
│   ├── cola.jpg              # Query product image
│   └── retail.jpg            # Shelf image
├── output/
│   └── run_YYYYMMDD_HHMMSS/  # Analysis results
└── README.md                 # This file
```

## 🔧 Technical Architecture

### 1. **Object Detection Pipeline**

```python
# YOLO Detection
model = YOLO('models/best.pt')
results = model.predict(
    source=shelf_img_path,
    save_crop=True,
    save_txt=True,
    conf=0.5
)
```

### 2. **Feature Extraction Pipeline**

```python
# ResNet18 Feature Extraction
img2vec = Img2VecResnet18()
query_vec = img2vec.getVec(query_image)
crop_vecs = [img2vec.getVec(crop) for crop in crops]
```

### 3. **Similarity Analysis**

```python
# Cosine Similarity Calculation
sims = cosine_similarity([query_vec_norm], crop_vecs_norm)[0]
matches = [i for i, s in enumerate(sims) if s >= threshold]
```

### 4. **Grid Organization**

```python
# Row/Column Assignment
# Sort by Y-coordinate → Group by vertical proximity → Sort by X-coordinate
row_assignments = assign_rows(bboxes)
col_assignments = assign_columns(bboxes)
```

## 📊 Output Files

### 1. **result.jpg**

Visual result image with:

- **Green boxes**: Matching products with similarity scores
- **Red boxes**: Non-matching products
- **Yellow labels**: Grid coordinates (R1C1, R2C5, etc.)
- **Legend**: Color coding explanation
- **Summary**: Match statistics

### 2. **predictions.txt**

CSV format with columns:

```
identifier,match,row,column,similarity
R1C1,NO,1,1,0.234
R2C5,YES,2,5,0.951
```

### 3. **detailed_analysis.txt**

Row-by-row breakdown:

```
Row 1: (22 products, 0 matches, 0.0% match rate)
    R1C1, 0.234, NO
    R1C2, 0.156, NO
    ...

Row 2: (19 products, 6 matches, 31.6% match rate)
    R2C1, 0.445, NO
    R2C5, 0.951, YES
    ...
```

### 4. **summary.txt**

Overall statistics:

```
Shelf Analysis Summary
====================
Query Image: input/cola.jpg
Shelf Image: input/retail.jpg
Threshold: 0.85
Results:
- Total products detected: 85
- Total matches: 6
- Overall match rate: 7.1%

Row-by-Row Statistics:
====================
Row 1:
  - Total products: 22
  - Matches: 0
  - Other products: 22
  - Match percentage: 0.0%
```

## 🎛️ Configuration Options

### Detection Parameters

```python
# YOLO Configuration
conf_threshold = 0.5        # Detection confidence
save_crops = True           # Save individual crops
save_labels = True          # Save bounding box coordinates
```

### Similarity Parameters

```python
# Feature Extraction
feature_dim = 512           # ResNet18 output dimension
normalization_eps = 1e-8    # Numerical stability

# Similarity Calculation
similarity_threshold = 0.85  # Match threshold
```

### Visualization Parameters

```python
# Color Scheme
match_color = (0, 255, 0)      # Green for matches
no_match_color = (0, 0, 255)   # Red for non-matches
label_color = (255, 255, 0)    # Yellow for grid labels

# Box Properties
match_thickness = 3            # Thick boxes for matches
no_match_thickness = 1         # Thin boxes for non-matches
```

## 🔍 Understanding the Results

### Similarity Scores

- **1.000**: Perfect match (identical products)
- **0.900-0.999**: Very similar products
- **0.800-0.899**: Similar products
- **0.700-0.799**: Somewhat similar
- **< 0.700**: Different products

### Grid Coordinates

- **R1C1**: Row 1, Column 1 (top-left)
- **R2C5**: Row 2, Column 5
- **R3C10**: Row 3, Column 10

### Match Statistics

- **Total Products**: All detected products
- **Matches**: Products above similarity threshold
- **Match Rate**: Percentage of products that match
- **Row Analysis**: Per-row breakdown of matches

## 🛠️ Advanced Usage

### Custom Threshold

```bash
# Lower threshold for more matches
python main.py --query input/cola.jpg --shelf input/retail.jpg --threshold 0.70

# Higher threshold for exact matches only
python main.py --query input/cola.jpg --shelf input/retail.jpg --threshold 0.95
```

### Batch Processing

```bash
# Process multiple shelf images
for shelf in input/shelves/*.jpg; do
    python main.py --query input/cola.jpg --shelf "$shelf" --threshold 0.85
done
```

## 🔧 Troubleshooting

### Common Issues

1. **No detections found**

   - Check YOLO model file exists: `models/best.pt`
   - Verify image quality and size
   - Adjust confidence threshold

2. **Low similarity scores**

   - Ensure query image is clear and well-lit
   - Try different similarity thresholds
   - Check for image format issues

3. **Incorrect row/column assignment**
   - Verify shelf image orientation
   - Check for overlapping products
   - Review detection confidence scores

### Debug Information

The system provides extensive debug output:

```
Top 10 similarities:
  1. retail11.jpg: 0.951 (bbox:(931, 142, 997, 337))
  2. retail.jpg: 0.949 (bbox:(1134, 147, 1202, 339))
  ...

Debug - First 5 crops and bboxes:
  Crop 1: retail.jpg -> bbox:(1134, 147, 1202, 339)
  Crop 2: retail10.jpg -> bbox:(1067, 146, 1131, 338)
  ...
```

## 📈 Performance Optimization

### Memory Usage

- **Feature Vectors**: ~85 × 512 × 4 bytes per analysis
- **Image Processing**: ~10-50MB per image
- **Model Loading**: ~500MB (YOLO + ResNet18)

### Processing Time

- **YOLO Detection**: 2-5 seconds
- **Feature Extraction**: 1-3 seconds per image
- **Similarity Calculation**: 0.1 seconds
- **Visualization**: 0.5 seconds

### Scalability

- **Small Shelves**: < 50 products (real-time)
- **Medium Shelves**: 50-200 products (fast)
- **Large Shelves**: 200+ products (moderate)

## 🤝 Contributing

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd shelf-product-identifier-main

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/
```

### Code Structure

- **main.py**: Main analysis pipeline
- **src/img2vec_resnet18.py**: Feature extraction
- **models/**: Trained YOLO models
- **input/**: Test images
- **output/**: Analysis results

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **YOLO**: Real-time object detection
- **ResNet18**: Deep learning feature extraction
- **OpenCV**: Image processing and visualization
- **scikit-learn**: Machine learning utilities

## 📞 Support

For questions, issues, or contributions:

1. Check the troubleshooting section
2. Review debug output
3. Open an issue with detailed information
4. Include sample images and error messages

---

**Happy Product Identification! 🎉**
