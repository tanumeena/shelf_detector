# Shelf Product Identifier

A computer vision project for detecting products in a shelf image and matching a query product against detected shelf items.

This is Tanu Meena's strongest current project for BlueDot Technical AI Safety positioning because it naturally connects applied ML with reliability, robustness, false positives, false negatives, dataset bias, and deployment risk.

## Problem Statement

Retail shelf images can contain many products with different lighting, angles, occlusion, packaging similarity, and shelf placement. This project attempts to:

1. Detect products in a shelf image.
2. Extract visual features from detected products.
3. Compare each product crop with a query product image.
4. Highlight likely matches and produce row/column-level analysis.

## Why This Matters

Shelf analytics systems can support stock estimation and planogram compliance, but they can also fail in subtle ways:

- Similar packaging can create false positives.
- Occluded products can create false negatives.
- Lighting and camera quality can shift model behavior.
- A threshold that works on one shelf may fail on another.
- Automated results may be trusted more than they deserve.

These are the same habits of thought needed in technical AI safety: define failure modes, test edge cases, document limitations, and avoid overclaiming reliability.

## Tech Stack

- Python
- OpenCV
- YOLO via Ultralytics
- ResNet18 feature extraction
- scikit-learn
- NumPy
- PIL / Pillow
- Jupyter notebooks

## Dataset / Input Format

Expected inputs:

- Query product image, for example `cola.jpg`
- Shelf image, for example `retail.jpg`
- YOLO model weights at `models/best.pt` `[NEEDS TANU CONFIRMATION - weights are referenced by code but not visible in this repo audit]`

The project contains sample image assets including:

- `cola.jpg`
- `retail.jpg`
- `image.png`
- `result.jpg`

`sample_image.jpeg` exists but is empty in the current clone and should be replaced or removed.

## Method / Pipeline

1. Load the query image and shelf image.
2. Use YOLO to detect product bounding boxes in the shelf image.
3. Crop detected products.
4. Use ResNet18-based embeddings to represent product crops and the query image.
5. Compute cosine similarity between query and shelf product embeddings.
6. Apply a similarity threshold to identify likely matches.
7. Assign row/column positions to detections.
8. Save predictions, detailed analysis, summary statistics, and a result image.

## Verified Sample Output

The repository includes `summary.txt` and `detailed_analysis.txt` from a saved run.

Verified sample run:

- Query image: `input/cola.jpg`
- Shelf image: `input/retail.jpg`
- Threshold: `0.85`
- Total products detected: `85`
- Total matches: `6`
- Overall match rate: `7.1%`

These numbers describe the saved sample analysis only. They should not be presented as general model accuracy.

## Model Evaluation

Current evaluation is based on detection output, similarity scores, thresholding, and manual/visual review.

Recommended next evaluation work:

- Create a labeled test set with ground-truth product matches.
- Track precision, recall, and F1-score for match detection.
- Compare thresholds such as 0.75, 0.80, 0.85, and 0.90.
- Separate detection errors from similarity-matching errors.
- Test under lighting variation, occlusion, blur, and similar packaging.

General accuracy score: `[NEEDS TANU CONFIRMATION]`

## Reliability and Safety Learnings

- False positives: Similar packaging or visual features can mark a wrong product as a match.
- False negatives: Occluded, blurry, or partially visible products may be missed.
- Lighting variation: Shadows and reflections can change embeddings and detection confidence.
- Occlusion: Products behind other products may not be detected reliably.
- Dataset bias: If training data has limited brands, angles, or stores, real-world performance may drop.
- Deployment risk: In a business setting, wrong detections could distort stock estimates or compliance reports.
- Threshold sensitivity: A single similarity threshold can be brittle across different shelf layouts.

## Failure Modes

- Missing or incompatible YOLO weights.
- Empty or invalid input images.
- Poor shelf image quality.
- Incorrect row/column assignment when shelves are irregular.
- Crops that include multiple products.
- High similarity between different products with similar packaging.

## Limitations

- The repository references `models/best.pt`, but model weight availability must be confirmed.
- Current results are from a sample run, not a complete benchmark.
- No formal precision/recall table is included yet.
- No data statement or model card is included yet.
- The project should not be used for automated business decisions without validation.

## Future Improvements

- Add labeled evaluation data.
- Add threshold comparison tables.
- Add a model card.
- Add failure-case images.
- Add tests for missing files and invalid inputs.
- Separate detection, embedding, and reporting code into smaller modules.
- Remove empty `sample_image.jpeg` or replace it with a valid sample.

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the main script:

```bash
python main.py --query cola.jpg --shelf retail.jpg --threshold 0.85
```

If the script expects input files under an `input/` folder or model weights under `models/`, create those folders and place the verified files there.

## Screenshots / Demo

Real image files present in this repo:

- `result.jpg`
- `retail.jpg`
- `image.png`

Use these only after checking that they render correctly on GitHub.

## BlueDot Positioning

This project is a practical example of reliability engineering for computer vision. The strongest application angle is not "perfect detection"; it is the ability to reason honestly about model failures, thresholds, evaluation, and deployment risk.

