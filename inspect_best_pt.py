from ultralytics import YOLO

# Load the YOLO model
model = YOLO('models/best.pt')

with open('best_pt_info.txt', 'w', encoding='utf-8') as f:
    f.write('Model architecture:\n')
    f.write(str(model.model) + '\n\n')

    # Print class names (if available)
    if hasattr(model, 'names'):
        f.write('Class names:\n')
        f.write(str(model.names) + '\n\n')

    # Print model info (if available)
    if hasattr(model, 'info'):
        f.write('Model info:\n')
        f.write(str(model.info()) + '\n') 