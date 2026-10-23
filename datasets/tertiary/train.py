from pathlib import Path
 
import torch
from ultralytics import YOLO
 
DATASET_DIR = Path(__file__).resolve().parent
DATA_YAML = str(DATASET_DIR / "data.yaml")
RUNS_DIR = str(DATASET_DIR / "runs")
 
MODEL = "yolo26s.pt"
IMGSZ = 640
EPOCHS = 1000
 
 
def pick_device():
    if torch.cuda.is_available():
        print("CUDA available!")
        return 0
    if torch.backends.mps.is_available():
        print("Apple Silicon available!")
        return "mps"
    print("No hardware acceleration found, falling back to cpu")
    return "cpu"
 
 
def main():
    device = pick_device()
 
    model = YOLO(MODEL)
    model.train(
        data=DATA_YAML,
        imgsz=IMGSZ,
        epochs=EPOCHS,
        patience=20,
        device=device,
        project=RUNS_DIR,
        name="umpire",
    )
 
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"\nBest weights: {best}")
 
    m = YOLO(best).val(
        data=DATA_YAML,
        split="test",
        imgsz=IMGSZ,
        device=device,
        project=RUNS_DIR,
        name="umpire_test",
    )
    print(f"precision {m.box.mp:.3f}  recall {m.box.mr:.3f}  "
          f"mAP50 {m.box.map50:.3f}  mAP50-95 {m.box.map:.3f}")
 
 
if __name__ == "__main__":
    main()
