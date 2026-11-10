
from src.model_training import train_all
import json

if __name__ == "__main__":
    metadata, _, _ = train_all()
    print("\nTraining complete.")
    print(json.dumps(metadata, indent=2))
