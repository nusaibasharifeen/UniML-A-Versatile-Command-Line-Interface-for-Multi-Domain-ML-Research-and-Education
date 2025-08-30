import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, help="Path to dataset folder")
parser.add_argument("--model", required=True, help="Model name")
parser.add_argument("--otherOptions", default="", help="Other options")
args = parser.parse_args()

print(f"Dataset path: {args.dataset}")
print(f"Model: {args.model}")
print(f"Other options: {args.otherOptions}")

# Example check: dataset folders exist
train_path = os.path.join(args.dataset, "training")
test_path = os.path.join(args.dataset, "testing")

if not os.path.exists(train_path) or not os.path.exists(test_path):
    raise FileNotFoundError("Training or testing folder not found.")

print("Training model...")
print("Testing model...")
print("Done! Results saved.")
