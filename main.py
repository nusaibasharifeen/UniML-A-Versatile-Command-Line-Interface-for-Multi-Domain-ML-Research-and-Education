import argparse
import subprocess
import os

parser = argparse.ArgumentParser(description="Unified CLI for all projects")
parser.add_argument('--project', type=str, required=True, help="Project name: CHATBOT, GAN, ImageCls, SPEECH")
parser.add_argument('--dataset', type=str, required=True, help="Dataset path relative to the project folder")
parser.add_argument('--model', type=str, required=True, help="Model name to use")
parser.add_argument('--mode', type=str, default='train', help="Mode: train/test/predict")
parser.add_argument('--otherOptions', type=str, default='', help="Additional options for the project")
args = parser.parse_args()

project_folder = os.path.join(os.getcwd(), args.project)
project_main = os.path.join(project_folder, 'codeP', 'main.py')

if not os.path.exists(project_main):
    raise FileNotFoundError(f"Project main.py not found in {project_main}")

cmd = [
    "python",
    project_main,
    "--dataset", args.dataset,
    "--model", args.model,
    "--mode", args.mode
]

if args.otherOptions:
    cmd.extend(args.otherOptions.split())

subprocess.run(cmd)
