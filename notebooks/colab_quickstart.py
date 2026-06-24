# Hypospadias Highlight Reel — Google Colab Notebook
#
# Upload this as a .ipynb or copy cells into Colab.
# Video should be in your Google Drive.

# ============================================================
# CELL 1: Install dependencies
# ============================================================
# !pip install opencv-python-headless numpy pyyaml anthropic torch torchvision

# ============================================================
# CELL 2: Mount Google Drive
# ============================================================
# from google.colab import drive
# drive.mount('/content/drive')

# ============================================================
# CELL 3: Clone the repo
# ============================================================
# !git clone -b claude/cool-ptolemy-nwoloj https://github.com/cxia0024/hypospadias-highlight-reel-1.git
# %cd hypospadias-highlight-reel-1

# ============================================================
# CELL 4: Set your Anthropic API key
# ============================================================
# import os
# os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."  # paste your key

# ============================================================
# CELL 5: Configure paths
# ============================================================
# VIDEO_PATH = "/content/drive/MyDrive/path/to/your/surgery_video.mp4"
# OUTPUT_DIR = "/content/drive/MyDrive/highlight_output/"

# ============================================================
# CELL 6: Run the pipeline
# ============================================================
# from src.pipeline import HighlightPipeline
# import yaml
#
# with open("config/default.yaml") as f:
#     config = yaml.safe_load(f)
#
# pipeline = HighlightPipeline(config)
# reel_path, report = pipeline.run(VIDEO_PATH, OUTPUT_DIR)
#
# print(report)

# ============================================================
# CELL 7: Play the highlight reel in Colab
# ============================================================
# from IPython.display import HTML
# from base64 import b64encode
#
# with open(str(reel_path), "rb") as f:
#     video_data = b64encode(f.read()).decode()
#
# HTML(f'''
# <video width="640" controls>
#   <source src="data:video/mp4;base64,{video_data}" type="video/mp4">
# </video>
# ''')
