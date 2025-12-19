import os
import subprocess
import sys

# Set the GPU to use with "0" (or CPU with "")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Forward any extra command-line args
cmd = [sys.executable, "test_video.py"] + sys.argv[1:]

subprocess.run(cmd, check=True)


# run the command:
# uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
# uv pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
