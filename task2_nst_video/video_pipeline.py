import cv2
import torch
# TODO: Import NST and Matting models

# TODO: Decode input_video.mp4
# TODO: For each frame:
#       1. Run matting to get alpha matte
#       2. Run NST on the frame (using previous stylized frame to initialize for temporal consistency)
#       3. Composite:
#          - Background stylized: O = alpha * F + (1 - alpha) * S
#          - Subject stylized: O = alpha * S + (1 - alpha) * F
#          - Fully stylized: O = S
# TODO: Save output videos
