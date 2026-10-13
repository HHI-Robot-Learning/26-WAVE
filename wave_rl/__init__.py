import sys

# progress lines reach a batch log as they are printed, not when a buffer fills
sys.stdout.reconfigure(line_buffering=True)
