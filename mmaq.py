import cv2
import os
import numpy as np
import heapq


class Node:
    def __init__(self, freq, symbol=None, left=None, right=None):
        self.freq = freq
        self.symbol = symbol  # leaf node stores pixel value
        self.left = left
        self.right = right
    def __lt__(self, other):
        return self.freq < other.freq  # needed by heapq

def build_huffman_tree(histogram):
    """Build Huffman tree from histogram (list of frequencies)."""
    pq = []
    for val, freq in enumerate(histogram):
        if freq > 0:
            heapq.heappush(pq, Node(freq, symbol=val))

    if len(pq) == 1:
        # Only one symbol: create parent with a single child
        only_node = heapq.heappop(pq)
        return Node(only_node.freq, left=only_node)

    while len(pq) > 1:
        left = heapq.heappop(pq)
        right = heapq.heappop(pq)
        parent = Node(left.freq + right.freq, left=left, right=right)
        heapq.heappush(pq, parent)

    return heapq.heappop(pq)

def generate_huffman_codes(root):
    """Traverse Huffman tree to assign codes for each symbol."""
    codebook = {}
    def traverse(node, code=""):
        if node.symbol is not None:
            codebook[node.symbol] = code or "0"  # single-symbol case
        else:
            traverse(node.left, code + "0")
            traverse(node.right, code + "1")
    traverse(root)
    return codebook

def get_frame_number(fname):
    return int(fname.replace('frame','').replace('.jpg',''))

# -----------------------
# Video to frame differences + histogram
# -----------------------

folder = "./Frame_Folder"
diff_folder = "./Diff_Folder"
os.makedirs(folder, exist_ok=True)
os.makedirs(diff_folder, exist_ok=True)

width, height = 1920, 1080

cap = cv2.VideoCapture('./Social_Network.avi')
count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(os.path.join(folder, f"frame{count}.jpg"), frame)
    count += 1
cap.release()

files = [f for f in os.listdir(folder) if f.startswith('frame')]
files.sort(key=get_frame_number)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('Social_Net_diff.avi', fourcc, 24, (width, height), False)

hist = np.zeros(256, dtype=np.int64)

for i in range(len(files) - 1):
    img1 = cv2.imread(os.path.join(folder, files[i]), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(os.path.join(folder, files[i+1]), cv2.IMREAD_GRAYSCALE)
    diff = cv2.absdiff(img2, img1)

    cv2.imwrite(os.path.join(diff_folder, f"Diff{i}.jpg"), diff)
    video.write(diff)

    vals, counts = np.unique(diff, return_counts=True)
    hist[vals] += counts

video.release()

# -----------------------
# Build Huffman code and compute compression
# -----------------------

root = build_huffman_tree(hist)
codebook = generate_huffman_codes(root)
total_bits=0
original_bits = hist.sum() * 8
print("Huffman codes (pixel_value: code):")

for sym in sorted(codebook.keys()):
    total_bits += hist[sym] * len(codebook[sym])
    print(f"{sym}: {codebook[sym]}")  

if total_bits>0 :
    compression_ratio = original_bits / total_bits  
else :
    compression_ratio = 0

print("\nOriginal size (bits):", original_bits)
print("Compressed size (bits):", total_bits)
print("Compression ratio (original/compressed):", compression_ratio)