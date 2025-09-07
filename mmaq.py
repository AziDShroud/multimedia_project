import cv2
import os
import numpy as np
import heapq


def predict_frame_with_motion(prev_frame, curr_frame, block_size=16, search_range=4):
    """
    Compute motion-compensated prediction of curr_frame using prev_frame.
    Both frames must be grayscale numpy arrays of the same shape.
    Returns:
        predicted_frame: np.ndarray (uint8)
        motion_vectors: list of ((x,y), (dx,dy)) for debugging
    """
    h, w = curr_frame.shape
    predicted = np.zeros_like(curr_frame)
    motion_vectors = []

    # iterate over blocks
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            # current block
            curr_block = curr_frame[y:y+block_size, x:x+block_size]
            best_sad = float('inf')
            best_dx, best_dy = 0, 0

            # search window in prev frame
            for dy in range(-search_range, search_range+1):
                for dx in range(-search_range, search_range+1):
                    ref_x = x + dx
                    ref_y = y + dy

                    # check boundaries
                    if ref_x < 0 or ref_y < 0:
                        continue
                    if ref_x+block_size > w or ref_y+block_size > h:
                        continue

                    ref_block = prev_frame[ref_y:ref_y+block_size, ref_x:ref_x+block_size]
                    sad = np.sum(np.abs(curr_block.astype(np.int32) - ref_block.astype(np.int32)))

                    if sad < best_sad:
                        best_sad = sad
                        best_dx, best_dy = dx, dy

            # copy best matching block into predicted frame
            ref_block = prev_frame[y+best_dy:y+best_dy+block_size,
                                   x+best_dx:x+best_dx+block_size]
            predicted[y:y+block_size, x:x+block_size] = ref_block
            motion_vectors.append(((x, y), (best_dx, best_dy)))

    return predicted.astype(np.uint8), motion_vectors

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
    '''extracts frame number from file name'''
    return int(fname.replace('frame','').replace('.jpg',''))


def split_vid_into_frames(VideoPath='./Social_Network.avi'):
    '''splits video into frames and saves them as photos'''
    cap = cv2.VideoCapture(VideoPath)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(folder, f"frame{count}.jpg"), frame)
        count += 1
    cap.release()

def frame_differences(UseMotionPredictor:bool, SaveVideo:bool, FileName:str):
    '''Finds the differences between the next and predicted frame, saves the videos, 
       calculates and returns the histogram of the colour values for the pixels.'''
    
    files = [f for f in os.listdir(folder) if f.startswith('frame')]
    files.sort(key=get_frame_number)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(f'Social_Net_{FileName}.avi', fourcc, 24, (width, height), False) 
    hist = np.zeros(256, dtype=np.int64)

    for i in range(len(files) - 1):
        img1 = cv2.imread(os.path.join(folder, files[i]), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(os.path.join(folder, files[i+1]), cv2.IMREAD_GRAYSCALE)
        if UseMotionPredictor:
            predicted, _ = predict_frame_with_motion(img1, img2, block_size=16, search_range=4)
            diff = cv2.absdiff(img2, predicted) 
        else:
            diff = cv2.absdiff(img2, img1)

        cv2.imwrite(os.path.join(diff_folder, f"{FileName}{i}.jpg"), diff)
        if SaveVideo:
            video.write(diff)
        
        vals, counts = np.unique(diff, return_counts=True)
        hist[vals] += counts
    video.release()
    return hist

def huffman_from_hist(hist):
    '''Build Huffman code and compute compression'''
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

###############################################################################

if __name__ == "__main__":
    folder = "./Frame_Folder"
    diff_folder = "./Diff_Folder"
    os.makedirs(folder, exist_ok=True)
    os.makedirs(diff_folder, exist_ok=True)

    width, height = 1920, 1080
    split_vid_into_frames()
    # question a
    hist = frame_differences(False,True,"diff_simple_predictor")
    # question b
    huffman_from_hist(hist)
    # question c
    hist = frame_differences(True,True,"diff_motion_predictor")
    huffman_from_hist(hist)




