
"""
Palm Leaf OCR - Final Corrected Version
- Manually keeps lines 1-5 (skip first and last)
- No density filtering (since densities vary)
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import fitz
import time

# ============================================================
# 1. PDF → Images
# ============================================================
def pdf_to_images(pdf_path, dpi=200, first_page=1, last_page=None, out_dir="pages"):
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    if last_page is None:
        last_page = len(doc)
    else:
        last_page = min(last_page, len(doc))
    image_paths = []
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    for page_num in range(first_page-1, last_page):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=mat)
        img_path = os.path.join(out_dir, f"page_{page_num+1}.png")
        pix.save(img_path)
        image_paths.append(img_path)
        print(f"Saved {img_path}")
    doc.close()
    return image_paths

# ============================================================
# 2. Manual Preprocessing
# ============================================================
def manual_convolution(image, kernel):
    from numpy.lib.stride_tricks import sliding_window_view
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    windows = sliding_window_view(padded, (kh, kw))
    result = np.tensordot(windows, kernel, axes=((2, 3), (0, 1)))
    return result

def manual_hist_eq(image):
    if image.dtype.kind == 'f':
        image = image.astype(np.uint8)
    flat = image.flatten()
    hist = np.bincount(flat, minlength=256)
    cdf = hist.cumsum()
    cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min()) * 255
    return cdf[image].astype(np.uint8)

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=np.float32) / 16.0
    blurred = manual_convolution(gray, kernel)
    blurred = np.clip(blurred, 0, 255).astype(np.uint8)
    equalized = manual_hist_eq(blurred)
    thresh = np.mean(equalized)
    binary = (equalized > thresh).astype(np.uint8) * 255
    if np.mean(binary) < 127:
        binary = 255 - binary
    return binary

# ============================================================
# 3. Line Segmentation (no filtering, just raw extraction)
# ============================================================
def segment_lines(binary_image, kernel_size=15, min_line_height=25):
    proj = np.sum(binary_image == 255, axis=1)
    smooth = np.convolve(proj, np.ones(kernel_size)/kernel_size, mode='same')
    threshold = 0.4 * np.max(smooth)
    lines = []
    in_line = False
    start = 0
    for i, val in enumerate(smooth):
        if val > threshold and not in_line:
            start = i
            in_line = True
        elif val <= threshold and in_line:
            if i - start > min_line_height:
                lines.append((start, i))
            in_line = False
    if in_line:
        lines.append((start, len(smooth)))
    line_images = []
    for (s, e) in lines:
        line_img = binary_image[s:e, :]
        line_img = line_img[:, np.any(line_img == 255, axis=0)]
        if line_img.shape[1] > 10:
            line_images.append(line_img)
    return line_images

# ============================================================
# 4. Manual CNN Feature Extraction
# ============================================================
EDGE_KERNEL = np.array([[1,0,-1],[1,0,-1],[1,0,-1]], dtype=np.float32)

def conv2d(image, kernel):
    from numpy.lib.stride_tricks import sliding_window_view
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh//2, kw//2
    padded = np.pad(image, ((pad_h,pad_h),(pad_w,pad_w)), mode='constant')
    windows = sliding_window_view(padded, (kh, kw))
    return np.tensordot(windows, kernel, axes=((2,3),(0,1)))

def relu(x):
    return np.maximum(0, x)

def max_pool2d(x, size=2):
    h, w = x.shape
    h_crop = h - (h % size)
    w_crop = w - (w % size)
    x_crop = x[:h_crop, :w_crop]
    return x_crop.reshape(h_crop//size, size, w_crop//size, size).max(axis=(1,3))

def manual_resize(image, target_height):
    h, w = image.shape
    if h == target_height:
        return image
    rows = np.round(np.linspace(0, h-1, target_height)).astype(int)
    return image[rows, :]

def extract_features(line_img, fixed_height=32):
    img = line_img.astype(np.float32) / 255.0
    conv = conv2d(img, EDGE_KERNEL)
    relu_out = relu(conv)
    pooled = max_pool2d(relu_out, size=2)
    if pooled.shape[0] != fixed_height:
        pooled = manual_resize(pooled, fixed_height)
    return pooled

def feature_map_to_sequence(fm):
    return fm.T

# ============================================================
# 5. Character Set (English lowercase + punctuation)
# ============================================================
ALLOWED_CHARS = " abcdefghijklmnopqrstuvwxyz.-'"
char_to_idx = {ch: i+1 for i, ch in enumerate(ALLOWED_CHARS)}
idx_to_char = {idx: ch for ch, idx in char_to_idx.items()}
blank_idx = 0
num_classes = len(ALLOWED_CHARS) + 1

def encode_text(text):
    return [char_to_idx[ch] for ch in text.lower() if ch in char_to_idx]

def decode_indices(indices):
    prev = blank_idx
    result = []
    for idx in indices:
        if idx != prev and idx != blank_idx:
            result.append(idx_to_char[int(idx)])
        prev = idx
    return ''.join(result)

# ============================================================
# 6. Dataset
# ============================================================
class LineDataset(Dataset):
    def __init__(self, line_images, texts):
        self.samples = []
        for img, txt in zip(line_images, texts):
            fm = extract_features(img)
            seq = feature_map_to_sequence(fm)
            label = encode_text(txt)
            if len(label) == 0:
                continue
            self.samples.append((seq, label))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        seq, label = self.samples[idx]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def collate_fn(batch):
    seqs, labels = zip(*batch)
    seq_lengths = [s.shape[0] for s in seqs]
    padded_seqs = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    target_lengths = [len(l) for l in labels]
    targets = torch.cat(labels)
    return padded_seqs, torch.tensor(seq_lengths), targets, torch.tensor(target_lengths)

# ============================================================
# 7. LSTM + CTC Model
# ============================================================
class OCRModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=2)
    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = self.fc(out)
        return self.log_softmax(out)

# ============================================================
# 8. Training with Live Tracking
# ============================================================
def train_model(model, train_loader, val_loader, epochs=30, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ctc_loss = nn.CTCLoss(blank=blank_idx, reduction='mean', zero_infinity=True)
    
    print(f"\n{'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12} {'Time (s)':<8}")
    print("-" * 45)
    
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            padded_seqs, seq_lengths, targets, target_lengths = batch
            padded_seqs = padded_seqs.to(device)
            seq_lengths = seq_lengths.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            log_probs = model(padded_seqs, seq_lengths)
            log_probs = log_probs.permute(1, 0, 2)
            loss = ctc_loss(log_probs, targets, seq_lengths, target_lengths)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                padded_seqs, seq_lengths, targets, target_lengths = batch
                padded_seqs = padded_seqs.to(device)
                seq_lengths = seq_lengths.to(device)
                targets = targets.to(device)
                target_lengths = target_lengths.to(device)
                log_probs = model(padded_seqs, seq_lengths).permute(1, 0, 2)
                loss = ctc_loss(log_probs, targets, seq_lengths, target_lengths)
                val_loss += loss.item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        elapsed = time.time() - start_time
        print(f"{epoch+1:<6} {avg_train:<12.4f} {avg_val:<12.4f} {elapsed:<8.2f}")
    
    return model

# ============================================================
# 9. Single line prediction
# ============================================================
def predict_line(model, line_img, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        fm = extract_features(line_img)
        seq = feature_map_to_sequence(fm)
        seq_t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
        seq_len = torch.tensor([seq.shape[0]]).to(device)
        log_probs = model(seq_t, seq_len)
        probs = torch.exp(log_probs.squeeze(0))
        indices = torch.argmax(probs, dim=1).cpu().numpy()
        return decode_indices(indices)

# ============================================================
# 10. MAIN - Keep only lines 1..5 (skip first and last)
# ============================================================
def main():
    pdf_path = r"C:/Users/skmda/Videos/gita.pdf"
    out_dir = r"C:\Users\skmda\OneDrive\Documents\iit_tirupati\OCR_training"
    train_page = 7
    test_page = None          # Set to 8 if you want to test on page 8
    epochs = 30
    batch_size = 4
    
    # --- Step 1: Load page 7 and extract raw lines ---
    print(f"Converting page {train_page} from PDF...")
    train_paths = pdf_to_images(pdf_path, dpi=200, first_page=train_page, last_page=train_page, out_dir=out_dir)
    binary = preprocess_image(train_paths[0])
    raw_lines = segment_lines(binary, kernel_size=15, min_line_height=25)
    print(f"Raw lines on page {train_page}: {len(raw_lines)}")
    
    # Print densities for debugging
    for i, line in enumerate(raw_lines):
        density = np.sum(line == 255) / (line.shape[0] * line.shape[1])
        print(f"Line {i}: density={density:.3f}")
    
    # --- Keep only lines 1 to 5 (indices 1 through 5) ---
    # Assumes raw_lines[0] and raw_lines[6] are blank or unwanted
    text_lines = raw_lines[1:6]   # lines 1,2,3,4,5
    print(f"\nKept {len(text_lines)} text lines (indices 1 to 5).")
    
    if len(text_lines) == 0:
        print("No lines kept. Check segmentation.")
        return
    
    # Show kept lines for visual verification
    for i, line in enumerate(text_lines):
        plt.imshow(line, cmap='gray')
        plt.title(f"Training line {i} (original index {i+1})")
        plt.axis('off')
        plt.show()
        if i >= 4:
            break
    
    # --- Step 2: Load labels (must have exactly 5 non-empty lines) ---
    label_file = "labels.txt"
    if not os.path.exists(label_file):
        print(f"\nERROR: {label_file} not found! Create with {len(text_lines)} non-empty lines.")
        return
    
    with open(label_file, 'r', encoding='utf-8') as f:
        all_labels = [line.strip() for line in f.readlines() if line.strip() != ""]
    
    if len(all_labels) < len(text_lines):
        print(f"ERROR: labels.txt has {len(all_labels)} non-empty lines, need {len(text_lines)}.")
        print("Please add more labels or adjust the line selection.")
        return
    labels = all_labels[:len(text_lines)]
    
    # Verify labels can be encoded
    print("\n=== Label verification ===")
    for i, label in enumerate(labels):
        encoded = encode_text(label)
        print(f"Line {i}: label='{label[:50]}' (encoded length={len(encoded)})")
        if len(encoded) == 0:
            print(f"  → WARNING: This label contains no allowed characters! It will be skipped.")
    
    # --- Step 3: Create dataset and split ---
    dataset = LineDataset(text_lines, labels)
    if len(dataset) == 0:
        print("No valid samples after encoding. Check labels (allowed: a-z, space, . - ').")
        return
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    sample_seq, _ = dataset[0]
    input_size = sample_seq.shape[1]
    print(f"\nFeature dimension: {input_size}")
    print(f"Training samples: {train_size}, Validation samples: {val_size}")
    
    # --- Step 4: Build and train model ---
    model = OCRModel(input_size=input_size, hidden_size=128, num_layers=2)
    print("\nStarting training...")
    model = train_model(model, train_loader, val_loader, epochs=epochs, lr=0.001)
    
    torch.save(model.state_dict(), "palm_ocr_trained.pth")
    print("\nModel saved as palm_ocr_trained.pth")
    
    # --- Step 5: Test on the same training lines (overfitting check) ---
    print("\n=== Testing on training lines (same page) ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    for i, line in enumerate(text_lines):
        pred = predict_line(model, line, device)
        print(f"Train line {i}: Ground truth = '{labels[i]}'")
        print(f"         Prediction = '{pred}'\n")
    
    # --- Step 6: Optionally test on page 8 ---
    if test_page is not None:
        print(f"\n--- Testing on page {test_page} ---")
        test_path = pdf_to_images(pdf_path, dpi=200, first_page=test_page, last_page=test_page, out_dir=out_dir)[0]
        binary_test = preprocess_image(test_path)
        raw_test = segment_lines(binary_test, kernel_size=15, min_line_height=25)
        # For page 8, keep all lines (or use a similar selection) – here we keep all with density > 0.1
        text_test = [line for line in raw_test if np.sum(line==255)/(line.shape[0]*line.shape[1]) > 0.1]
        print(f"Found {len(text_test)} text lines on page {test_page}")
        for i, line in enumerate(text_test):
            pred = predict_line(model, line, device)
            print(f"Page {test_page} line {i}: {pred}")

if __name__ == "__main__":
    main()