# src/feature_extractor.py
import cv2
import numpy as np
from scipy.fftpack import dct
from skimage.restoration import estimate_sigma
from skimage.util import img_as_float
from skimage.feature import local_binary_pattern


def resize_and_crop(img, size=(128, 128)):
    return cv2.resize(img, size)

def extract_dct_features(img, block_size=10):
    """
    提取图像的 DCT 特征，取左上角 10x10 的低频区域。
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img_as_float(img)
    dct_map = dct(dct(img.T, norm='ortho').T, norm='ortho')
    return dct_map[:block_size, :block_size].flatten()

def total_variation(img):
    """
    计算图像的总变差（Total Variation）
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tv = np.sum(np.abs(np.diff(img, axis=0))) + np.sum(np.abs(np.diff(img, axis=1)))
    return np.array([tv / (img.shape[0] * img.shape[1])], dtype=np.float32)

def extract_features(img):
    """
    提取综合特征：DCT + TV + LBP histogram + Color statistics + Noise RMS
    增加边缘密度、灰度直方图统计量、纹理熵、色彩偏斜度
    """
    img = resize_and_crop(img)
    dct_feat = extract_dct_features(img)
    tv_feat = total_variation(img)
    lbp_feat = extract_lbp_hist(img)
    color_feat = color_statistics(img)
    noise_feat = noise_rms(img)
    edge_feat = edge_density(img)
    gray_hist_feat = gray_histogram_stats(img)
    entropy_feat = texture_entropy(img)
    skewness_feat = color_skewness(img)
    return np.concatenate([
        dct_feat, tv_feat, lbp_feat, color_feat, noise_feat,
        edge_feat, gray_hist_feat, entropy_feat, skewness_feat,
        fft_energy(img),
        jpeg_block_artifacts(img),
        color_channel_shift(img),
        multi_scale_lbp(img)
    ])


# Helper functions
def extract_lbp_hist(img, P=8, R=1, bins=10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P, R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, bins + 1), density=True)
    return hist.astype(np.float32)

def color_statistics(img):
    stats = []
    for space in [cv2.COLOR_BGR2RGB, cv2.COLOR_BGR2HSV]:
        converted = cv2.cvtColor(img, space)
        mean = np.mean(converted, axis=(0, 1))
        std = np.std(converted, axis=(0, 1))
        stats.extend(mean)
        stats.extend(std)
    return np.array(stats, dtype=np.float32)

def noise_rms(img):
    """
    提取图像的噪声残差 RMS 特征
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    residual = gray - blur
    rms = np.sqrt(np.mean(residual ** 2))
    return np.array([rms], dtype=np.float32)


# 批量提取特征，可选PCA降维
from concurrent.futures import ThreadPoolExecutor
def extract_features_batch(image_list, use_pca=False, n_components=100, verbose=False):
    features = []
    with ThreadPoolExecutor() as executor:
        if verbose:
            import sys
            from tqdm import tqdm
            results = list(tqdm(executor.map(extract_features, image_list), total=len(image_list), file=sys.stdout))
        else:
            results = list(executor.map(extract_features, image_list))
        features = np.array(results)

    if use_pca:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        features = pca.fit_transform(features)
    return features
# 辅助特征函数
def edge_density(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    return np.array([density], dtype=np.float32)

def gray_histogram_stats(img, bins=16):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256]).flatten()
    hist = hist / np.sum(hist)
    mean = np.mean(hist)
    std = np.std(hist)
    return np.array([mean, std], dtype=np.float32)

def texture_entropy(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / np.sum(hist)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    return np.array([entropy], dtype=np.float32)

from scipy.stats import skew
def color_skewness(img):
    skewness_vals = []
    for c in cv2.split(img):
        skewness_vals.append(skew(c.reshape(-1)))
    return np.array(skewness_vals, dtype=np.float32)

def fft_energy(img):
    """
    计算图像的频域能量（高频和低频比例）
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    h, w = magnitude_spectrum.shape
    center = (int(h / 2), int(w / 2))
    low_freq_energy = np.sum(magnitude_spectrum[center[0]-10:center[0]+10, center[1]-10:center[1]+10])
    total_energy = np.sum(magnitude_spectrum)
    ratio = low_freq_energy / (total_energy + 1e-8)
    return np.array([ratio], dtype=np.float32)

def jpeg_block_artifacts(img):
    """
    估计图像中的 JPEG 压缩伪影强度
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    block_diff = []
    for i in range(8, h, 8):
        block_diff.append(np.mean(np.abs(gray[i, :] - gray[i - 1, :])))
    for j in range(8, w, 8):
        block_diff.append(np.mean(np.abs(gray[:, j] - gray[:, j - 1])))
    return np.array([np.mean(block_diff)], dtype=np.float32)

def color_channel_shift(img):
    """
    计算颜色通道之间的均值差异
    """
    b, g, r = cv2.split(img.astype(np.float32))
    rg = np.mean(np.abs(r - g))
    gb = np.mean(np.abs(g - b))
    br = np.mean(np.abs(b - r))
    return np.array([rg, gb, br], dtype=np.float32)

def multi_scale_lbp(img, radii=[1, 2, 3]):
    """
    多尺度 LBP 特征拼接
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feats = []
    for R in radii:
        P = 8 * R
        lbp = local_binary_pattern(gray, P, R, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), density=True)
        feats.extend(hist)
    return np.array(feats, dtype=np.float32)