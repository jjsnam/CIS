import os
import shutil
import random
from collections import defaultdict
from tqdm import tqdm

def extract_real_identity(filename):
    # e.g. "00001_face_0.jpg" -> "00001"
    return filename.split('_')[0]

def extract_fake_identities(filename):
    # e.g. "id0_id1_0000_face_163.jpg" -> "id0_id1"
    return '_'.join(filename.split('_')[:2])  # e.g. id0_id1

def subsample_images(input_dir, output_dir, max_per_identity=3, is_fake=False):
    os.makedirs(output_dir, exist_ok=True)

    identity_to_files = defaultdict(list)

    # 遍历文件分组
    for fname in os.listdir(input_dir):
        if not fname.endswith(".jpg"):
            continue
        identity = extract_fake_identities(fname) if is_fake else extract_real_identity(fname)
        identity_to_files[identity].append(fname)

    # 随机采样并复制
    for identity, files in tqdm(identity_to_files.items(), desc="Sampling"):
        sampled = random.sample(files, min(len(files), max_per_identity))
        for fname in sampled:
            src = os.path.join(input_dir, fname)
            dst = os.path.join(output_dir, fname)
            shutil.copy(src, dst)

    print(f"Total identities processed: {len(identity_to_files)}")
    print(f"Output saved to: {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_input', type=str, required=True, help="Input dir for real images")
    parser.add_argument('--fake_input', type=str, required=True, help="Input dir for fake images")
    parser.add_argument('--real_output', type=str, required=True, help="Output dir for real images")
    parser.add_argument('--fake_output', type=str, required=True, help="Output dir for fake images")
    parser.add_argument('--max_per_identity', type=int, default=3, help="Max images per identity")
    args = parser.parse_args()

    print("Subsampling REAL...")
    subsample_images(args.real_input, args.real_output, max_per_identity=args.max_per_identity, is_fake=False)
    print("Subsampling FAKE...")
    subsample_images(args.fake_input, args.fake_output, max_per_identity=args.max_per_identity, is_fake=True)

""" 
python subsample_identities.py \
  --real_input /root/Project/datasets/Celeb_V2/Train/real \
  --fake_input /root/Project/datasets/Celeb_V2/Train/fake \
  --real_output /root/Project/datasets/Celeb_V2_Subset/Train/real \
  --fake_output /root/Project/datasets/Celeb_V2_Subset/Train/fake \
  --max_per_identity 3
"""