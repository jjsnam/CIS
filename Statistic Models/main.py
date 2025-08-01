# main.py
from src.train import train_loop

if __name__ == "__main__":
    train_loop("/root/Project/datasets/Celeb_V2/Train/real", "/root/Project/datasets/Celeb_V2/Train/fake",
               "/root/Project/datasets/Celeb_V2/Val/real", "/root/Project/datasets/Celeb_V2/Val/fake")