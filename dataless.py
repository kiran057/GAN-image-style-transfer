import os
import shutil

def reduce_dataset(path, keep=500):
    files = sorted(os.listdir(path))[:keep]
    reduced_path = path + '_reduced'
    os.makedirs(reduced_path, exist_ok=True)
    for f in files:
        shutil.copy(os.path.join(path, f), reduced_path)

reduce_dataset('datasets/your_dataset/trainA', 500)
reduce_dataset('datasets/your_dataset/trainB', 500)
