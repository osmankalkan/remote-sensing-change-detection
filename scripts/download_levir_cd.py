"""LEVIR-CD Dataset Download Script.

Supports downloading via:
1. HuggingFace Hub (e.g. datasets/nils-werner/LEVIR-CD or similar)
2. Direct zip link / gdown
"""
import os
import argparse
import urllib.request
import zipfile
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    print(f"Downloading from {url} to {output_path}...")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def main():
    parser = argparse.ArgumentParser(description="Download LEVIR-CD Dataset")
    parser.add_argument("--dest_dir", type=str, default="data/raw", help="Target extraction directory")
    parser.add_argument("--method", choices=["synthetic", "huggingface", "url"], default="synthetic",
                        help="Download method (synthetic for quick local smoke tests, huggingface/url for full dataset)")
    parser.add_argument("--url", type=str, default=None, help="Custom direct download URL for LEVIR-CD zip")
    args = parser.parse_args()

    os.makedirs(args.dest_dir, exist_ok=True)

    if args.method == "synthetic":
        from prepare_sample_data import prepare_sample_dataset
        prepare_sample_dataset(raw_dir=args.dest_dir)
    elif args.method == "huggingface":
        print("\nDownloading LEVIR-CD from Hugging Face...")
        print("Run:")
        print("  pip install huggingface_hub")
        print("  huggingface-cli download <dataset_repo> --local-dir data/raw --repo-type dataset")
    elif args.method == "url":
        if not args.url:
            print("Please specify --url <zip_url>")
            return
        zip_path = os.path.join(args.dest_dir, "levir_cd.zip")
        download_url(args.url, zip_path)
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(args.dest_dir)
        print("Extraction complete!")


if __name__ == "__main__":
    main()
