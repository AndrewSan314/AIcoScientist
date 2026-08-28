import io
import urllib.request
import zipfile

from src.utils import SEM_IMAGE_DIR, ensure_dirs


ARCHIVE_URL = (
    "https://codeload.github.com/MIIMSEKAIST/"
    "CNN_for_NCM-composition-and-state-prediction/zip/refs/heads/main"
)
IMAGE_NUMBERS = range(453, 456)
MAX_ARCHIVE_BYTES = 25_000_000
MAX_IMAGE_BYTES = 5_000_000


def _download_archive(url=ARCHIVE_URL):
    req = urllib.request.Request(url, headers={"User-Agent": "GTIP-SEM-demo/1.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        if response.status != 200:
            raise RuntimeError(f"Unexpected archive response from {url}")
        data = response.read(MAX_ARCHIVE_BYTES + 1)
    if len(data) > MAX_ARCHIVE_BYTES:
        raise RuntimeError(f"Archive exceeds {MAX_ARCHIVE_BYTES} bytes")
    return data


def fetch_sem_demo_images(count=3):
    if not 1 <= count <= len(IMAGE_NUMBERS):
        raise ValueError(f"count must be between 1 and {len(IMAGE_NUMBERS)}")
    ensure_dirs()
    paths = []
    with zipfile.ZipFile(io.BytesIO(_download_archive())) as archive:
        members = {name.rsplit("/", 1)[-1]: name for name in archive.namelist()}
        for number in list(IMAGE_NUMBERS)[:count]:
            source = f"ETRI_ 20 kV_SE_HighVac_x500 __{number}.jpg"
            info = archive.getinfo(members[source])
            if info.file_size > MAX_IMAGE_BYTES:
                raise RuntimeError(f"{source} exceeds {MAX_IMAGE_BYTES} bytes")
            path = SEM_IMAGE_DIR / f"NCM622_CYCLED_{number}.jpg"
            path.write_bytes(archive.read(info))
            paths.append(path)
    return paths


def main():
    paths = fetch_sem_demo_images()
    print(f"Saved {len(paths)} SEM demo images:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
