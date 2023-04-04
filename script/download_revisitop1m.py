import os
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from argparse import ArgumentParser


def download_distractors(dataset_root, num_images):
    assert num_images <= int(1e6), 'num_images cannot exceed 1M.'

    src_dir = 'http://ptak.felk.cvut.cz/revisitop/revisitop1m/jpg'
    src_file = 'revisitop1m.{}.tar.gz'

    dataset_dir = os.path.join(dataset_root, 'revisitop1m')
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    cur_images = 0
    for shard in range(1, 101):
        filename = src_file.format(shard)
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dataset_dir, filename)

        print(f'>> [{shard}/100] Downloading {src_path}...')
        urllib.request.urlretrieve(src_path, dst_path)

        print(f'>> [{shard}/100] Extracting {dst_path}...')
        with tempfile.TemporaryDirectory(dir=dataset_dir) as tmpdir:
            tar = tarfile.open(dst_path)
            tar.extractall(path=tmpdir)
            tar.close()

            for image_path in Path(tmpdir).rglob('*.jpg'):
                cur_images += 1
                image_name = f'revisitop1m-{str(cur_images).zfill(7)}.jpg'
                image_path.rename(os.path.join(dataset_dir, image_name))

                if cur_images == num_images:
                    os.remove(dst_path)
                    print(f'>> [{cur_images}/{num_images}] Finished')
                    return

        os.remove(dst_path)
        print(f'>> [{cur_images}/{num_images}] Images downloaded')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--num_images', type=int, default=100000)
    args = parser.parse_args()

    download_distractors('./dataset', args.num_images)
