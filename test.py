from pathlib import Path
import PIL
from PIL import Image
from tqdm import tqdm


image_path = Path('images')
smaller_image_path = Path('small_images')


for image in tqdm(image_path.iterdir()):
    with open(image.as_posix(), 'rb') as f:
        im = Image.open(f)
        scale_factor = 400 / max(im.size)
        im = im.resize(size=(int(im.size[0] * scale_factor),
                             int(im.size[1] * scale_factor)))
        im = im.convert('RGB')

    with open(smaller_image_path / (image.name.split('.')[0] + '.jpg'), 'wb') as f:
        im.save(f, "JPEG")
