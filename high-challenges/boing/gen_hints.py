from PIL import Image
from PIL.ExifTags import TAGS
import zipfile
import os

# Create an empty image of one pixel
image = Image.new('RGB', (1, 1))

exif = image.getexif()

# Write to the 'UserComment' tag
reverse_tags = {v: k for k, v in TAGS.items()}
exif[reverse_tags['UserComment']] = b'USER=foo:111231111111111111111111111111111111111111 ;'

img_out_fname = os.path.dirname(__file__) + '/img.jpg'
image.save(img_out_fname, exif=exif)

# zip the image
zip_fname = os.path.dirname(__file__) + '/hint.zip'
with zipfile.ZipFile(zip_fname, 'w') as zf:
    zf.write(img_out_fname, 'img.jpg')
