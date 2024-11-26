import sys
import re
import time
import os
import cv2
import math
from PIL import Image
from PIL.ExifTags import TAGS

input_file = sys.argv[1]
output_file = sys.argv[2]

# Open the image
image = Image.open(input_file)

# Construct the regular expression for user information
pat = re.compile(r'USER=(?P<name>.+):(?P<uid>\d+|\w){3,};')

# Write the metadata header
fout = open(output_file, 'w')
fout.write('== metadata ==\n')
fout.flush()

# Attempt to find faces
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cv2_image = cv2.imread(input_file)
gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
faces = face_classifier.detectMultiScale(gray, 1.3, 5)

# Count how many faces we found and write the result to the metadata
fout.write('Faces: %d\n' % len(faces))
fout.flush()

# How much contrast is in the image?
contrast = cv2.Laplacian(gray, cv2.CV_64F).var()
fout.write('Contrast: %f\n' % contrast)
fout.flush()

# How big is the image?
width, height = image.size
area = width * height
fout.write('Area: %d\n' % area)

# Compute the interesting-ness score
score = math.log(len(faces) * contrast * area + 1)
fout.write('Score: %f\n' % score)
fout.flush()

meta = image.getexif()
for tag, value in sorted(meta.items(), key=lambda x: x[0]):
    tag_name = TAGS.get(tag, tag)
    print(f'Found tag: {tag_name}')
    if tag_name == 'UserComment':
        print(f'UserComment: {value}')
        if isinstance(value, bytes):
            value = value.decode()
        else:
            value = str(value)
        fout.write('UserComment: %s\n' % value)
        fout.flush()
        # Extract the user information if it is present
        match = pat.match(value)
        if match:
            fout.write('User: %s\n' % match.group('name'))
            fout.write('UID: %s\n' % match.group('uid'))
    elif tag_name == 'DateTime':
        if isinstance(value, bytes):
            value = value.decode()
        else:
            value = str(value)
        fout.write('DateTime: %s\n' % value)
        fout.flush()
    elif tag_name == 'ImageDescription':
        if isinstance(value, bytes):
            value = value.decode()
        else:
            value = str(value)
        fout.write('ImageDescription: %s\n' % value)
        fout.flush()

fout.write('Timestamp: %s\n' % time.ctime())
fout.write('Filename: %s\n' % os.path.basename(input_file))
fout.write('== end ==\n')

print('Done')

fout.close()    
