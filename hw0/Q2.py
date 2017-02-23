#!/usr/local/bin/python3.5
import sys
from PIL import Image

try:
    infilename1 = sys.argv[1]
    infilename2 = sys.argv[2]
    img1 = Image.open(infilename1)
    img2 = Image.open(infilename2)
except Exception as e:
    print('Error:', str(e))
    sys.exit(1)

list1 = list(img1.getdata())
list2 = list(img2.getdata())

out_list = []

for i in range(len(list1)):
    out_list.append((0, 0, 0, 0) if list1[i] == list2[i] else list2[i])

out_img = Image.new(img1.mode, img1.size)
out_img.putdata(out_list)
out_img.save('ans_two.png', 'PNG')

