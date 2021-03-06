from PIL import Image, ImageFilter

# Image load
load_img = Image.open("{YOUR IMAGE PATH}")
# Image check
print(load_img.size)
print(load_img.format)
load_img.show()

# Image resizing
resized_img = load_img.resize((100, 100))

# Image flip
flip_img = resized_img.transpose(Image.FLIP_LEFT_RIGHT)
# Getting individual RGB channel to image
r, g, b = resized_img.split()

# Processed image save
flip_img.save("{YOUR IMAGE PATH}", "PNG")
r.save("{YOUR IMAGE PATH}", "PNG")
g.save("{YOUR IMAGE PATH}", "PNG")
b.save("{YOUR IMAGE PATH}", "PNG")
