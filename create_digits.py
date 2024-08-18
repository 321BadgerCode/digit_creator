import os
from PIL import Image, ImageFont, ImageDraw

def text_to_image(
text:str,
font_filepath:str,
font_size:int,
color:tuple,
font_align="center") -> object:
	font = ImageFont.truetype(font_filepath, size=font_size)
	box = font.getsize(text)
	img = Image.new("RGBA", (box[0], box[1]))
	draw = ImageDraw.Draw(img)
	draw_point = (0, 0)
	draw.multiline_text(draw_point, text, font=font, fill=color, align=font_align)
	return img

if not os.path.exists("./digit/"):
	os.makedirs("./digit/")

fonts:list = [
	"/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
	"/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
	"/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
	"/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
	"/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-BoldOblique.ttf",
	"/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Oblique.ttf",
	"/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf",
	"/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
	"/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf",
	"/usr/share/fonts/truetype/freefont/FreeMonoOblique.ttf",
	"/usr/share/fonts/truetype/freefont/FreeSansOblique.ttf",
	"/usr/share/fonts/truetype/freefont/FreeSerifOblique.ttf"
]
for font in fonts:
	path:str = f"./digit/{font.split('/')[-1].split('.')[0]}"
	if os.path.exists(path):
		continue
	os.makedirs(path)
	for i in range(10):
		img:object = text_to_image(
			str(i),
			font,
			24,
			(0, 0, 0),
			"center"
		)
		img = img.resize((8, 8))
		img.save(f"{path}/{i}.png")