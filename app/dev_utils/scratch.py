from PIL import Image



def convert_image_to_pdf(image_path, pdf_path):
    image = Image.open(image_path)
    image.convert("RGB").save(pdf_path)

convert_image_to_pdf("input_image.png", "output.pdf")