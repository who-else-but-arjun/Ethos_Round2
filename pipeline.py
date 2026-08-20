from matplotlib import pyplot as plt
from DeblurGANv2 import deblur_image
from SRCNN import enhance_images
from LIME import enhance_image

def main(image):
    lighten = enhance_image(image)
    enhanced = enhance_images(lighten)
    deblurred = deblur_image(enhanced)   
    final = deblurred 
    return final
