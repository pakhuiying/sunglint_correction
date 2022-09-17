import PIL.Image
import io
import base64
import numpy as np

def convert_to_bytes(file_or_bytes, resize=None):
    '''
    Will convert into bytes and optionally resize an image that is a file or a base64 bytes object.
    Turns into  PNG format in the process so that can be displayed by tkinter
    :param file_or_bytes: either a string filename or a bytes base64 image object
    :type file_or_bytes:  (Union[str, bytes])
    :param resize:  optional new size
    :type resize: (Tuple[int, int] or None)
    :return: (bytes) a byte-string object
    :rtype: (bytes)
    *Preview only works for RGB images
    '''
    if isinstance(file_or_bytes, str):
        img = PIL.Image.open(file_or_bytes)
    else:
        try:
            img = PIL.Image.open(io.BytesIO(base64.b64decode(file_or_bytes)))
        except Exception as e:
            dataBytesIO = io.BytesIO(file_or_bytes)
            img = PIL.Image.open(dataBytesIO)

    cur_width, cur_height = img.size
    if resize:
        new_width, new_height = resize
        scale = min(new_height/cur_height, new_width/cur_width)
        img = img.resize((int(cur_width*scale), int(cur_height*scale)), PIL.Image.ANTIALIAS)
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    del img
    return bio.getvalue()

def array_to_bytes(file_or_img,resize=None,binary=True):
    """ 
    img (2D or 3D np.array)
    Have to represent it as an RGB image for preview to work
    """
    if isinstance(file_or_img, str):
        file_or_img = np.asarray(PIL.Image.open(file_or_img))

    if file_or_img.ndim == 2:
        img_rgb = np.repeat(file_or_img[:,:,np.newaxis],3,axis=2)
    elif file_or_img.ndim == 3:
        img_rgb = file_or_img

    if binary is True:
        img_rgb = img_rgb*255
        
    img = PIL.Image.fromarray(img_rgb,'RGB')
    cur_width, cur_height = img.size
    if resize:
        new_width, new_height = resize
        scale = min(new_height/cur_height, new_width/cur_width)
        img = img.resize((int(cur_width*scale), int(cur_height*scale)), PIL.Image.ANTIALIAS)
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    del img
    return bio.getvalue()
