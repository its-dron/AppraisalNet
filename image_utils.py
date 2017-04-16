import tensorflow as tf

# For safety, always assume image must be in [0,1] range

def crop_and_resize_proper(image, out_shape):
    '''
    Rescales image by:
    1) 0-Padding Image to a square
    2) Cropping center to a square using the average dim-length
    3) Rescale to out_shape
    '''
    im_shape = tf.shape(image)[0:2]
    crop_len = tf.to_int32(tf.reduce_mean(im_shape))
    im_square = tf.image.resize_image_with_crop_or_pad(image, crop_len, crop_len)
    result = tf.image.resize_images(im_square, out_shape[0:2])
    return result

def random_crop_and_resize_proper(image, out_shape):
    '''
    Randomly crops a square thats has sidelength:
        SIDELENGTH_PERCENTAGE*${shortest_side}.
    Then resizes to out_shape
    '''
    im_shape = tf.shape(image)[0:2]
    n_channels = tf.shape(image)[2]
    longest_len = tf.to_int32(tf.maximum(im_shape[0], im_shape[1]))
    crop_len = tf.to_int32(tf.reduce_mean(im_shape))
    im_square = tf.image.resize_image_with_crop_or_pad(image, longest_len, longest_len)
    rand_crop = tf.random_crop(im_square, size=[crop_len, crop_len, n_channels])
    resized = tf.image.resize_images(rand_crop, out_shape[0:2])
    return resized

def random_color_augmentation(image):
    '''
    Randomly augments the image color attributes
    (brightness, saturation, hue, contrast)

    Expects input to be in [0,1] Range
    '''
    BRIGHTNESS_MAX_DELTA = 28.0/255.0
    SATURATION_LOWER = 0.6
    SATURATION_UPPER = 1.4
    HUE_MAX_DELTA = 0.15
    CONTRAST_LOWER = 0.6
    CONTRAST_UPPER = 1.4

    order = tf.convert_to_tensor([0,1,2,3])
    order = tf.random_shuffle(order,seed=None)

    def f1(im):
        return tf.image.random_brightness(im,
                    max_delta=BRIGHTNESS_MAX_DELTA)
    def f2(im):
        return tf.image.random_saturation(im,
                    lower=SATURATION_LOWER, upper=SATURATION_UPPER)
    def f3(im):
        return tf.image.random_hue(im,
                    max_delta=HUE_MAX_DELTA)
    def f4(im):
        return tf.image.random_contrast(im,
                    lower=CONTRAST_LOWER, upper=CONTRAST_UPPER)

    def body(i, im):
        im = tf.case({
                tf.equal(tf.constant(1), order[i]): lambda: \
                    (tf.image.random_brightness(im,max_delta=BRIGHTNESS_MAX_DELTA)),
                tf.equal(tf.constant(2), order[i]): lambda: \
                    (tf.image.random_saturation(im,lower=SATURATION_LOWER, upper=SATURATION_UPPER)),
                tf.equal(tf.constant(3), order[i]): lambda: \
                    (tf.image.random_hue(im,max_delta=HUE_MAX_DELTA)),
                tf.equal(tf.constant(4), order[i]): lambda: \
                    (tf.image.random_contrast(im,
                    lower=CONTRAST_LOWER, upper=CONTRAST_UPPER))},
                default=lambda:(im), exclusive=True)
        im = tf.reshape(im, [224,224,3])
        return (i+1, im)
    counter = tf.constant(0)
    c = lambda i, im: tf.less(i,4)
    _, image = tf.while_loop(c, body, loop_vars=(counter, image))

    image = tf.clip_by_value(image, 0.0, 1.0)
    return image

