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
    Randomly augments the image color attributes in random operation order
    (brightness, saturation, hue, contrast)
    To replace random_color_augmentation()
    Expects input to be in [0,1] Range
    '''
    BRIGHTNESS_MAX_DELTA = 28.0/255.0
    SATURATION_LOWER = 0.6
    SATURATION_UPPER = 1.4
    HUE_MAX_DELTA = 0.15
    CONTRAST_LOWER = 0.6
    CONTRAST_UPPER = 1.4

    order = tf.convert_to_tensor([1,2,3,4])
    order = tf.random_shuffle(order,seed=None)

    def body(op_idx, image):
        image = tf.cond(tf.equal(order[op_idx], 1),
                lambda: tf.image.random_brightness(image,
                            max_delta=BRIGHTNESS_MAX_DELTA),
                lambda: image)
        image = tf.cond(tf.equal(order[op_idx], 2),
                lambda: tf.image.random_saturation(image,
                            lower=SATURATION_LOWER, upper=SATURATION_UPPER),
                lambda: image)
        image = tf.cond(tf.equal(order[op_idx], 3),
                lambda: tf.image.random_hue(image,
                            max_delta=HUE_MAX_DELTA),
                lambda: image)
        image = tf.cond(tf.equal(order[op_idx], 4),
                lambda: tf.image.random_contrast(image,
                            lower=CONTRAST_LOWER, upper=CONTRAST_UPPER),
                lambda: image)
        return op_idx + 1, image

    def cond(op_idx, image):
        return op_idx < 4

    # Loop through the augmentation ops
    i = tf.constant(0) # counter for augment operations
    _, image = tf.while_loop(cond, body,
            loop_vars=[i, image])

    # Augmentations may produce something outside the [0.0,1.0] range
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image
