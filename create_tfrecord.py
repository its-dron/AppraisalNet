import tensorflow as tf
from tqdm import tqdm #fast extensible progresss meter
import argparse
import os
import csv
import pdb

def createExample(image_bin, price, im_name):
    example = tf.train.Example(features=tf.train.Features(feature=
        {
        'id': tf.train.Feature(int64_list=tf.train.Int64List(value=[im_name])),
        'price': tf.train.Feature(float_list=tf.train.FloatList(value=[price])),
        'image_raw': tf.train.Feature(bytes_list=
                tf.train.BytesList(value=[image_bin]))
        }))
    return example

def main(args):
    '''
    Takes a csv file (first col image_name, second col price) and
    creates a TFRecords file including the prices and all images
    '''

    writer = tf.python_io.TFRecordWriter(args.output)
    with open(args.csv_file, 'rb') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in tqdm(csv_reader):
            try:
                im_filename = row[0]
                price = float(row[1])
                im_path = os.path.join(args.image_dir,
                        im_filename + '.' + args.image_ext)
                with open(im_path) as f:
                    image_bin = f.read()
                example = createExample(image_bin, price, int(im_filename))
                writer.write(example.SerializeToString())
            except:
                # Failed to load image
                pass
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        type=str,
        default='crawler_images',
        help='Directory containing all images.'
    )
    parser.add_argument(
        '--image_ext',
        type=str,
        default='jpg',
        help='Image Extensions.'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='db.tfrecords',
        help='TFRecords output filename.'
    )
    parser.add_argument(
        '--csv_file',
        type=str,
        default=None,
        help='''
        Path to csv file (comma delimiter).
        First column is image name (no ext).
        Second column is price.
        ''')

    args, unparsed = parser.parse_known_args()
    main(args)
