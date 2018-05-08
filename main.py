import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    # load VGG16 meta graph:
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    vgg_graph = tf.get_default_graph()

    # extract tensors:
    vgg_input_tensor = vgg_graph.get_tensor_by_name(vgg_input_tensor_name)
    vgg_keep_prob_tensor = vgg_graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out_tensor = vgg_graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out_tensor = vgg_graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out_tensor = vgg_graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer3_out_tensor, vgg_layer4_out_tensor, vgg_layer7_out_tensor
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # decoder layer1:
    decoder_layer1 = tf.layers.conv2d(
        inputs = vgg_layer7_out,
        filters = num_classes,
        kernel_size = 1,
        strides = (1, 1),
        padding = 'same',
        kernel_initializer = tf.contrib.layers.xavier_initializer(seed = 42)
    )
    decoder_layer1 = tf.layers.conv2d_transpose(
        inputs = decoder_layer1,
        filters = num_classes,
        kernel_size = 4,
        strides = (2, 2),
        padding = 'same',
        kernel_initializer = tf.contrib.layers.xavier_initializer(seed = 42)
    ) 
    # decoder layer2:
    decoder_layer2 = tf.layers.conv2d(
        inputs = vgg_layer4_out,
        filters = num_classes,
        kernel_size = 1,
        strides = (1, 1),
        padding = 'same',
        kernel_initializer = tf.contrib.layers.xavier_initializer(seed = 42)
    )
    decoder_layer2 = tf.add(
        decoder_layer1,
        decoder_layer2
    )
    decoder_layer2 = tf.layers.conv2d_transpose(
        inputs = decoder_layer2,
        filters = num_classes,
        kernel_size = 4,
        strides = (2, 2),
        padding = 'same',
        kernel_initializer = tf.contrib.layers.xavier_initializer(seed = 42)
    ) 
    # decoder layer3:
    decoder_layer3 = tf.layers.conv2d(
        inputs = vgg_layer3_out,
        filters = num_classes,
        kernel_size = 1,
        strides = (1, 1),
        padding = 'same',
        kernel_initializer = tf.contrib.layers.xavier_initializer(seed = 42)
    )
    decoder_layer3 = tf.add(
        decoder_layer2,
        decoder_layer3
    )
    decoder_layer3 = tf.layers.conv2d_transpose(
        inputs = decoder_layer3,
        filters = num_classes,
        kernel_size = 16,
        strides = (8, 8),
        padding = 'same',
        kernel_initializer = tf.contrib.layers.xavier_initializer(seed = 42)
    )

    return decoder_layer3

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    # logits:
    labels = tf.reshape(correct_label, (-1, num_classes))
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    # cross-entropy loss:
    cross_entropy_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels = labels,
            logits = logits 
        )
    )
    # optimization
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    import numpy as np

    # initialize variables:
    sess.run(tf.global_variables_initializer())


    # tensorboard visualization
    tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)

    summary_op = tf.summary.merge_all()

    # initialize logger:
    logger = tf.summary.FileWriter(
        'tensorboard/train',
        sess.graph
    )

    # training:
    for epoch in range(epochs):
        losses = []
        batch_idx = 0
        # generate mini batch:
        for X, Y in get_batches_fn(batch_size):
            _, loss, summary = sess.run(
                [train_op, cross_entropy_loss, summary_op],
                feed_dict = {
                    input_image: X,
                    correct_label: Y,
                    keep_prob: 0.9,
                    learning_rate: 1e-3
                }
            )

            batch_idx += 1

            losses.append(loss)
            # update logger:
            logger.add_summary(summary, epoch * batch_size + batch_idx)
        mean_loss = np.asarray(losses).mean()
        print("[Epoch {}]: {}".format(epoch + 1, mean_loss))

tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'model/vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'kitti/training'), image_shape)
        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer3_out_tensor, vgg_layer4_out_tensor, vgg_layer7_out_tensor = load_vgg(sess, vgg_path)
        decoder_output = layers(vgg_layer3_out_tensor, vgg_layer4_out_tensor, vgg_layer7_out_tensor, num_classes)

        H, W = image_shape
        correct_label = tf.placeholder(tf.float32, shape=[None, H, W, num_classes], name='label_groudtruth')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        logits, train_op, cross_entropy_loss = optimize(decoder_output, correct_label, learning_rate, num_classes)

        # Train NN using the train_nn function
        train_nn(
            sess, 
            8, 8, 
            get_batches_fn,
            train_op, cross_entropy_loss,
            vgg_input_tensor, correct_label,
            vgg_keep_prob_tensor, learning_rate
        )

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(
            runs_dir, data_dir, 
            sess, 
            image_shape, 
            logits, vgg_keep_prob_tensor, vgg_input_tensor
        )

        # OPTIONAL: Apply the trained model to a video

if __name__ == '__main__':
    run()
