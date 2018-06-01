import tensorflow as tf
from main import config2params
from utils.quantization import quant_conv_sequence


conf_file = './configs/local_test/wav2letter_v1.config'
env_param,params = config2params(conf_file)

imgs = tf.placeholder(tf.float32, [64,650,39])

pre_out,quant_weights,orig_weights = quant_conv_sequence(inputs = imgs, conv_type = 'conv',
                            filters = params.get('filters'),
                            widths = params.get('widths'),
                            strides = params.get('strides'),
                            activation = params.get('activation'),
                            data_format = params.get('data_format'),
                            dropouts = params.get('dropouts'),
                            batchnorm = params.get('bn'),
                            train = False,
							num_bits = 4,
							bucket_size = 256,
							stochastic = False)
