import tensorflow as tf
import numpy as np



def bucket_tensor(tensor,bucket_size):
  
  if not bucket_size:
    return tensor
  
  tensor = tf.reshape(tensor,[-1])
  size = tensor.get_shape().as_list()[-1]
  mul,rest = divmod(size,bucket_size)
  fill_value = tensor[-1]
  
  if mul != 0 and rest != 0:
    print("Fill : {}-{}".format(mul,rest))
    to_add = tf.ones([bucket_size-rest]) * fill_value
    tensor = tf.concat([tensor,to_add], axis =  0)
    
  if mul == 0:
    print("Original size : {}-{}".format(mul,rest))
    tensor = tf.reshape(tensor, [1,size])
    
  else:
    print("To bucket size : {}-{}".format(mul,rest))
    tensor = tf.reshape(tensor,[-1, bucket_size])
      
  return tensor

if __name__ == "__main__":

  raw_shape = input("Shape : ")
  raw_shape_split = raw_shape.split(',')
  shape = [int(x) for x in raw_shape_split]  
  bucket_size = 256
  tensor = tf.placeholder(tf.float32, shape)
  buck_tensor = bucket_tensor(tensor, bucket_size)
	
  with tf.Session() as sess:
    my_buck = sess.run(buck_tensor, feed_dict = {tensor : np.random.rand(*shape)})
    print(my_buck.shape)
  
    	
		
