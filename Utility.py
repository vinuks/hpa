

class utility:

    def one_hot_matrix(labels, C):
        """
        Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                         corresponds to the jth training example. So if example j had a label i. Then entry (i,j) 
                         will be 1. 
                         
        Arguments:
        labels -- vector containing the labels 
        C -- number of classes, the depth of the one hot dimension
        
        Returns: 
        one_hot -- one hot matrix
        """
        
        ### START CODE HERE ###
        
        # Create a tf.constant equal to C (depth), name it 'C'. (approx. 1 line)
        C = tf.constant(C, name = 'C')
        
        # Use tf.one_hot, be careful with the axis (approx. 1 line)
        one_hot_matrix = tf.one_hot(labels, C, 1)
        
        # Create the session (approx. 1 line)
        sess = tf.Session()
        
        # Run the session (approx. 1 line)
        one_hot = sess.run(one_hot_matrix)
        
        # Close the session (approx. 1 line). See method 1 above.
        sess.close()
        
        ### END CODE HERE ###
        
        return (one_hot.T)




    def ones(shape):
        """
        Creates an array of ones of dimension shape
        
        Arguments:
        shape -- shape of the array you want to create
            
        Returns: 
        ones -- array containing only ones
        """
        
        ### START CODE HERE ###
        
        # Create "ones" tensor using tf.ones(...). (approx. 1 line)
        ones = tf.ones(shape)
        
        # Create the session (approx. 1 line)
        sess = tf.Session()
        
        # Run the session to compute 'ones' (approx. 1 line)
        ones = sess.run(ones)
        
        # Close the session (approx. 1 line). See method 1 above.
        sess.close()
        
        ### END CODE HERE ###
        return ones


    def sigmoid(z):
        """
        Computes the sigmoid of z
        
        Arguments:
        z -- input value, scalar or vector
        
        Returns: 
        results -- the sigmoid of z
        """
        
        ### START CODE HERE ### ( approx. 4 lines of code)
        # Create a placeholder for x. Name it 'x'.
        x = tf.placeholder(tf.float32, name = "x")

        # compute sigmoid(x)
        sigmoid = tf.sigmoid(x)

        # Create a session, and run it. Please use the method 2 explained above. 
        # You should use a feed_dict to pass z's value to x. 
        with tf.Session() as sess:
        
            # Run session and call the output "result"
            result = sess.run(sigmoid, feed_dict = {x : z})
        
        ### END CODE HERE ###
        
        return result