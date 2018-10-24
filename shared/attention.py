import tensorflow as tf

class Attention(object):
    """A generic attention module for a decoder in seq2seq models"""
    def __init__(self, dim, use_tanh=False, C=10,_name='Attention',_scope=''):
        self.use_tanh = use_tanh
        self._scope = _scope

        with tf.variable_scope(_scope+_name):
            # self.v: is a variable with shape [1 x dim]
            self.v = tf.get_variable('v',[1,dim],
                       initializer=tf.contrib.layers.xavier_initializer())
            self.v = tf.expand_dims(self.v,2)
        self.project_query = tf.layers.Dense(dim,_scope=_scope+_name +'/dense')
        self.project_ref = tf.layers.Conv1D(dim,1,_scope=_scope+_name +'/conv1d')
        self.C = C  # tanh exploration parameter
        self.tanh = tf.nn.tanh

    def __call__(self, query, ref, *args, **kwargs):
        """
        This function gets a query tensor and ref rensor and returns the logit op.
        Args: 
            query: is the hidden state of the decoder at the current
                time step. [batch_size x dim]
            ref: the set of hidden states from the encoder. 
                [batch_size x max_time x dim]

        Returns:
            e: convolved ref with shape [batch_size x max_time x dim]
            logits: [batch_size x max_time]
        """
        # expanded_q,e: [batch_size x max_time x dim]
        e = self.project_ref(ref)
        q = self.project_query(query) #[batch_size x dim]
        expanded_q = tf.tile(tf.expand_dims(q,1),[1,tf.shape(e)[1],1])

        # v_view:[batch_size x dim x 1]
        v_view = tf.tile( self.v, [tf.shape(e)[0],1,1]) 
        
        # u : [batch_size x max_time x dim] * [batch_size x dim x 1] = 
        #       [batch_size x max_time]
        u = tf.squeeze(tf.matmul(self.tanh(expanded_q + e), v_view),2)

        if self.use_tanh:
            logits = self.C * self.tanh(u)
        else:
            logits = u  

        return e, logits

if __name__ == "__main__":
    sess = tf.InteractiveSession()
    tf.set_random_seed(100)
    q = tf.random_uniform([2,128])
    ref = tf.random_uniform([2,10,128])
    attention = Attention(128,use_tanh=True, C=10)
    e, logits = attention(q,ref)
    sess.run(tf.global_variables_initializer())
    print(sess.run([logits, tf.nn.softmax(logits)]))
