import tensorflow as tf

class AttentionVRPActor(object):
    """A generic attention module for the attention in vrp model"""
    def __init__(self, dim, use_tanh=False, C=10,_name='Attention',_scope=''):
        self.use_tanh = use_tanh
        self._scope = _scope

        with tf.compat.v1.variable_scope(_scope+_name):
            # self.v: is a variable with shape [1 x dim]
            self.v = tf.compat.v1.get_variable('v',[1,dim],
                       initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
            self.v = tf.expand_dims(self.v,2)
            
        self.emb_d = tf.compat.v1.layers.Conv1D(dim,1,_scope=_scope+_name+'/emb_d' ) #conv1d
        self.emb_ld = tf.compat.v1.layers.Conv1D(dim,1,_scope=_scope+_name+'/emb_ld' ) #conv1d_2

        self.project_d = tf.compat.v1.layers.Conv1D(dim,1,_scope=_scope+_name+'/proj_d' ) #conv1d_1
        self.project_ld = tf.compat.v1.layers.Conv1D(dim,1,_scope=_scope+_name+'/proj_ld' ) #conv1d_3
        self.project_query = tf.compat.v1.layers.Dense(dim,_scope=_scope+_name+'/proj_q' ) #
        self.project_ref = tf.compat.v1.layers.Conv1D(dim,1,_scope=_scope+_name+'/proj_ref' ) #conv1d_4


        self.C = C  # tanh exploration parameter
        self.tanh = tf.nn.tanh

    def __call__(self, query, ref, env):
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
        # get the current demand and load values from environment
        demand = env.demand
        load = env.load
        max_time = tf.shape(input=demand)[1]

        # embed demand and project it
        # emb_d:[batch_size x max_time x dim ]
        emb_d = self.emb_d(tf.expand_dims(demand,2))
        # d:[batch_size x max_time x dim ]
        d = self.project_d(emb_d)

        # embed load - demand
        # emb_ld:[batch_size*beam_width x max_time x hidden_dim]
        emb_ld = self.emb_ld(tf.expand_dims(tf.tile(tf.expand_dims(load,1),[1,max_time])-
                                              demand,2))
        # ld:[batch_size*beam_width x hidden_dim x max_time ] 
        ld = self.project_ld(emb_ld)

        # expanded_q,e: [batch_size x max_time x dim]
        e = self.project_ref(ref)
        q = self.project_query(query) #[batch_size x dim]
        expanded_q = tf.tile(tf.expand_dims(q,1),[1,max_time,1])

        # v_view:[batch_size x dim x 1]
        v_view = tf.tile( self.v, [tf.shape(input=e)[0],1,1]) 
        
        # u : [batch_size x max_time x dim] * [batch_size x dim x 1] = 
        #       [batch_size x max_time]
        u = tf.squeeze(tf.matmul(self.tanh(expanded_q + e + d + ld), v_view),2)

        if self.use_tanh:
            logits = self.C * self.tanh(u)
        else:
            logits = u  

        return e, logits


class AttentionVRPCritic(object):
    """A generic attention module for the attention in vrp model"""
    def __init__(self, dim, use_tanh=False, C=10,_name='Attention',_scope=''):

        self.use_tanh = use_tanh
        self._scope = _scope

        with tf.compat.v1.variable_scope(_scope+_name):
            # self.v: is a variable with shape [1 x dim]
            self.v = tf.compat.v1.get_variable('v',[1,dim],
                       initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
            self.v = tf.expand_dims(self.v,2)
            
        self.emb_d = tf.compat.v1.layers.Conv1D(dim,1,_scope=_scope+_name +'/emb_d') #conv1d
        self.project_d = tf.compat.v1.layers.Conv1D(dim,1,_scope=_scope+_name +'/proj_d') #conv1d_1
        
        self.project_query = tf.compat.v1.layers.Dense(dim,_scope=_scope+_name +'/proj_q') #
        self.project_ref = tf.compat.v1.layers.Conv1D(dim,1,_scope=_scope+_name +'/proj_e') #conv1d_2

        self.C = C  # tanh exploration parameter
        self.tanh = tf.nn.tanh
        
    def __call__(self, query, ref, env):
        """
        This function gets a query tensor and ref rensor and returns the logit op.
        Args: 
            query: is the hidden state of the decoder at the current
                time step. [batch_size x dim]
            ref: the set of hidden states from the encoder. 
                [batch_size x max_time x dim]

            env: keeps demand ond load values and help decoding. Also it includes mask.
                env.mask: a matrix used for masking the logits and glimpses. It is with shape
                         [batch_size x max_time]. Zeros in this matrix means not-masked nodes. Any 
                         positive number in this mask means that the node cannot be selected as next 
                         decision point.
                env.demands: a list of demands which changes over time.

        Returns:
            e: convolved ref with shape [batch_size x max_time x dim]
            logits: [batch_size x max_time]
        """
        # we need the first demand value for the critic
        demand = env.input_data[:,:,-1]
        max_time = tf.shape(input=demand)[1]

        # embed demand and project it
        # emb_d:[batch_size x max_time x dim ]
        emb_d = self.emb_d(tf.expand_dims(demand,2))
        # d:[batch_size x max_time x dim ]
        d = self.project_d(emb_d)


        # expanded_q,e: [batch_size x max_time x dim]
        e = self.project_ref(ref)
        q = self.project_query(query) #[batch_size x dim]
        expanded_q = tf.tile(tf.expand_dims(q,1),[1,max_time,1])

        # v_view:[batch_size x dim x 1]
        v_view = tf.tile( self.v, [tf.shape(input=e)[0],1,1]) 
        
        # u : [batch_size x max_time x dim] * [batch_size x dim x 1] = 
        #       [batch_size x max_time]
        u = tf.squeeze(tf.matmul(self.tanh(expanded_q + e + d), v_view),2)

        if self.use_tanh:
            logits = self.C * self.tanh(u)
        else:
            logits = u  

        return e, logits