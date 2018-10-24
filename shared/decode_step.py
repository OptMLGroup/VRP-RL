import tensorflow as tf

class DecodeStep(object):
    '''
    Base class of the decoding (without RNN)
    '''
    def __init__(self, 
            ClAttention,
            hidden_dim,
            use_tanh=False,
            tanh_exploration=10.,
            n_glimpses=0,
            mask_glimpses=True,
            mask_pointer=True,
            _scope=''):
        '''
        This class does one-step of decoding.
        Inputs:
            ClAttention:    the class which is used for attention
            hidden_dim:     hidden dimension of RNN
            use_tanh:       whether to use tanh exploration or not
            tanh_exploration: parameter for tanh exploration
            n_glimpses:     number of glimpses
            mask_glimpses:  whether to use masking for the glimpses or not
            mask_pointer:   whether to use masking for the glimpses or not
            _scope:         variable scope
        '''

        self.hidden_dim = hidden_dim
        self.use_tanh = use_tanh
        self.tanh_exploration = tanh_exploration
        self.n_glimpses = n_glimpses
        self.mask_glimpses = mask_glimpses
        self.mask_pointer = mask_pointer
        self._scope = _scope
        self.BIGNUMBER = 100000.


        # create glimpse and attention instances as well as tf.variables.
        ## create a list of class instances
        self.glimpses = [None for _ in range(self.n_glimpses)]
        for i in range(self.n_glimpses):
            self.glimpses[i] = ClAttention(hidden_dim, 
                use_tanh=False,
                _scope=self._scope,
                _name="Glimpse"+str(i))
            
        # build TF variables required for pointer
        self.pointer = ClAttention(hidden_dim, 
            use_tanh=use_tanh, 
            C=tanh_exploration,
            _scope=self._scope,
            _name="Decoder/Attention")

    def get_logit_op(self,
                     decoder_inp,
                     context,
                     Env,
                    *args,
                    **kwargs):
        """
        For a given input to deocoder, returns the logit op.
        Input:
            decoder_inp: it is the input problem with dimensions [batch_size x dim].
                        Usually, it is the embedded problem with dim = embedding_dim.
            context: the context vetor from the encoder. It is usually the output of rnn with
                      shape [batch_size x max_time x dim]
            Env: an instance of the environment. It should have:
                Env.mask: a matrix used for masking the logits and glimpses. It is with shape
                         [batch_size x max_time]. Zeros in this matrix means not-masked nodes. Any 
                         positive number in this mask means that the node cannot be selected as 
                         the next decision point.
        Returns:
            logit: the logits which will used by decoder for producing a solution. It has shape
            [batch_size x max_time].
        """

        # glimpses
        for i in range(self.n_glimpses):
            # ref: [batch_size x max_time x hidden_dim], logit : [batch_size x max_time]
            ref, logit = self.glimpses[i](decoder_inp, context,Env)
            if self.mask_glimpses:
                logit -= self.BIGNUMBER* Env.mask
            # prob: [batch_size x max_time
            prob = tf.nn.softmax(logit)
            # decoder_inp : [batch_size x 1 x max_time ] * [batch_size x max_time x hidden_dim] -> 
            #[batch_size x hidden_dim ]
            decoder_inp = tf.squeeze(tf.matmul( tf.expand_dims(prob,1),ref) ,1)

        # attention
        _, logit = self.pointer(decoder_inp,context,Env)
        if self.mask_pointer:
            logit -= self.BIGNUMBER* Env.mask

        return logit , None

    def step(self,
            decoder_inp,
            context,
            Env,
            decoder_state=None,
            *args,
            **kwargs):
        '''
        get logits and probs at a given decoding step.
        Inputs:
            decoder_input: Input of the decoding step with shape [batch_size x embedding_dim]
            context: context vector to use in attention
            Env: an instance of the environment
            decoder_state: The state of the LSTM cell. It can be None when we use a decoder without 
                LSTM cell.
        Returns:
            logit: logits with shape [batch_size x max_time]
            prob: probabilities for the next location visit with shape of [batch_size x max_time]
            logprob: log of probabilities
            decoder_state: updated state of the LSTM cell
        '''

        logit, decoder_state = self.get_logit_op(
                     decoder_inp,
                     context,
                     Env, 
                     decoder_state)

        logprob = tf.nn.log_softmax(logit)
        prob = tf.exp(logprob)

        return logit, prob, logprob, decoder_state

class RNNDecodeStep(DecodeStep):
    '''
    Decodes the sequence. It keeps the decoding history in a RNN.
    '''
    def __init__(self, 
            ClAttention,
            hidden_dim,
            use_tanh=False,
            tanh_exploration=10.,
            n_glimpses=0,
            mask_glimpses=True,
            mask_pointer=True,
            forget_bias=1.0,
            rnn_layers=1,
            _scope=''):

        '''
        This class does one-step of decoding which uses RNN for storing the sequence info.
        Inputs:
            ClAttention:    the class which is used for attention
            hidden_dim:     hidden dimension of RNN
            use_tanh:       whether to use tanh exploration or not
            tanh_exploration: parameter for tanh exploration
            n_glimpses:     number of glimpses
            mask_glimpses:  whether to use masking for the glimpses or not
            mask_pointer:   whether to use masking for the glimpses or not
            forget_bias:    forget bias of LSTM
            rnn_layers:     number of LSTM layers
            _scope:         variable scope

        '''

        super(RNNDecodeStep,self).__init__(ClAttention,
                                        hidden_dim,
                                        use_tanh=use_tanh,
                                        tanh_exploration=tanh_exploration,
                                        n_glimpses=n_glimpses,
                                        mask_glimpses=mask_glimpses,
                                        mask_pointer=mask_pointer,
                                        _scope=_scope)
        self.forget_bias = forget_bias
        self.rnn_layers = rnn_layers     
#         self.dropout = tf.placeholder(tf.float32,name='decoder_rnn_dropout')

        # build a multilayer LSTM cell
        single_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim, 
            forget_bias=forget_bias)
        self.dropout = tf.placeholder(tf.float32,name='decoder_rnn_dropout') 
        single_cell = tf.contrib.rnn.DropoutWrapper(
                cell=single_cell, input_keep_prob=(1.0 - self.dropout))
        self.cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * rnn_layers)

    def get_logit_op(self,
                    decoder_inp,
                    context,
                    Env,
                    decoder_state,
                    *args,
                    **kwargs):
        """
        For a given input to decoder, returns the logit op and new decoder_state.
        Input:
            decoder_inp: it is the input problem with dimensions [batch_size x dim].
                        Usually, it is the embedded problem with dim = embedding_dim.
            context: the context vetor from the encoder. It is usually the output of rnn with
                      shape [batch_size x max_time x dim]
            Env: an instance of the environment. It should have:
                Env.mask: a matrix used for masking the logits and glimpses. It is with shape
                         [batch_size x max_time]. Zeros in this matrix means not-masked nodes. Any 
                         positive number in this mask means that the node cannot be selected as 
                         the next decision point.
            decoder_state: The state as a list of size rnn_layers, and each element is a
                    LSTMStateTuples with  x 2 tensors with dimension of [batch_size x hidden_dim].
                    The first one corresponds to c and the second one is h.
        Returns:
            logit: the logits which will used by decoder for producing a solution. It has shape
                    [batch_size x max_time].
            decoder_state: the update decoder state.
        """

#         decoder_inp = tf.reshape(decoder_inp,[-1,1,self.hidden_dim])
        _ , decoder_state = tf.nn.dynamic_rnn(self.cell,
                                              decoder_inp,
                                              initial_state=decoder_state,
                                              scope=self._scope+'Decoder/LSTM/rnn')
        hy = decoder_state[-1].h

        # glimpses
        for i in range(self.n_glimpses):
            # ref: [batch_size x max_time x hidden_dim], logit : [batch_size x max_time]
            ref, logit = self.glimpses[i](hy,context,Env)
            if self.mask_glimpses:
                logit -= self.BIGNUMBER* Env.mask
            prob = tf.nn.softmax(logit)
            
            # hy : [batch_size x 1 x max_time ] * [batch_size x max_time x hidden_dim] -> 
            #[batch_size x hidden_dim ]
            hy = tf.squeeze(tf.matmul( tf.expand_dims(prob,1),ref) ,1)

        # attention
        _, logit = self.pointer(hy,context,Env)
        if self.mask_pointer:
            logit -= self.BIGNUMBER* Env.mask
    
        return logit , decoder_state