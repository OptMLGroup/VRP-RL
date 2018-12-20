import numpy as np
import tensorflow as tf
import os
import warnings
import collections

def create_TSP_dataset(
        n_problems,
        n_nodes,
        data_dir,
        seed=None,
        data_type='train'):
    '''
    This function creates TSP instances and saves them on disk. If a file is already available,
    it will load the file.
    Input:
        n_problems: number of problems to generate.
        n_nodes: number of nodes in the problem.
        data_dir: the directory to save or load the file.
        seed: random seed for generating the data.
        data_type: the purpose for generating the data. It can be 'train', 'val', or any string.
    output:
        data: a numpy array with shape [n_problems x n_nodes x 2]
     '''

    # set random number generator
    if seed == None:
        rnd = np.random
    else:
        rnd = np.random.RandomState(seed)
    
    # build task name and datafiles
    task_name = 'tsp-size-{}-len-{}-{}.txt'.format(n_problems, n_nodes,data_type)
    fname = os.path.join(data_dir, task_name)

    # cteate/load data
    if os.path.exists(fname):
        print('Loading dataset for {}...'.format(task_name))
        data = np.loadtxt(fname)
        data = data.reshape(-1, n_nodes,2)
    else:
        print('Creating dataset for {}...'.format(task_name))
        # Generate a training set of size n_problems 
        data= rnd.uniform(0,1,size=(n_problems,n_nodes,2))
        np.savetxt(fname, data.reshape(-1, n_nodes*2))                         

    return data

class DataGenerator(object):
    def __init__(self, 
                 args):

        '''
        This class generates TSP problems for training and test
        Inputs:
            args: the parameter dictionary. It should include:
                args['random_seed']: random seed
                args['test_size']: number of problems to test
                args['n_nodes': number of nodes
                args['batch_size']: batchsize for training

        '''
        self.args = args
        self.rnd = np.random.RandomState(seed= args['random_seed'])
        print('Created train iterator.')

        # create test data
        self.n_problems = args['test_size']
        self.test_data = create_TSP_dataset(self.n_problems,args['n_nodes'],'./data',
            seed = args['random_seed']+1,data_type='test')

        self.reset()

    def reset(self):
        self.count = 0

    def get_train_next(self):
        '''
        Get next batch of problems for training
        '''
        input_data = self.rnd.uniform(0,1,
            size=[self.args['batch_size'],self.args['n_nodes'],2])

        return input_data
 
    def get_test_next(self):
        '''
        Get next batch of problems for testing
        '''
        if self.count<self.args['test_size']:
            input_data = self.test_data[self.count:self.count+1]
            self.count +=1
        else:
            warnings.warn("The test iterator reset.") 
            self.count = 0
            input_data = self.test_data[self.count:self.count+1]
            self.count +=1

        return input_data

    def get_test_all(self):
        '''
        Get all test problems
        '''
        return self.test_data

class State(collections.namedtuple("State",
                                        ("mask"))):
    pass
class Env(object):
    def __init__(self, 
                 args):
        '''
        This is the environment for TSP.
        Inputs: 
            args: the parameter dictionary. It should include:
                args['n_nodes']: number of nodes in TSP
                args['input_dim']: dimension of the problem which is 2
        '''

        self.n_nodes = args['n_nodes']
        self.input_dim = args['input_dim']
        self.input_data = tf.placeholder(tf.float32,\
            shape=[None,self.n_nodes,args['input_dim']])
        self.input_pnt = self.input_data
        self.batch_size = tf.shape(self.input_data)[0] 

    def reset(self,beam_width=1):
        '''
        Resets the environment. This environment might be used with different decoders. 
        In case of using with beam-search decoder, we need to have to increase the rows of 
        the mask by a factor of beam_width.
        '''
        self.beam_width = beam_width
        
        self.input_pnt = self.input_data
        self.mask = tf.zeros([self.batch_size*beam_width,self.n_nodes],dtype=tf.float32)

        state = State(mask = self.mask )

        return state

    def step(self,
             idx,
             beam_parent=None):
        '''
        Mask the nodes that can be visited in next steps.
        '''
        # if the environment is used in beam search decoder
        if beam_parent is not None:
            # BatchBeamSeq: [batch_size*beam_width x 1]
            # [0,1,2,3,...,127,0,1,...],
            batchBeamSeq = tf.expand_dims(tf.tile(tf.cast(tf.range(self.batch_size), tf.int64),
                                                 [self.beam_width]),1)
            # batchedBeamIdx:[batch_size*beam_width]
            batchedBeamIdx= batchBeamSeq + tf.cast(self.batch_size,tf.int64)*beam_parent
 
            #MASK:[batch_size*beam_width x sourceL]
            self.mask = tf.gather_nd(self.mask,batchedBeamIdx)

        self.mask = self.mask + tf.one_hot(tf.squeeze(idx,1),self.n_nodes)

        state = State(mask = self.mask )

        return state


def reward_func(sample_solution=None):
    """The reward for the TSP task is defined as the 
    negative value of the route length. This function gets the decoded
    actions and computed the reward.

    Args:
        sample_solution : a list of tensors with len decode_len 
            each having a shape [batch_size x input_dim]

    Returns:
        rewards: tensor of size [batch_size]

    Example:
        sample_solution = [[[1,1],[2,2]],[[3,3],[4,4]],[[5,5],[6,6]]]
        decode_len = 3
        batch_size = 2
        input_dim = 2
        sample_solution_tilted[ [[5,5]
                                                    #  [6,6]]
                                                    # [[1,1]
                                                    #  [2,2]]
                                                    # [[3,3]
                                                    #  [4,4]] ]
    """

    # make sample_solution of shape [sourceL x batch_size x input_dim]
    sample_solution = tf.stack(sample_solution,0)

    sample_solution_tilted = tf.concat((tf.expand_dims(sample_solution[-1],0),
         sample_solution[:-1]),0)
    # get the reward based on the route lengths


    route_lens_decoded = tf.reduce_sum(tf.pow(tf.reduce_sum(tf.pow(\
        (sample_solution_tilted - sample_solution) ,2), 2) , .5), 0)
    return route_lens_decoded 