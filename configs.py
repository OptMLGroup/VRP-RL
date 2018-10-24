import argparse
import shared.misc_utils as utils 
import os
from task_specific_params import task_lst

def str2bool(v):
    return v.lower() in ('true', '1')

def initialize_task_settings(args,task):

    try:
        task_params = task_lst[task]
    except:
        raise Exception('Task is not implemented.') 

    for name, value in task_params._asdict().items():
    	args[name] = value


    # args['task_name'] = task_params.task_name
    # args['input_dim'] = task_params.input_dim
    # args['n_nodes'] = task_params.n_nodes
    # if args['decode_len'] == None:
    #     args['decode_len'] = task_params.decode_len

    return args

def ParseParams():
    parser = argparse.ArgumentParser(description="Neural Combinatorial Optimization with RL")

    # Data
    parser.add_argument('--task', default='vrp10', help="Select the task to solve; i.e. tsp10")
    parser.add_argument('--batch_size', default=128,type=int, help='Batch size in training')
    parser.add_argument('--n_train', default=260000,type=int, help='Number of training steps')
    parser.add_argument('--test_size', default=1000,type=int, help='Number of problems in test set')

    # Network
    parser.add_argument('--agent_type', default='attention', help="attention|pointer")
    parser.add_argument('--forget_bias', default=1.0,type=float, help="Forget bias for BasicLSTMCell.")
    parser.add_argument('--embedding_dim', default=128,type=int, help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', default=128,type=int, help='Dimension of hidden layers in Enc/Dec')
    parser.add_argument('--n_process_blocks', default=3,type=int,                     
                        help='Number of process block iters to run in the Critic network')
    parser.add_argument('--rnn_layers', default=1, type=int, help='Number of LSTM layers in the encoder and decoder')
    parser.add_argument('--decode_len', default=None,type=int,                     
                        help='Number of time steps the decoder runs before stopping')
    parser.add_argument('--n_glimpses', default=0, type=int, help='Number of glimpses to use in the attention')
    parser.add_argument('--tanh_exploration', default=10.,  type=float,                   
             help='Hyperparam controlling exploration in the net by scaling the tanh in the softmax')
    parser.add_argument('--use_tanh', type=str2bool, default=False, help='')
    parser.add_argument('--mask_glimpses', type=str2bool, default=True, help='')
    parser.add_argument('--mask_pointer', type=str2bool, default=True, help='')
    parser.add_argument('--dropout', default=0.1, type=float, help='The dropout prob')

    # Training
    parser.add_argument('--is_train', default=True,type=str2bool, help="whether to do the training or not")
    parser.add_argument('--actor_net_lr', default=1e-4,type=float, help="Set the learning rate for the actor network")
    parser.add_argument('--critic_net_lr', default=1e-4,type=float, help="Set the learning rate for the critic network")
    parser.add_argument('--random_seed', default=24601,type=int, help='')
    parser.add_argument('--max_grad_norm', default=2.0, type=float, help='Gradient clipping')
    parser.add_argument('--entropy_coeff', default=0.0, type=float, help='coefficient for entropy regularization')
    # parser.add_argument('--loss_type', type=int, default=1, help='1,2,3')

    # inference
    parser.add_argument('--infer_type', default='batch', 
        help='single|batch: do inference for the problems one-by-one, or run it all at once')
    parser.add_argument('--beam_width', default=10, type=int, help='')

    # Misc
    parser.add_argument('--stdout_print', default=True, type=str2bool, help='print control')
    parser.add_argument("--gpu", default='3', type=str,help="gpu number.")
    parser.add_argument('--log_interval', default=200,type=int, help='Log info every log_step steps')
    parser.add_argument('--test_interval', default=200,type=int, help='test every test_interval steps')
    parser.add_argument('--save_interval', default=10000,type=int, help='save every save_interval steps')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--model_dir', type=str, default='')
    parser.add_argument('--load_path', type=str, default='', help='Path to load trained variables')
    parser.add_argument('--disable_tqdm', default=True, type=str2bool)
                        
    args, unknown = parser.parse_known_args()
    args = vars(args)

    args['log_dir'] = "{}/{}-{}".format(args['log_dir'],args['task'], utils.get_time())
    if args['model_dir'] =='':
        args['model_dir'] = os.path.join(args['log_dir'],'model')

    # file to write the stdout
    try:
        os.makedirs(args['log_dir'])
        os.makedirs(args['model_dir'])
    except:
        pass

    # create a print handler
    out_file = open(os.path.join(args['log_dir'], 'results.txt'),'w+') 
    prt = utils.printOut(out_file,args['stdout_print'])

    os.environ["CUDA_VISIBLE_DEVICES"]=  args['gpu'] 

    args = initialize_task_settings(args,args['task'])

    # print the run args
    for key, value in sorted(args.items()):
        prt.print_out("{}: {}".format(key,value))

    return args, prt

