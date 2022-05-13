import argparse
from utee import misc, quant, selector
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark =True
from collections import OrderedDict

def main():
    # add parameters 
    parser = argparse.ArgumentParser(description='PyTorch SVHN Example')
    parser.add_argument('--type', default='cifar10', help='|'.join(selector.known_models))
    parser.add_argument('--quant_method', default='linear', help='linear|minmax|log|tanh')
    parser.add_argument('--batch_size', type=int, default=100, help='input batch size for training (default: 64)')
    parser.add_argument('--gpu', default=None, help='index of gpus to use')
    parser.add_argument('--ngpu', type=int, default=8, help='number of gpus to use')
    parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
    parser.add_argument('--model_root', default='~/.torch/models/', help='folder to save the model')
    parser.add_argument('--data_root', default='/data/public_dataset/pytorch/', help='folder to save the model')
    parser.add_argument('--logdir', default='log/default', help='folder to save to the log')

    parser.add_argument('--input_size', type=int, default=224, help='input size of image')
    parser.add_argument('--n_sample', type=int, default=20, help='number of samples to infer the scaling factor')
    parser.add_argument('--param_bits', type=int, default=8, help='bit-width for parameters')
    parser.add_argument('--bn_bits', type=int, default=32, help='bit-width for running mean and std')
    parser.add_argument('--fwd_bits', type=int, default=8, help='bit-width for layer output')
    parser.add_argument('--overflow_rate', type=float, default=0.0, help='overflow rate')
    args = parser.parse_args()

    args.gpu = misc.auto_select_gpu(utility_bound=0, num_gpu=args.ngpu, selected_gpus=args.gpu)
    args.ngpu = len(args.gpu)
    misc.ensure_dir(args.logdir)
    args.model_root = misc.expand_user(args.model_root)
    args.data_root = misc.expand_user(args.data_root)
    args.input_size = 299 if 'inception' in args.type else args.input_size
    assert args.quant_method in ['linear', 'minmax', 'log', 'tanh']
    print("=================FLAGS==================")
    for k, v in args.__dict__.items():
        print('{}: {}'.format(k, v)) # print all args info 
    print("========================================")

    assert torch.cuda.is_available(), 'no cuda'
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # load model and dataset fetcher
    model_raw, ds_fetcher, is_imagenet = selector.select(args.type, model_root=args.model_root) # output model info
    # print('selector.select model_raw: ', model_raw)
    # print('selector.select ds_fetcher: ', ds_fetcher)
    args.ngpu = args.ngpu if is_imagenet else 1

    print('load into ... selector ....')
    for param_state in model_raw.state_dict():
        print(param_state , '\t', model_raw.state_dict()[param_state].size())
        
# features.0.weight        torch.Size([128, 3, 3, 3])
# features.0.bias          torch.Size([128])
# features.1.running_mean          torch.Size([128])
# features.1.running_var   torch.Size([128])
# features.1.num_batches_tracked   torch.Size([])
    # quantize parameters
    if args.param_bits < 32:
        state_dict = model_raw.state_dict()
        # print('state_dict: ', state_dict)
        state_dict_quant = OrderedDict()
        # sf_dict = OrderedDict()
        for k, v in state_dict.items():
            # print('k: ', k, ' v: tensor([...])') # k:  features.0.weight  v: tensor([...])
            if 'running' in k: # represent computation of mean and var in BN
                if args.bn_bits >=32:
                    print("Ignoring {}".format(k))
                    state_dict_quant[k] = v # take value into key
                    continue
                else:
                    bits = args.bn_bits
            else:
                bits = args.param_bits # bit-width for parameters for weight, bias

            if args.quant_method == 'linear':
                sf = bits - 1. - quant.compute_integral_part(v, overflow_rate=args.overflow_rate)
                v_quant  = quant.linear_quantize(v, sf, bits=bits)
            elif args.quant_method == 'log':
                v_quant = quant.log_minmax_quantize(v, bits=bits)
            elif args.quant_method == 'minmax':
                v_quant = quant.min_max_quantize(v, bits=bits)
            else:
                v_quant = quant.tanh_quantize(v, bits=bits)
            # print('k: ', k, ' v: ', v_quant)
            state_dict_quant[k] = v_quant
            print(k, bits) # print each layer and applied bits
        model_raw.load_state_dict(state_dict_quant) # take pretrained model params into model_raw

    # quantize forward activation
    if args.fwd_bits < 32:
        # print('before duplicate model with quantize......')
        # for k,v in model_raw._modules.items():
        #     print('k: ', k, ' v: ', v)
            
        model_raw = quant.duplicate_model_with_quant(model_raw, bits=args.fwd_bits, overflow_rate=args.overflow_rate,
                                                     counter=args.n_sample, type=args.quant_method)
        print('after duplicate model ', model_raw)
        val_ds_tmp = ds_fetcher(10, data_root=args.data_root, train=False, input_size=args.input_size) # function pont to ?
        misc.eval_model(model_raw, val_ds_tmp, ngpu=1, n_sample=args.n_sample, is_imagenet=is_imagenet)

    # eval model
    val_ds = ds_fetcher(args.batch_size, data_root=args.data_root, train=False, input_size=args.input_size)
    acc1, acc5 = misc.eval_model(model_raw, val_ds, ngpu=args.ngpu, is_imagenet=is_imagenet)

    # print sf
    print(model_raw)
    res_str = "type={}, quant_method={}, param_bits={}, bn_bits={}, fwd_bits={}, overflow_rate={}, acc1={:.4f}, acc5={:.4f}".format(
        args.type, args.quant_method, args.param_bits, args.bn_bits, args.fwd_bits, args.overflow_rate, acc1, acc5)
    print(res_str)
    with open('acc1_acc5.txt', 'a') as f:
        f.write(res_str + '\n')


if __name__ == '__main__':
    main()
