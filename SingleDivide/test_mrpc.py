import argparse
import os


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected in train_cifar.')


parser = argparse.ArgumentParser(description='test')
parser.add_argument(
    "--model_name_or_path",
    type=str,
    default="bert-base-uncased",
    help="Path to pretrained model or model identifier from huggingface.co/models.",
    required=True,
)
parser.add_argument('--hadamard', type=str2bool, default=False, help='apply Hadamard transformation on gradients')
parser.add_argument('--dynamic', type=str2bool, default=True, help='whether apply dynamic Hadamard transformation on gradients')
parser.add_argument('--bmm', type=str2bool, default=True, help='whether apply bmm Hadamard transformation on gradients')
parser.add_argument('--twolayers_gradweight', '--2gw', type=str2bool, default=False, help='use two 4 bit to simulate a 8 bit')
parser.add_argument('--twolayers_gradinputt', '--2gi', type=str2bool, default=False, help='use two 4 bit to simulate a 8 bit')
parser.add_argument('--weight_quant_method', '--wfq', default='ptq', type=str, metavar='strategy',
                    choices=['uniform', 'lsq', 'ptq'])
parser.add_argument('--input_quant_method', '--ifq', default='ptq', type=str, metavar='strategy',
                    choices=['uniform', 'lsq', 'ptq'])
parser.add_argument('--learnable_step_size', type=str2bool, default=True, help='Debug to draw the variance and leverage score')
parser.add_argument('--learnable_hadamard', type=str2bool, default=True, help='Debug to draw the variance and leverage score')
parser.add_argument('--lsq_layerwise_input', '--lli', type=str, default='layer',
                    help='Debug to draw the variance and leverage score', choices=['layer', 'row', 'column'])
parser.add_argument('--lsq_layerwise_weight', '--llw', type=str, default='layer',
                    help='Debug to draw the variance and leverage score', choices=['layer', 'row', 'column'])
parser.add_argument('--retain_large_value', type=str2bool, default=False,
                    help='Debug to draw the variance and leverage score')
parser.add_argument('--quantize_large_value', type=str2bool, default=False,
                    help='Debug to draw the variance and leverage score')
parser.add_argument('--draw_value', type=str2bool, default=False,
                    help='Debug to draw the variance and leverage score')
parser.add_argument('--fp16', type=str2bool, default=False,
                    help='whether use torch amp')

parser.add_argument('--luq', type=str2bool, default=False, help='use luq for backward')
parser.add_argument('--training-bit', type=str, default='', help='weight number of bits', required=True,
                    choices=['exact', 'qat', 'all8bit', 'star_weight', 'only_weight', 'weight4', 'all4bit', 'forward8',
                             'forward4', 'plt'])
parser.add_argument('--plt-bit', type=str, default='', help='')
parser.add_argument('--choice', nargs='+', type=str, required=True, help='Choose a linear layer to quantize')

parser.add_argument('--training-strategy', default='scratch', type=str, metavar='strategy',
                    choices=['scratch', 'checkpoint', 'gradually', 'checkpoint_from_zero', 'checkpoint_full_precision'])
parser.add_argument('--checkpoint_epoch_full_precision', type=int, default=0, help='full precision')
parser.add_argument('--epochs', default=3, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--task', type=str, default='mrpc', help='apply LSQ')
parser.add_argument('--seed', type=int, default=27, help='apply LSQ')
parser.add_argument("--per_device_train_batch_size", '--pdtbs', type=int, default=32, help="Batch size (per device) for the training dataloader.",)
parser.add_argument("--lr", type=float, default=2e-5, help="Initial learning rate (after the potential warmup period) to use.",)

parser.add_argument('--clip-value', type=float, default=100, help='Choose a linear layer to quantize')
parser.add_argument('--plt-debug', type=str2bool, default=False, help='Debug to draw the variance and leverage score')
parser.add_argument('--clip_lr', default=2e-5, type=float, help='Use a seperate lr for clip_vals / stepsize')
parser.add_argument('--clip_wd', default=0.0, type=float, help='weight decay for clip_vals / stepsize')

parser.add_argument("--resume_from_checkpoint", '--rfc', type=str, default=None, help="If the training should continue from a checkpoint folder.")
parser.add_argument("--checkpointing_steps", '--cs', type=str, default=None, help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",)
parser.add_argument("--pad_to_max_length", action="store_true", help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.")

parser.add_argument('--track_step_size', type=str2bool, default=False,
                    help='Debug to draw the variance and leverage score')

args = parser.parse_args()


arg = " -c quantize --qa=True --qw=True --qg=True"
if args.training_bit == 'all8bit':
    bbits, bwbits, abits, wbits = 8, 8, 8, 8
elif args.training_bit == 'exact':
    arg = ''
    bbits, bwbits, abits, wbits = 0, 0, 0, 0
elif args.training_bit == 'qat':
    bbits, bwbits, abits, wbits = 0, 0, 8, 9
    arg = "-c quantize --qa=True --qw=True --qg=False"
elif args.training_bit == 'star_weight':
    bbits, bwbits, abits, wbits = 8, 4, 4, 4
elif args.training_bit == 'only_weight':
    bbits, bwbits, abits, wbits = 8, 4, 8, 8
    arg = "-c quantize --qa=False --qw=False --qg=True"
elif args.training_bit == 'weight4':
    bbits, bwbits, abits, wbits = 8, 4, 8, 8
elif args.training_bit == 'all4bit':
    bbits, bwbits, abits, wbits = 4, 4, 4, 4
elif args.training_bit == 'forward8':
    bbits, bwbits, abits, wbits = 4, 4, 8, 8
elif args.training_bit == 'forward4':
    bbits, bwbits, abits, wbits = 8, 8, 4, 4
else:
    bbits, bwbits, abits, wbits = 0, 0, 0, 0
    if args.training_bit == 'plt':
        arg = "-c quantize --qa={} --qw={} --qg={}".format(args.plt_bit[0], args.plt_bit[1], args.plt_bit[2])
        bbits, bwbits, abits, wbits = args.plt_bit[6], args.plt_bit[5], args.plt_bit[3], args.plt_bit[4]

if args.twolayers_gradweight:
    assert int(bwbits) == 4
if args.twolayers_gradinputt:
    assert int(bbits) == 4

argchoice = ''
arg_choice_without_space = ''
for cho in args.choice:
    argchoice += cho + ' '
    arg_choice_without_space += cho + '_'
argchoice = argchoice[:-1]

resume_from_checkpoint = ''
if args.resume_from_checkpoint:
    resume_from_checkpoint = '--resume_from_checkpoint {}'\
        .format(args.resume_from_checkpoint)

checkpointing_steps = ''
if args.checkpointing_steps:
    checkpointing_steps = '--checkpointing_steps epoch'

if args.input_quant_method == "lsq" and args.weight_quant_method == "lsq":
    if args.hadamard:
        forward_method = "hadamard"
    else:
        forward_method = "lsq"
else:
    forward_method = "minimax"

if args.twolayers_gradweight and args.twolayers_gradinputt:
    backward_method = "twolayer"
elif args.luq:
    backward_method = "luq"
else:
    backward_method = "minimax"

if args.choice == ['classic']:
    choice = "classic"
elif args.choice == ['quantize']:
    choice = "quantize"

if args.fp16:
    output_dir = './resultTime/{}/{}/{}/seed={}/dynamic={}/bmm={}/fp16'\
        .format(args.model_name_or_path, choice, args.task, args.seed, args.dynamic, args.bmm)
else:
    output_dir = './resultTime/{}/{}/{}/seed={}/dynamic={}/bmm={}/fp32'\
        .format(args.model_name_or_path, choice, args.task, args.seed, args.dynamic, args.bmm)

if args.pad_to_max_length:
    pad_to_max_length = "--pad_to_max_length "
else:
    pad_to_max_length = ''

if "bert" in args.model_name_or_path:
    arch = "BertForSequenceClassification"

print("dynamic is:", args.dynamic)
print("bmm is:", args.bmm)
print("fp16 is:", args.fp16)
print("output at", output_dir)
def op_None(x):
    return "None" if x is None else x
os.system("accelerate launch test_glue.py --model_name_or_path {} --task_name {} --max_length 128 "
          "--per_device_train_batch_size {} --learning_rate {} --seed {} --num_train_epochs {} "
          "--output_dir {} --arch {} {} --choice {} "
          "--bbits {} --bwbits {} --abits {} --wbits {} "
          "--2gw {} --2gi {} --luq {} --hadamard {} --dynamic {} --bmm {} "
          "--weight_quant_method {} --input_quant_method {} "
          "--clip-value {} "
          "--clip_lr {} --clip_wd {} {} {} "
          "{} --learnable_step_size {} --learnable_hadamard {} --lsq_layerwise_input {} --lsq_layerwise_weight {} "
          "--retain_large_value {} --quantize_large_value {} --draw_value {} "
          "--track_step_size {} --fp16 {}"
          .format(args.model_name_or_path, args.task, args.per_device_train_batch_size, args.lr, args.seed, args.epochs,
                  output_dir, arch, arg, argchoice,
                  bbits, bwbits, abits, wbits,
                  args.twolayers_gradweight, args.twolayers_gradinputt, args.luq, args.hadamard, args.dynamic, args.bmm,
                  args.weight_quant_method, args.input_quant_method,
                  args.clip_value,
                  args.clip_lr, args.clip_wd, checkpointing_steps, resume_from_checkpoint,
                  pad_to_max_length, args.learnable_step_size, args.learnable_hadamard, args.lsq_layerwise_input, args.lsq_layerwise_weight,
                  args.retain_large_value, args.quantize_large_value, args.draw_value,
                  args.track_step_size, args.fp16))
