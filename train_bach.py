import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import utils
from tqdm import tqdm
import os
import argparse
import time
import random
import math
import networks

parser = argparse.ArgumentParser()
parser.add_argument('--max_epochs', type=int, default=300)
parser.add_argument('--dataset', choices=['bach'], default='bach', help='dataset')
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--model', choices=['finetune'], default='finetune')
parser.add_argument('--mode', choices=['train', 'trainval', 'test', 'val'], default='train')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--wd', type=float, default=1e-3)
parser.add_argument('--load_epoch', type=int, default=-1)
parser.add_argument('--save_epoch', type=int, default=25)
parser.add_argument('--tqdm_off', action='store_true', default=False)
parser.add_argument('--source_dir', default='', type=str)
parser.add_argument('--target_dir', default='', type=str)
parser.add_argument('--lr_steps', default=[100, 200], nargs='+', type=int)

args = parser.parse_args()


save_path = 'results/%s' % (args.dataset)
save_path = save_path + '/' + args.model


if not os.path.exists(save_path):
    os.makedirs(save_path)

if not os.path.exists('results/prediction'):
    os.makedirs('results/prediction')

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.tqdm_off:
    def nop(it, *a, **k):
        return it
    tqdm = nop

dataset = utils.__dict__['Bach_dataset']

def save_checkpoint():

    checkpoint = {
        'model': model.state_dict(),
        'opt': opt.state_dict(),
    }
    torch.save(checkpoint, '%s/checkpoint_%d_%d.pth' % (save_path, args.seed, epoch))

def save_best_checkpoint():
   
    checkpoint = {
        'model': model.state_dict(),
        'opt': opt.state_dict(),
    }
   
    torch.save(checkpoint, '%s/checkpoint_best_%d.pth' % (save_path, args.seed))   

def load_checkpoint(load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model'])
    opt.load_state_dict(checkpoint['opt'])

def compute_acc(class_out, targets):
    preds = torch.max(class_out, 1)[1]
    pos = 0; 
    for ix in range(preds.size(0)):
        if preds[ix] == targets[ix]:
            pos = pos + 1
    accuracy = pos / preds.size(0) * 100

    return accuracy


def train_model():
    model.train()
    avg_loss = 0
    avg_acc = 0
    curr_acc = 0
    count = 0
    for _, (data, target) in enumerate(tqdm(data_loader)):
        opt.zero_grad()

        data, target  = data.cuda(), target.long().cuda()
        out = model(data)
        
        loss = ent_loss(out, target)
        loss.backward()
        opt.step()
        curr_acc = compute_acc(out.data, target.data)
        avg_loss = avg_loss + loss.item()
        avg_acc = avg_acc + curr_acc
        count = count + 1
    avg_loss = avg_loss / count
    avg_acc = avg_acc / count
    print('Epoch: %d; Loss: %f; Acc: %.2f%% ' % (epoch, avg_loss, avg_acc))

    return avg_loss

softmax = torch.nn.Softmax(-1)

def eval_model():
    print('Validation')
    model.eval()
 
    pos=0; total=0;
    prediction_list = []
    groundtruth_list = []
    avg_loss = 0
    count = 0 
    
    for _, (data, target) in enumerate(tqdm(eval_data_loader)):
        #bs,ncrops, c, h,w = data.size()
        #data = data.view(-1, c, h, w)
       
        data, target  = data.cuda(), target.long().cuda()
        with torch.no_grad():
            #print(data[0].mean())
            out = model(data)
            #out = out.view(bs, ncrops, -1).mean(1)
            #out = out.view(bs, ncrops, -1)
            #print(out.size())
            #loss = ent_loss(out, target)
           
            #print(out, out.size(), out.dim())
            pred = torch.max(out, out.dim() - 1)[1]
            pos += torch.eq(pred.cpu().long(), target.data.cpu().long()).sum().item()
            total += data.size(0)
               
            #avg_loss += loss.item()
            count += 1
    acc = pos * 1.0 / total * 100
    print('Val:  Acc: %.2f%% ' % (acc))
   
    #return loss

def final_eval():
    print('Generating test output')
    model.eval()
  
    pos=0; total=0;
    prediction_list = []
    groundtruth_list = []
    avg_loss = 0
    count = 0 

    outf = open('results/prediction/pred.csv', 'w')
    outf.write('case, class\n')
    for _, (data, name) in enumerate(tqdm(eval_data_loader)):
        #bs,ncrops, c, h,w = data.size()
        #data = data.view(-1, c, h, w)
   
        data  = data.cuda()
        with torch.no_grad():
            out = model(data)
            #out = out.view(bs, ncrops, -1).mean(1)
           
            pred = torch.max(out, out.dim() - 1)[1]
            for pix in range(len(pred)):
                case = name[pix].replace('test', '').replace('.tif', '')
                outf.write('%s, %s\n' %  (case, str(pred[pix].item())))
    
    outf.close()
    os.system('zip -FSjr results/prediction.zip results/prediction')

if 'train' in args.mode:
    data_loader = torch.utils.data.DataLoader(dataset(mode=args.mode), batch_size=args.batch_size, shuffle=True, num_workers=4)
    eval_data_loader = torch.utils.data.DataLoader(dataset(mode='val'), batch_size=args.batch_size, num_workers=4)
else:
    eval_data_loader = torch.utils.data.DataLoader(dataset(mode=args.mode), batch_size=args.batch_size, num_workers=4)

model = networks.Model_bach()

model = model.cuda()


opt = optim.Adam([{'params': model.parameters()}], lr=args.lr, weight_decay=args.wd)

model = torch.nn.DataParallel(model)

sch = torch.optim.lr_scheduler.MultiStepLR(opt, args.lr_steps, gamma=0.1)

if not os.path.exists(save_path):
    os.makedirs(save_path)

ent_loss = torch.nn.CrossEntropyLoss().cuda()
epoch = 1
if args.load_epoch != -1:
    epoch = args.load_epoch + 1
    load_checkpoint('%s/checkpoint_%d_%d.pth' % (save_path, args.seed, args.load_epoch))
    sch.step(args.load_epoch) 

if 'train' in args.mode:
    best_acc = 0
    while True:
        loss = train_model()
        print(opt.param_groups[0]['lr'])
        sch.step(epoch)

        if epoch % args.save_epoch == 0:
            eval_model()
            save_checkpoint()

        if epoch == args.max_epochs:
            break

        epoch += 1
elif 'val' in args.mode:
    load_checkpoint('%s/checkpoint_%d_%d.pth' % (save_path, args.seed, args.max_epochs))
    eval_model()
else:
    load_checkpoint('%s/checkpoint_%d_%d.pth' % (save_path, args.seed, args.max_epochs))
    final_eval()
