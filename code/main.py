import argparse
import os
import os.path as osp
from copy import deepcopy
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm

from ogb.graphproppred import Evaluator

from datasets.malware_dataset import MalwareDataset 

from models.SDE-IG import SDE-IG
from utils.logger import Logger
from utils.util import args_print, set_seed

def mix_criterion(input, target, size_average=True):
    """Categorical cross-entropy with logits input and one-hot target"""
    l = -(target * torch.log(F.softmax(input, dim=1) + 1e-10)).sum(1)
    if size_average:
        l = l.mean()
    else:
        l = l.sum()
    return l

@torch.no_grad()
def eval_model(model, device, loader, evaluator, eval_metric='acc', save_pred=False, c_pred=False):
    model.eval()
    y_true = []
    y_pred = []

    for batch in loader:
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                batch.x = batch.x.float()
                batch.y  = batch.y.reshape(-1)
                pred = model(batch)
            is_labeled = batch.y == batch.y
            if eval_metric == 'acc':
                if len(batch.y.size()) == len(batch.y.size()):
                    y_true.append(batch.y.view(-1, 1).detach().cpu())
                    y_pred.append(torch.argmax(pred.detach(), dim=1).view(-1, 1).cpu())
                else:
                    y_true.append(batch.y.unsqueeze(-1).detach().cpu())
                    y_pred.append(pred.argmax(-1).unsqueeze(-1).detach().cpu())
            elif eval_metric == 'rocauc':
                pred = F.softmax(pred, dim=-1)[:, 1].unsqueeze(-1)
                if len(batch.y.size()) == len(batch.y.size()):
                    y_true.append(batch.y.view(-1, 1).detach().cpu())
                    y_pred.append(pred.detach().view(-1, 1).cpu())
                else:
                    y_true.append(batch.y.unsqueeze(-1).detach().cpu())
                    y_pred.append(pred.unsqueeze(-1).detach().cpu())
            elif eval_metric == 'mat':
                y_true.append(batch.y.unsqueeze(-1).detach().cpu())
                y_pred.append(pred.argmax(-1).unsqueeze(-1).detach().cpu())
            elif eval_metric == 'ap':
                y_true.append(batch.y.view(pred.shape).detach().cpu())
                y_pred.append(pred.detach().cpu())
            else:
                batch.y = batch.y[is_labeled]
                pred = pred[is_labeled]
                y_true.append(batch.y.view(pred.shape).unsqueeze(-1).detach().cpu())
                y_pred.append(pred.detach().unsqueeze(-1).cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    if eval_metric == 'mat':
        res_metric = matthews_corrcoef(y_true, y_pred)
    else:
        input_dict = {"y_true": y_true, "y_pred": y_pred}
        res_metric = evaluator.eval(input_dict)[eval_metric]

    if save_pred:
        return res_metric, y_pred
    else:
        return res_metric


def main():
    parser = argparse.ArgumentParser(description='Causality Inspired Invariant Graph LeArning for Malware')
    parser.add_argument('--device', default=0, type=int, help='cuda device')
    parser.add_argument('--root', default='./data', type=str, help='directory for datasets.')
    parser.add_argument('--dataset', default='APTMalware', type=str)
    
    # training config
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--epoch', default=400, type=int, help='training iterations')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate for the predictor')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--pretrain', default=20, type=int, help='pretrain epoch before early stopping')

    # model config
    parser.add_argument('--emb_dim', default=32, type=int)
    parser.add_argument('-c_dim', '--classifier_emb_dim', default=32, type=int)
    parser.add_argument('-c_in', '--classifier_input_feat', default='raw', type=str)
    parser.add_argument('--model', default='gin', type=str)
    parser.add_argument('--pooling', default='mean', type=str)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--early_stopping', default=5, type=int)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--virtual_node', action='store_true')
    parser.add_argument('--eval_metric', default='', type=str, help='specify a particular eval metric')

    # Model Config 
    parser.add_argument('--contrast_rep', default='raw', type=str)
    parser.add_argument('--contrast_pooling', default='add', type=str)
    parser.add_argument('--spurious_rep', default='raw', type=str)

    # Log/Experiment Config
    parser.add_argument('--my', default=0.0, type=float)
    parser.add_argument('--erm', default=0.0, type=float)
    parser.add_argument('--contrast', default=0.0, type=float)
    parser.add_argument('--spu_coe', default=0.0, type=float)
    parser.add_argument('--num_envs', default=1, type=int)
    parser.add_argument('--bias', default='0.5', type=str)

    # loss / learning config
    parser.add_argument('--c_pred', default=0.0, type=float, help='use casual part to predict')
    parser.add_argument('--s_pred', default=0.0, type=float, help='use spu part to predict')
    parser.add_argument('--caus', default=0.0, type=float, help='add casual manifold mixup')
    parser.add_argument('--mix', default=0.0, type=float)
    parser.add_argument('--r', default=0.7, type=float, help='selected ratio')
    parser.add_argument('--my_irm', default=0.0, type=float)
    parser.add_argument('--my_vrex', default=0.0, type=float)

    # misc
    parser.add_argument('--no_tqdm', action='store_true')
    parser.add_argument('--commit', default='', type=str, help='experiment name')
    args = parser.parse_args()
    
    # log
    datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.device <= 7:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device("cpu")
    print(f"[*] Using device: {device}")
    
    def ce_loss(a, b, reduction='mean'):
        return F.cross_entropy(a, b, reduction=reduction)

    criterion = ce_loss
    eval_metric = 'acc' if len(args.eval_metric) == 0 else args.eval_metric
    edge_dim = -1.

    # 开放对所有数据集的支持，新增 zfzmalware
    valid_datasets = ['aptmalware', 'big2015', 'dike', 'zfzmalware']
    if args.dataset.lower() in valid_datasets:
        # 处理 ZfzMalware 大小写前缀问题
        dataset_folder_prefix = 'ZfzMalware' if args.dataset.lower() == 'zfzmalware' else args.dataset
        
        # 兼容原来的根目录挂载逻辑
        if args.root != './data':
            data_root = os.path.join(args.root, f'{dataset_folder_prefix}_PyG_Data')
        else:
            data_root = f'/home/zhufengzhou/{dataset_folder_prefix}_PyG_Data'
            
        print(f"[*] Loading dataset from: {data_root}")
        
        dataset = MalwareDataset(root=data_root, dataset_name=args.dataset)
        split_idx = dataset.get_idx_split()

        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False)

        input_dim = 768  # CLAP embedding 维度
        num_classes = dataset.num_classes
        evaluator = Evaluator('ogbg-ppa') # 复用 OGB 的评测器计算 ACC
        eval_metric = 'acc'
    else:
        raise Exception(f"Unsupported dataset: {args.dataset}. This script only supports: {', '.join(valid_datasets)}.")
    
    all_info = {
        'test_acc': [],
        'train_acc': [],
        'val_acc': [],
    }
    
    experiment_name = f'{args.dataset}_my_{args.my}_erm_{args.erm}_coes{args.contrast}-{args.spu_coe}_seed{args.seed}_{datetime_now}'
    exp_dir = os.path.join('./logs/', experiment_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir) 
        
    logger = Logger.init_logger(filename=exp_dir + '/log.log')
    args_print(args, logger)
    logger.info(f"Using criterion {criterion}")
    logger.info(f"# Train: {len(train_loader.dataset)}  #Val: {len(valid_loader.dataset)} #Test: {len(test_loader.dataset)} ")
    
    best_weights = None
    set_seed(args.seed)
        model = SDE-IG(ratio=args.r,
                input_dim=input_dim,
                edge_dim=edge_dim,
                out_dim=num_classes,
                gnn_type=args.model,
                num_layers=args.num_layers,
                emb_dim=args.emb_dim,
                drop_ratio=args.dropout,
                graph_pooling=args.pooling,
                virtual_node=args.virtual_node,
                c_dim=args.classifier_emb_dim,
                c_in=args.classifier_input_feat,
                c_rep=args.contrast_rep,
                c_pool=args.contrast_pooling,
                s_rep=args.spurious_rep).to(device)
                    
    model_optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr)
    
    last_train_acc, last_test_acc, last_val_acc = 0, 0, 0
    cnt = 0

    for epoch in range(args.epoch):
        batch_ratio_list = []
        batch_weight_list = []
        all_loss, n_bw = 0, 0
        all_losses = {}
        contrast_loss, all_contrast_loss = torch.zeros(1).to(device), 0.
        spu_pred_loss = torch.zeros(1).to(device)
        model.train()
        torch.autograd.set_detect_anomaly(True)
        num_batch = (len(train_loader.dataset) // args.batch_size) + int(
            (len(train_loader.dataset) % args.batch_size) > 0)
            
        for step, graph in tqdm(enumerate(train_loader), total=num_batch, desc=f"Epoch [{epoch}] >>  ", disable=args.no_tqdm, ncols=60):
            n_bw += 1
            graph.to(device)
            graph.x = graph.x.float()
            graph.y = graph.y.reshape(-1)
            is_labeled = graph.y == graph.y
            
            if args.caus:
                mixup_x,mixup_y,ori_pred,mix_pred,new_y,c_pred,edge_ratio_list,edge_weight_list,c_graph_pred,mix_rep = model(graph,return_data="feat",casual_mix=True,num_label=num_classes)
                cau_loss = mix_criterion(mixup_x[is_labeled],mixup_y[is_labeled])
            else:
                ori_pred,mix_pred,new_y,c_pred = model(graph,return_data="feat")
                # fix none strategy
                edge_ratio_list = []
                edge_weight_list = []
                c_graph_pred = torch.zeros_like(ori_pred)
                cau_loss = torch.zeros(1).to(device)
                
            batch_ratio_list.extend(edge_ratio_list)
            batch_weight_list.extend(edge_weight_list)

            dummy_w = torch.tensor(1.).to(device).requires_grad_()
            ori_pred_loss = criterion(ori_pred[is_labeled], graph.y[is_labeled], reduction='none')
            mix_pred_loss = criterion(mix_pred[is_labeled], new_y[is_labeled].long(), reduction='none')
            c_loss = criterion(c_pred[is_labeled], graph.y[is_labeled].long(), reduction='none')

            cgraph_loss = criterion(c_graph_pred[is_labeled], graph.y[is_labeled].long(), reduction='none')
            loss0 = criterion(ori_pred[is_labeled]*dummy_w, graph.y[is_labeled].long())
            loss1 = criterion(mix_pred[is_labeled]*dummy_w, new_y[is_labeled].long())
            grad_0 = torch.autograd.grad(loss0, dummy_w, create_graph=True)[0]
            grad_1 = torch.autograd.grad(loss1, dummy_w, create_graph=True)[0]
            irm_loss = torch.sum(grad_0 * grad_1)
            vrex_loss = torch.var(torch.FloatTensor([ori_pred_loss.mean(), mix_pred_loss.mean()]).to(device))
            all_losses['irm'] = (all_losses.get('irm', 0) * (n_bw - 1) + irm_loss.item()) / n_bw
            
            pred_loss =  ori_pred_loss.mean() + args.mix * mix_pred_loss.mean() + args.my_vrex * vrex_loss\
                        + args.my_irm * irm_loss +args.caus * cau_loss.mean()+args.c_pred * (cgraph_loss.mean()) 
            
            # compile losses
            batch_loss = pred_loss 
            model_optimizer.zero_grad()
            batch_loss.backward()
            model_optimizer.step()
            all_loss += batch_loss.item()
            
        all_contrast_loss /= n_bw
        all_loss /= n_bw

        model.eval()
        train_acc = eval_model(model, device, train_loader, evaluator, eval_metric=eval_metric,c_pred=args.c_pred)
        val_acc = eval_model(model, device, valid_loader, evaluator, eval_metric=eval_metric,c_pred=args.c_pred)
        test_acc = eval_model(model,
                                device,
                                test_loader,
                                evaluator,
                                eval_metric=eval_metric,c_pred=args.c_pred)
                                
        if val_acc > last_val_acc:
            last_train_acc = train_acc
            last_val_acc = val_acc
            last_test_acc = test_acc
            best_weights = deepcopy(model.state_dict()) # 直接无条件保存最佳权重
            cnt = 0 if last_val_acc != 1.0 else (cnt + int(epoch >= args.pretrain))
        else:
            cnt += int(epoch >= args.pretrain)

        if epoch >= args.pretrain and cnt >= args.early_stopping:
            logger.info("Early Stopping")
            logger.info("+" * 50)
            logger.info("Last: Test_ACC: {:.3f} Train_ACC:{:.3f} Val_ACC:{:.3f} ".format(
                last_test_acc, last_train_acc, last_val_acc))
            break

        all_info['test_acc'].append(last_test_acc)
        all_info['train_acc'].append(last_train_acc)
        all_info['val_acc'].append(last_val_acc)

        print("      [{:3d}/{:d}]".format(epoch, args.epoch) +
                    "\n       train_ACC: {:.4f} / {:.4f}"
                    "\n       valid_ACC: {:.4f} / {:.4f}"
                    "\n       tests_ACC: {:.4f} / {:.4f}\n".format(
                        train_acc, torch.tensor(all_info['train_acc']).max(),
                        val_acc, torch.tensor(all_info['test_acc']).max(),
                        test_acc, torch.tensor(all_info['val_acc']).max()))
    logger.info("=" * 50)

    if best_weights is not None:
        print("Saving best weights..")
        save_dir = 'save_my'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_path = os.path.join(save_dir, f"{args.dataset}_{args.r}_{datetime_now}.pt")
        for k, v in best_weights.items():
            best_weights[k] = v.cpu()
        torch.save(best_weights, model_path)
        print(f"Done.. Model saved to: {model_path}")
    else:
        print("Warning: No weights were saved (model never improved during training).")

    print("\n\n\n")
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()