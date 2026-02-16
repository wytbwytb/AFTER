import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
import os
from einops import rearrange
import pickle
import torch.nn.functional as F
from tqdm import tqdm
from utils import get_com_directions, load_and_conbine_activations, get_separated_activations


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model', type=str, default='llava_v1.5_7B_lht', help="specifies the model to be evaluated.")
    parser.add_argument('--dataset_name', type=str, default='', help="specifies the path to the data")
    parser.add_argument('--input_dim', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_heads', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=32)
    parser.add_argument('--query_set', type=str, default='POPE_train_I+Q')
    parser.add_argument('--caption_set', type=str, default='POPE_train_T+Q_query')
    parser.add_argument('--vector_set', type=str, default='POPE_train_T+Q_best')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--subfix", type=str, default='')
    parser.add_argument("--path", type=str, default='/path/to/your/workdir/AFTER/features')
    parser.add_argument("--save_path", type=str, default='')
    
    args = parser.parse_args()
    return args


class OffsetGenerator(nn.Module):
    def __init__(self, input_dim=128, latent_dim=128, num_layer=32, num_head=32):
        super().__init__()
        self.num_layer = num_layer
        self.num_head = num_head
        self.nets = nn.ModuleList()
        for i in range(num_layer * num_head):
            self.nets.append(nn.Sequential(
            nn.Linear(input_dim, latent_dim, dtype=torch.float32), 
            nn.GELU(),
            nn.Linear(latent_dim, input_dim, dtype=torch.float32)
        ))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        """
        x shape: [batch_size, 32, 32, 128]
        """
        output = torch.zeros_like(x, device=x.device, dtype=torch.float32)
    
        for i in range(self.num_layer):
            for j in range(self.num_head):
                idx = i * self.num_head + j
                output_patch = self.nets[idx](x[:, i, j, :])
                if torch.isnan(output_patch).any():  
                    print(f"NaN出现在layer{i},{j}!")
                output[:, i, j, :] = output_patch
                
        return output


def train(offset_model, vector_data, train_dataloader, val_dataloader, optimizer, epochs, args):
    offset_model.train()
    loss_fn = nn.MSELoss()  
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for batch_idx, (query, vector_gt) in enumerate(train_dataloader):
            optimizer.zero_grad()
            
            offset = offset_model(query)
            pred = offset + vector_data
            loss = loss_fn(pred, vector_gt)

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader.dataset)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
        
        if (epoch + 1) % 5 == 0:
            print(f'Save {args.save_path}_q_{epoch + 1}.pth')
            torch.save({
                'model_state_dict': offset_model.state_dict(),
                'model_config': {
                    'input_dim': args.input_dim,
                    'latent_dim': args.latent_dim,
                    'num_layer': 32,
                    'num_head': 32
                }
            }, f'{args.save_path}_q_{epoch + 1}.pth')
    return offset_model

def val(offset_model, vector_data, val_dataloader):
    if val_dataloader == None:
        return
    # 测试重构能力
    sims = []
    dsts = []
    for batch_idx, (query, vector_gt) in enumerate(val_dataloader):
        # data = data.to(args.device)
        
        offset = offset_model(query)
        pred = offset + vector_data
        
        sim = torch.nn.functional.cosine_similarity(pred, vector_gt, dim=-1)
        dst = torch.norm(pred - vector_gt, p=2, dim=-1)
        
        sims.append(torch.mean(sim).item())
        dsts.append(torch.mean(dst).item())
        # print(loss)
    
    avg_sim = sum(sims) / len(sims)
    avg_dst = sum(dsts) / len(dsts)
    
    print(f'Val sim: {avg_sim:.4f}, Val dst: {avg_dst:.4f}')


def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + 0.1 * KLD

def prepare_data(query_data_path, caption_data_path, batch_size=16, num_layer=32, num_head=32):
    # 生成示例数据（替换为真实数据加载逻辑）
    
    query_data = np.load(query_data_path)
    caption_data = np.load(caption_data_path)
    vector_gt_data = caption_data - query_data
    
    query_data = query_data.reshape(query_data.shape[0], num_layer, num_head, -1)
    vector_gt_data = vector_gt_data.reshape(vector_gt_data.shape[0], num_layer, num_head, -1)
    
    dataset = TensorDataset(torch.from_numpy(query_data).cuda().float(), torch.from_numpy(vector_gt_data).cuda().float())
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == '__main__':
    args = get_args()
    # 超参数设置
    
    # args.epochs = 10
    # args.lr = 1e-3
    # args.model = 'llava_v1.5_7B_lht'
    # args.query_set = 'POPE_train_YR_I+Q'
    # args.caption_set = 'POPE_train_YR_C_p2+Q_query'
    # args.vector_set = 'POPE_train_YR_C_p2+Q_best'
    # # args.vector_set = ['POPE_train_I+Q','POPE_train_C+Q_best']
    # args.save_path = f'/path/to/your/workdir/AFTER/probes/{args.model}_offset_generator_YR'

    print(os.path.exists(args.save_path))
    
    model = OffsetGenerator().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
    subfix = f'head_wise{args.subfix}'
    
    vector_data_path = f'{args.model}_{args.vector_set}_{subfix}.npy'
    vector_data_path = os.path.join(args.path, vector_data_path)
    
    caption_data_path = f'{args.model}_{args.caption_set}_{subfix}.npy'
    caption_data_path = os.path.join(args.path, caption_data_path)
        
    query_data_path = f'{args.model}_{args.query_set}_{subfix}.npy'
    query_data_path = os.path.join(args.path, query_data_path)

    head_wise_activations, labels = load_and_conbine_activations(
        pos_path=vector_data_path,
        neg_path=query_data_path
        )
    
    # change h according to different model
    head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h = args.num_heads)
    
    if 'POPE' in args.query_set:
        split_range = 12
    elif 'AMBER' in args.query_set:
        split_range = 2

    separated_head_wise_activations, separated_labels, idxs_to_split_at = get_separated_activations(labels, head_wise_activations, split_range)
    
    usable_labels = np.concatenate(separated_labels, axis=0)
    vector_data = []
    for layer in range(args.num_layers): 
        for head in range(args.num_heads): 
            usable_head_wise_activations = np.concatenate([separated_head_wise_activations[i][:,layer,head,:] for i in range(len(separated_head_wise_activations))], axis=0)
            true_mass_mean = np.mean(usable_head_wise_activations[usable_labels == 1], axis=0)
            false_mass_mean = np.mean(usable_head_wise_activations[usable_labels == 0], axis=0)
            vector_data.append(true_mass_mean - false_mass_mean)
    vector_data = np.array(vector_data)
    vector_data = vector_data.reshape(args.num_layers, args.num_heads, -1)
    vector_data = torch.from_numpy(vector_data).cuda().float()

    train_dataloader = prepare_data(query_data_path, caption_data_path, args.batch_size)
    
    val_dataloader = None
    # 训练阶段
    print("Starting Training...")
    trained_model = train(model, vector_data, train_dataloader, val_dataloader, optimizer, args.epochs, args)
