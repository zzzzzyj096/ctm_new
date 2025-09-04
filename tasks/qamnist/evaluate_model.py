import argparse
import multiprocessing  # Used for GIF generation
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm
sns.set_style('darkgrid')
from utils.samplers import QAMNISTSampler
from utils.losses import qamnist_loss
from tasks.qamnist.utils import get_dataset, prepare_model
from models.utils import reshape_predictions, get_latest_checkpoint
from tasks.parity.utils import reshape_attention_weights
import glob
import pandas as pd
import time



def parse_args():
    parser = argparse.ArgumentParser()

    # 从训练脚本中复制必要的参数
    parser.add_argument('--q_num_images', type=int, default=3)
    parser.add_argument('--q_num_images_delta', type=int, default=2)
    parser.add_argument('--q_num_repeats_per_input', type=int, default=10)
    parser.add_argument('--q_num_operations', type=int, default=3)
    parser.add_argument('--q_num_operations_delta', type=int, default=2)
    parser.add_argument('--q_num_answer_steps', type=int, default=10)

    parser.add_argument('--model_type', type=str, default='ctm')
    parser.add_argument('--d_model', type=int, default=1024)
    parser.add_argument('--d_input', type=int, default=64)
    parser.add_argument('--synapse_depth', type=int, default=1)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--n_synch_out', type=int, default=32)
    parser.add_argument('--n_synch_action', type=int, default=32)
    parser.add_argument('--neuron_select_type', type=str, default='random')
    parser.add_argument('--n_random_pairing_self', type=int, default=256)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--memory_length', type=int, default=30)
    parser.add_argument('--deep_memory', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--memory_hidden_dims', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--do_normalisation', action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument('--batch_size_test', type=int, default=16)
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='data/')
    parser.add_argument('--seed', type=int, default=412)
    parser.add_argument('--device', type=int, nargs='+', default=[-1])
    parser.add_argument('--n_test_batches', type=int, default=10)
    parser.add_argument('--checkpoint_interval', type=int, default=1000, help='检查点保存间隔')
    parser.add_argument('--start_iter', type=int, default=0, help='起始检查点迭代步')
    parser.add_argument('--end_iter', type=int, default=100000, help='结束检查点迭代步')
    parser.add_argument('--skip_existing', action='store_true', help='跳过已评估的检查点')

    return parser.parse_args()


def compare_args(saved_args, current_args):
    """比较两个参数对象是否相同"""
    # 比较关键参数
    key_params = [
        'd_model', 'd_input', 'n_synch_out', 'n_synch_action',
        'heads', 'synapse_depth', 'memory_length', 'deep_memory'
    ]
    for key in key_params:
        if getattr(saved_args, key, None) != getattr(current_args, key, None):
            return False
    return True


def main():
    args = parse_args()

    # 设置设备
    if args.device[0] != -1:
        device = f'cuda:{args.device[0]}'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Evaluating model on {device}')

    # 加载数据
    print("Loading dataset...")
    train_data, test_data, class_labels, dataset_mean, dataset_std = get_dataset(
        args.q_num_images, args.q_num_images_delta, args.q_num_repeats_per_input,
        args.q_num_operations, args.q_num_operations_delta
    )

    # 添加缺失的属性设置
    args.out_dims = len(class_labels)
    args.use_most_certain = args.model_type == "ctm"

    test_sampler = QAMNISTSampler(test_data, batch_size=args.batch_size_test)
    testloader = torch.utils.data.DataLoader(test_data, num_workers=0, batch_sampler=test_sampler)

    train_sampler = QAMNISTSampler(train_data, batch_size=args.batch_size_test)
    trainloader = torch.utils.data.DataLoader(train_data, num_workers=0, batch_sampler=train_sampler)

    # 初始化模型为None，将在加载检查点时构建
    model = None

    # 查找所有检查点
    checkpoint_files = glob.glob(os.path.join(args.log_dir, 'checkpoint_*.pt'))

    # 提取迭代步数并排序
    checkpoint_iters = []
    for file in checkpoint_files:
        try:
            iter_num = int(os.path.basename(file).split('_')[1].split('.')[0])
            if args.start_iter <= iter_num <= args.end_iter:
                checkpoint_iters.append(iter_num)
        except:
            continue

    checkpoint_iters.sort()
    # 自动设置结束迭代步数为最大迭代步数
    if not checkpoint_iters:
        print(f"No checkpoints found in {args.log_dir} between iterations {args.start_iter} and {args.end_iter}")
        return

    max_iter = max(checkpoint_iters)
    args.end_iter = max_iter  # 更新结束迭代步数

    # 过滤在指定范围内的检查点
    filtered_checkpoint_iters = [iter_num for iter_num in checkpoint_iters
                                 if args.start_iter <= iter_num <= args.end_iter]

    print(f"Found {len(filtered_checkpoint_iters)} checkpoints to evaluate (from {min(filtered_checkpoint_iters)} to {max(filtered_checkpoint_iters)} iterations)")

    # 创建结果存储
    iters = []
    train_losses = []
    test_losses = []
    train_accuracies = []  # 对应训练脚本中的 train_accuracies
    test_accuracies = []  # 对应训练脚本中的 test_accuracies
    train_accuracies_most_certain = []  # 对应训练脚本中的 train_accuracies_most_certain
    test_accuracies_most_certain = []  # 对应训练脚本中的 test_accuracies_most_certain

    # 评估每个检查点
    for iter_num in tqdm(checkpoint_iters, desc="Evaluating checkpoints"):
        checkpoint_path = os.path.join(args.log_dir, f'checkpoint_{iter_num}.pt')

        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            continue

        start_time = time.time()

        # 加载检查点
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False
            )

            # 关键修改：使用检查点中保存的参数构建模型
            if 'args' in checkpoint:
                saved_args = checkpoint['args']
                # 确保使用与训练时相同的设备
                saved_args.device = args.device

                # 如果模型尚未构建或参数不同，则重新构建
                if model is None or not compare_args(saved_args, args):
                    print(f"Building model with saved parameters for iteration {iter_num}")
                    model = prepare_model(saved_args, device)
            else:
                # 如果检查点中没有保存参数，使用当前参数
                if model is None:
                    model = prepare_model(args, device)

            # 加载模型权重
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_path}: {e}")
            continue

        # 设置为评估模式
        model.eval()

        # 评估测试集
        test_loss, test_accuracy, test_accuracy_most_certain = evaluate_model(
            model, testloader, device, args, num_batches=args.n_test_batches
        )

        # 评估训练集（部分数据）
        train_loss, train_accuracy, train_accuracy_most_certain = evaluate_model(
            model, trainloader, device, args, num_batches=min(5, args.n_test_batches)  # 使用更少的训练批次
        )

        eval_time = time.time() - start_time

        # 存储结果
        iters.append(iter_num)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        train_accuracies_most_certain.append(train_accuracy_most_certain)
        test_accuracies_most_certain.append(test_accuracy_most_certain)

        print(f"Iter {iter_num}: Train Loss={train_loss:.4f}, Acc={train_accuracy_most_certain:.4f} | "
              f"Test Loss={test_loss:.4f}, Acc={test_accuracy_most_certain:.4f} | Time={eval_time:.1f}s")

        # 清理内存
        del checkpoint
        if 'cuda' in device:
            torch.cuda.empty_cache()

    # 生成与训练脚本完全相同的图表
    plot_results(
        iters,
        train_losses, test_losses,
        train_accuracies, test_accuracies,
        train_accuracies_most_certain, test_accuracies_most_certain,
        args.log_dir
    )

    print(f"Evaluation completed! Plots saved to {args.log_dir}")


def evaluate_model(model, dataloader, device, args, num_batches=-1):
    """评估模型并返回损失和准确率"""
    all_predictions = []
    all_predictions_most_certain = []
    all_targets = []
    all_losses = []

    prediction_reshaper = [-1]

    with torch.no_grad():
        total_batches = min(num_batches, len(dataloader)) if num_batches != -1 else len(dataloader)
        pbar = tqdm(total=total_batches, desc='Evaluating', leave=False)

        for i, (inputs, z, question_readable, targets) in enumerate(dataloader):
            if num_batches != -1 and i >= num_batches:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)
            z = torch.stack(z, 1).to(device)

            predictions, certainties, synchronisation = model(inputs, z)

            predictions_answer_steps = predictions[:, :, -args.q_num_answer_steps:]
            certainties_answer_steps = certainties[:, :, -args.q_num_answer_steps:]

            loss, where_most_certain = qamnist_loss(
                predictions_answer_steps, certainties_answer_steps, targets,
                use_most_certain=args.use_most_certain
            )

            all_losses.append(loss.item())
            all_targets.append(targets.cpu().numpy())

            # 计算所有时间步的平均准确率
            all_time_steps_accuracy = (predictions_answer_steps.argmax(1) == targets.unsqueeze(1)).float().mean().item()

            # 计算最确定时间步的准确率
            most_certain_accuracy = (predictions_answer_steps.argmax(1)[
                                         torch.arange(predictions_answer_steps.size(0), device=device),
                                         where_most_certain
                                     ] == targets).float().mean().item()

            pbar.update(1)

        pbar.close()

    # 计算指标
    avg_loss = np.mean(all_losses)

    return avg_loss, all_time_steps_accuracy, most_certain_accuracy


def plot_results(iters,train_losses, test_losses,train_accuracies, test_accuracies,train_accuracies_most_certain, test_accuracies_most_certain,log_dir):

    if not iters:
        print("No results to plot")
        return

    # 创建图表目录
    plot_dir = os.path.join(log_dir, 'evaluation_plots')
    os.makedirs(plot_dir, exist_ok=True)

    # 获取最后一个点的数值
    last_iter = iters[-1]
    last_train_loss = train_losses[-1]
    last_test_loss = test_losses[-1]
    last_train_acc_mc = train_accuracies_most_certain[-1]
    last_test_acc_mc = test_accuracies_most_certain[-1]

    # 1. 准确率图表（与训练脚本中的 accuracies.png 相同）
    figacc = plt.figure(figsize=(10, 10))
    axacc_train = figacc.add_subplot(211)
    axacc_test = figacc.add_subplot(212)

    # 训练集准确率
    axacc_train.plot(iters, train_accuracies_most_certain, 'k--', alpha=0.7,
                     label=f'Most certain: {last_train_acc_mc:.4f}')
    axacc_train.set_title('Train')
    axacc_train.set_xlabel('Iteration')
    axacc_train.set_ylabel('Accuracy')
    axacc_train.legend(loc='lower right')
    if iters:
        axacc_train.set_xlim([0, max(iters)])
    axacc_train.grid(True)


    # 测试集准确率
    axacc_test.plot(iters, test_accuracies_most_certain, 'k--', alpha=0.7,
                    label=f'Most certain: {last_test_acc_mc:.4f}')
    axacc_test.set_title('Test')
    axacc_test.set_xlabel('Iteration')
    axacc_test.set_ylabel('Accuracy')
    axacc_test.legend(loc='lower right')
    if iters:
        axacc_test.set_xlim([0, max(iters)])
    axacc_test.grid(True)


    figacc.tight_layout()
    figacc.savefig(os.path.join(plot_dir, 'accuracies.png'), dpi=150)
    plt.close(figacc)

    # 2. 损失图表（与训练脚本中的 losses.png 相同）
    figloss = plt.figure(figsize=(10, 5))
    axloss = figloss.add_subplot(111)

    # 训练和测试损失
    axloss.plot(iters, train_losses, 'b-', linewidth=1, alpha=0.8,
                label=f'Train: {last_train_loss:.4f}')
    axloss.plot(iters, test_losses, 'r-', linewidth=1, alpha=0.8,
                label=f'Test: {last_test_loss:.4f}')
    axloss.set_xlabel('Iteration')
    axloss.set_ylabel('Loss')
    axloss.legend(loc='upper right')
    if iters:
        axloss.set_xlim([0, max(iters)])
    axloss.grid(True)


    figloss.tight_layout()
    figloss.savefig(os.path.join(plot_dir, 'losses.png'), dpi=150)
    plt.close(figloss)

    # 3. 组合图表（可选）
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # 损失子图
    ax1.plot(iters, train_losses, 'b-', label=f'Train: {last_train_loss:.4f}')
    ax1.plot(iters, test_losses, 'r-', label=f'Test: {last_test_loss:.4f}')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend(loc='upper right')
    ax1.grid(True)


    # 准确率子图
    ax2.plot(iters, train_accuracies_most_certain, 'b-', label=f'Train: {last_train_acc_mc:.4f}')
    ax2.plot(iters, test_accuracies_most_certain, 'r-', label=f'Test: {last_test_acc_mc:.4f}')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Curves')
    ax2.legend(loc='lower right')
    ax2.grid(True)


    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'combined_curves.png'), dpi=150)
    plt.close(fig)

    print(f"Plots saved to {plot_dir}")


if __name__ == '__main__':
    main()