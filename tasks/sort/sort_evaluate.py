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
from tasks.sort.utils import prepare_model
from data.custom_datasets import SortDataset
from utils.losses import sort_loss
from tasks.sort.utils import compute_ctc_accuracy, decode_predictions



def parse_args():
    parser = argparse.ArgumentParser()

    # 从训练脚本中复制必要的参数
    parser.add_argument('--d_model', type=int, default=64, help='Dimension of the model.')
    parser.add_argument('--d_input', type=int, default=128, help='Dimension of the input.')
    parser.add_argument('--heads', type=int, default=3, help='Number of attention heads.')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of LSTM/RNN layers')
    parser.add_argument('--iterations', type=int, default=25, help='Number of internal ticks.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--positional_embedding_type', type=str, default='none',
                        help='Type of positional embedding.', choices=['none',
                                                                       'learnable-fourier',
                                                                       'multi-learnable-fourier',
                                                                       'custom-rotational'])
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--batch_size_test', type=int, default=32, help='Batch size for testing.')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate for the model.')
    parser.add_argument('--training_iterations', type=int, default=100001, help='Number of training iterations.')
    parser.add_argument('--warmup_steps', type=int, default=5000, help='Number of warmup steps.')
    parser.add_argument('--use_scheduler', action=argparse.BooleanOptionalAction, default=True,
                        help='Use a learning rate scheduler.')
    parser.add_argument('--scheduler_type', type=str, default='cosine', choices=['multistep', 'cosine'],
                        help='Type of learning rate scheduler.')
    parser.add_argument('--milestones', type=int, default=[8000, 15000, 20000], nargs='+',
                        help='Learning rate scheduler milestones.')
    parser.add_argument('--gamma', type=float, default=0.1, help='Learning rate scheduler gamma for multistep.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay factor.')
    parser.add_argument('--weight_decay_exclusion_list', type=str, nargs='+', default=[],
                        help='List to exclude from weight decay. Typically good: bn, ln, bias, start')
    parser.add_argument('--gradient_clipping', type=float, default=-1,
                        help='Gradient quantile clipping value (-1 to disable).')
    parser.add_argument('--use_amp', action=argparse.BooleanOptionalAction, default=False, help='AMP autocast.')
    parser.add_argument('--do_compile', action=argparse.BooleanOptionalAction, default=False,
                        help='Try to compile the synapses, backbone, and nlms.')
    parser.add_argument('--N_to_sort', type=int, default=6, help='N numbers to sort.')
    parser.add_argument('--save_every', type=int, default=1000, help='Save checkpoints every this many iterations.')
    parser.add_argument('--seed', type=int, default=412, help='Random seed.')
    parser.add_argument('--reload', action=argparse.BooleanOptionalAction, default=False, help='Reload from disk?')
    parser.add_argument('--reload_model_only', action=argparse.BooleanOptionalAction, default=False,
                        help='Reload only the model from disk?')
    parser.add_argument('--ei_log_dir', type=str, required=True)
    parser.add_argument('--simple_log_dir', type=str, required=True)
    parser.add_argument('--out_log_dir', type=str, default='logs/sort/ana_compared', required=True)
    parser.add_argument('--track_every', type=int, default=1000, help='Track metrics every this many iterations.')
    parser.add_argument('--n_test_batches', type=int, default=2,
                        help='How many minibatches to approx metrics. Set to -1 for full eval')
    #parser.add_argument('--output_dir', type=str, default='./comparison_results',help='Directory to save comparison results')
    parser.add_argument('--device', type=int, nargs='+', default=[-1],
                        help='List of GPU(s) to use. Set to -1 to use CPU.')

    args = parser.parse_args()
    return args


def set_global_seed(seed):
    """设置所有随机库的全局种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    SEED = 412
    args = parse_args()

    os.makedirs(args.out_log_dir, exist_ok=True)
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
    validation_data = SortDataset(args.N_to_sort)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=args.batch_size, shuffle=True, num_workers=1)
    prediction_reshaper = [-1]  # Problem specific
    args.out_dims = args.N_to_sort + 1

    # Storage for results
    results = {
        'modelei': {'validation_acc_fine': [], 'validation_acc_fulllist': []},
        'modelsimple': {'validation_acc_fine': [], 'validation_acc_fulllist': []}
    }



    # 初始化模型1并加载检查点时构建
    modelei = None
    checkpoint_path_ei = os.path.join(args.ei_log_dir, 'checkpoint.pt')
    # --- 加载模型检查点 ---
    print(f"Loading checkpoint: {checkpoint_path_ei}")
    # 加载检查点
    try:
        checkpoint = torch.load(
            checkpoint_path_ei,
            map_location=device,
            weights_only=False
        )

        # 使用检查点中保存的参数构建模型
        if 'args' in checkpoint:
            saved_args = checkpoint['args']

            if hasattr(saved_args, 'out_dims'):
                saved_args.d_input = saved_args.out_dims - 1

            #print("Saved args:", saved_args)
            #print("Current args:", args)

            if modelei is None or not compare_args(saved_args, args):
                print(f"Building model with saved parameters")
                # 注意：prepare_model 需要 prediction_reshaper 作为第一个参数
                prediction_reshaper = [-1]  # 排序任务的 reshaper
                modelei = prepare_model(prediction_reshaper, saved_args, device)
        else:
            if modelei is None:
                prediction_reshaper = [-1]
                modelei = prepare_model(prediction_reshaper, args, device)

                # 加载模型权重
        modelei.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print(f"Successfully loaded modelei from {checkpoint_path_ei}")

    except FileNotFoundError:
        print(f"Checkpoint file not found: {checkpoint_path_ei}")
        return
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path_ei}: {e}")
        return
    finally:
        # 清理内存（如果checkpoint已加载）
        if 'checkpoint' in locals():
            del checkpoint
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


    # 创建结果存储
    #iters = []
    #validation_losses = []
    #validation_accuracies_fine = []
    #validation_accuracies_most_certain = []  # 对应训练脚本中的 train_accuracies_most_certain
    #validation_accuracies_full_list = []


    # 设置为评估模式
    modelei.eval()

    # 评估测试集
    #validation_loss, validation_accuracy_fine, validation_accuracy_full = evaluate_model(
        #modelei, validationloader, device, args, num_batches=min(10, args.n_test_batches)
    #)
    # 替换单次评估
    fine_accs, full_accs = evaluate_model_multiple_runs(
        modelei, validation_data, device, args, num_runs=10
    )

    # 存储结果
    #validation_accuracies_fine.append(validation_accuracy_fine)
    #validation_accuracies_full_list.append(validation_accuracy_full)
    #results['modelei']['validation_acc_fine'].append(validation_accuracy_fine) #单论评估
    #results['modelei']['validation_acc_fulllist'].append(validation_accuracy_full) #单轮评估
    results['modelei']['validation_acc_fine'].extend(fine_accs)
    results['modelei']['validation_acc_fulllist'].extend(full_accs)
    avg_fine = np.mean(results['modelei']['validation_acc_fine'])
    avg_full = np.mean(results['modelei']['validation_acc_fulllist'])
    print(f"ModelEI - FineAcc={avg_fine:.4f}, FullAcc={avg_full:.4f}")


    # 初始化模型2并加载检查点时构建
    modelsimple = None
    checkpoint_path_simple = os.path.join(args.simple_log_dir, 'checkpoint.pt')
    # --- 加载模型检查点 ---
    print(f"Loading checkpoint: {checkpoint_path_simple}")
    # 加载检查点
    try:
        checkpoint = torch.load(
            checkpoint_path_simple,
            map_location=device,
            weights_only=False
        )

        # 使用检查点中保存的参数构建模型
        if 'args' in checkpoint:
            saved_args = checkpoint['args']

            if modelsimple is None or not compare_args(saved_args, args):
                print(f"Building model with saved parameters")
                # 注意：prepare_model 需要 prediction_reshaper 作为第一个参数
                prediction_reshaper = [-1]  # 排序任务的 reshaper
                modelsimple = prepare_model(prediction_reshaper, saved_args, device)
        else:
            if modelsimple is None:
                prediction_reshaper = [-1]
                modelsimple = prepare_model(prediction_reshaper, args, device)

                # 加载模型权重
        modelsimple.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print(f"Successfully loaded modelsimple from {checkpoint_path_simple}")

    except FileNotFoundError:
        print(f"Checkpoint file not found: {checkpoint_path_simple}")
        return
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path_simple}: {e}")
        return
    finally:
        # 清理内存（如果checkpoint已加载）
        if 'checkpoint' in locals():
            del checkpoint
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 创建结果存储
    #iters = []
    #validation_losses = []
    #validation_accuracies_fine = []
    # validation_accuracies_most_certain = []  # 对应训练脚本中的 train_accuracies_most_certain
    #validation_accuracies_full_list = []

    # 设置为评估模式
    modelsimple.eval()

    # 评估测试集
    #validation_loss, validation_accuracy_fine, validation_accuracy_full = evaluate_model(
        #modelsimple, validationloader, device, args, num_batches=min(10, args.n_test_batches)
    #)
    # 替换单次评估
    fine_accs, full_accs = evaluate_model_multiple_runs(
        modelsimple, validation_data, device, args, num_runs=10
    )

    # 存储结果
    # validation_accuracies_fine.append(validation_accuracy_fine)
    # validation_accuracies_full_list.append(validation_accuracy_full)
    #results['modelsimple']['validation_acc_fine'].append(validation_accuracy_fine) #单轮评估
    #results['modelsimple']['validation_acc_fulllist'].append(validation_accuracy_full) #单轮评估

    results['modelsimple']['validation_acc_fine'].extend(fine_accs)
    results['modelsimple']['validation_acc_fulllist'].extend(full_accs)
    avg_fine = np.mean(results['modelsimple']['validation_acc_fine'])
    avg_full = np.mean(results['modelsimple']['validation_acc_fulllist'])
    print(f"ModelSimple - FineAcc={avg_fine:.4f}, FullAcc={avg_full:.4f}")


    plot_results(results,args.out_log_dir)

    print(f"Evaluation completed! Plots saved to {args.out_log_dir}")


def evaluate_model(model, dataloader, device, args, num_batches=-1):
    """评估模型并返回损失和准确率"""
    all_losses = []
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        total_batches = min(num_batches, len(dataloader)) if num_batches != -1 else len(dataloader)
        pbar = tqdm(total=total_batches, desc='Evaluating', leave=False)

        for i, (inputs, targets) in enumerate(dataloader):  # 修正：只有inputs和targets
            if num_batches != -1 and i >= num_batches:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)

            predictions, certainties, synchronisation = model(inputs)  # 修正：不需要z参数

            loss = sort_loss(predictions, targets)  # 修正：使用sort_loss
            all_losses.append(loss.item())

            # 计算CTC准确率
            accuracy = compute_ctc_accuracy(predictions, targets, predictions.shape[1] - 1)

            # 解码预测结果用于详细评估
            decoded = [d[:targets.shape[1]] for d in decode_predictions(predictions, predictions.shape[1] - 1)]
            decoded = torch.stack([
                torch.concatenate((d, torch.zeros(targets.shape[1] - len(d), device=targets.device) + targets.shape[1]))
                if len(d) < targets.shape[1] else d for d in decoded
            ], 0)

            all_predictions.append(decoded.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

            pbar.update(1)

        pbar.close()

        # 计算指标
    avg_loss = np.mean(all_losses)

    if all_predictions:
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)

        # 细粒度准确率（每个位置）
        fine_grained_accuracy = (all_predictions == all_targets).mean()

        # 完整列表准确率
        full_list_accuracy = (all_predictions == all_targets).all(-1).mean()

        return avg_loss, fine_grained_accuracy, full_list_accuracy
    else:
        return avg_loss, 0.0, 0.0


def evaluate_model_multiple_runs(model, dataset, device, args, num_runs=10):
    """多次评估模型以计算标准误"""
    fine_accs = []
    full_accs = []

    for run in range(num_runs):
        # 每次使用不同的随机种子创建数据加载器
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size_test,
            shuffle=True, num_workers=1
        )

        _, fine_acc, full_acc = evaluate_model(
            model, dataloader, device, args,
            num_batches=args.n_test_batches
        )

        fine_accs.append(fine_acc)
        full_accs.append(full_acc)

    return fine_accs, full_accs

def plot_results(results, out_log_dir):
    """绘制带标准误的柱状图"""
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    os.makedirs(out_log_dir, exist_ok=True)
    # 计算均值和标准误
    models = ['modelei', 'modelsimple']
    fine_means = []
    fine_stds = []
    full_means = []
    full_stds = []

    for model in models:
        fine_acc = np.array(results[model]['validation_acc_fine'])
        full_acc = np.array(results[model]['validation_acc_fulllist'])

        fine_means.append(np.mean(fine_acc))
        fine_stds.append(np.std(fine_acc) / np.sqrt(len(fine_acc)))  # 标准误
        full_means.append(np.mean(full_acc))
        full_stds.append(np.std(full_acc) / np.sqrt(len(full_acc)))  # 标准误

    # 创建柱状图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(models))
    width = 0.35

    # Fine-grained accuracy 柱状图
    bars1 = ax1.bar(x, fine_means, width, yerr=fine_stds, capsize=5,
                    color=['skyblue', 'lightcoral'], alpha=0.8)
    ax1.set_xlabel('models')
    ax1.set_ylabel('Fine-grained Accuracy')
    ax1.set_title(' Fine-grained Accuracy ')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.grid(True, alpha=0.3)

    # 在柱状图上添加数值标签
    for i, (mean, std) in enumerate(zip(fine_means, fine_stds)):
        ax1.text(i, mean + std + 0.01, f'{mean:.3f}±{std:.3f}',
                 ha='center', va='bottom')

        # Full list accuracy 柱状图
    bars2 = ax2.bar(x, full_means, width, yerr=full_stds, capsize=5,
                    color=['skyblue', 'lightcoral'], alpha=0.8)
    ax2.set_xlabel('models')
    ax2.set_ylabel('Full List Accuracy')
    ax2.set_title(' Full List Accuracy ')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.grid(True, alpha=0.3)

    # 在柱状图上添加数值标签
    for i, (mean, std) in enumerate(zip(full_means, full_stds)):
        ax2.text(i, mean + std + 0.01, f'{mean:.3f}±{std:.3f}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f'{out_log_dir}/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"plog saved in {out_log_dir}/model_comparison.png")

if __name__ == '__main__':
    main()