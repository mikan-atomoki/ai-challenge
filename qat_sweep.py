"""
QAT Seed Sweep — 1GPUで1シードを回す軽量スクリプト
Usage: CUDA_VISIBLE_DEVICES=0 python qat_sweep.py --seed 42 --out results/run_42.pt
"""
import os
import argparse
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from models import ResNet20, convert_to_bitnet

# Config (main.py と同じ)
QAT_EPOCHS = 100
QAT_LR = 0.005
QAT_BATCH = 512
NUM_WORKERS = 4
DATA_DIR = "./data"
PRUNED_CKPT = "./checkpoints/student_pruned.pt"


def evaluate(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=QAT_BATCH, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=QAT_BATCH, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True)

    # Load pruned model
    pruned_model = ResNet20(num_classes=10).to(device)
    pruned_model.load_state_dict(
        torch.load(PRUNED_CKPT, map_location=device, weights_only=True))

    # Sparsity mask
    sparsity_masks = {}
    for name, param in pruned_model.named_parameters():
        if 'weight' in name:
            sparsity_masks[name] = (param.data != 0).float().to(device)

    # BitNet model
    model = copy.deepcopy(pruned_model)
    model = convert_to_bitnet(model).to(device)

    optimizer = optim.Adam(model.parameters(), lr=QAT_LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=QAT_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None

    for epoch in range(1, QAT_EPOCHS + 1):
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in sparsity_masks:
                        param.data.mul_(sparsity_masks[name])
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1 or epoch == QAT_EPOCHS:
            test_acc = evaluate(model, testloader, device)
            print(f"  [seed={args.seed}] Epoch {epoch:3d}/{QAT_EPOCHS} | "
                  f"Test Acc: {test_acc:.2f}%")
            if test_acc > best_acc:
                best_acc = test_acc
                best_state = copy.deepcopy(model.state_dict())

    # Save result
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"seed": args.seed, "accuracy": best_acc,
                "model_state_dict": best_state}, args.out)
    print(f"  [seed={args.seed}] DONE — Best: {best_acc:.2f}% -> {args.out}")


if __name__ == "__main__":
    main()
