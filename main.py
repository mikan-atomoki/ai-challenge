"""
CIFAR-10 Extreme Compression Pipeline
======================================
1. Teacher Training   : WideResNet-28-10
2. Knowledge Distill. : Teacher -> ResNet-20 (Student)
3. Pruning            : Iterative magnitude pruning (30%->50%->70%->80%)
4. 1.58-bit QAT       : BitNet b1.58 quantization-aware training
"""

import os
import copy
import time
import gzip
import io
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms

from models import WideResNet, ResNet20, convert_to_bitnet

# =============================================================================
# Config
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4
DATA_DIR = "./data"
SAVE_DIR = "./checkpoints"

# Teacher training
TEACHER_EPOCHS = 200
TEACHER_LR = 0.4
TEACHER_BATCH = 512

# Distillation
DISTILL_EPOCHS = 200
DISTILL_LR = 0.4
DISTILL_BATCH = 512
DISTILL_T = 4.0      # Temperature
DISTILL_ALPHA = 0.7   # KL weight

# Pruning
PRUNE_STAGES = [0.30, 0.50, 0.70, 0.80]
PRUNE_FINETUNE_EPOCHS = 30
PRUNE_LR = 0.01

# QAT (1.58bit)
QAT_EPOCHS = 150
QAT_LR = 0.005
QAT_BATCH = 512
QAT_WARMUP = 10      # LR warmup epochs


# =============================================================================
# Data
# =============================================================================
def get_dataloaders(batch_size=128):
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
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True)
    return trainloader, testloader


# =============================================================================
# Evaluation helpers
# =============================================================================
@torch.no_grad()
def evaluate(model, testloader):
    model.eval()
    correct = 0
    total = 0
    for images, labels in testloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_nonzero_parameters(model):
    total = 0
    nonzero = 0
    for p in model.parameters():
        total += p.numel()
        nonzero += p.count_nonzero().item()
    return nonzero, total


def model_size_mb(model):
    """Model size in MB (state_dict on disk)."""
    import tempfile
    tmp_path = os.path.join(tempfile.gettempdir(), "_tmp_model.pt")
    torch.save(model.state_dict(), tmp_path)
    size = os.path.getsize(tmp_path) / (1024 * 1024)
    os.remove(tmp_path)
    return size


def model_size_gz_mb(model):
    """Model size in MB after gzip compression."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    raw = buffer.getvalue()
    gz_buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=gz_buffer, mode="wb") as f:
        f.write(raw)
    return len(gz_buffer.getvalue()) / (1024 * 1024)


def compute_sparsity(model):
    nonzero, total = count_nonzero_parameters(model)
    return 100.0 * (1.0 - nonzero / total) if total > 0 else 0.0


def print_model_stats(name, model, accuracy):
    nonzero, total = count_nonzero_parameters(model)
    sparsity = compute_sparsity(model)
    size = model_size_mb(model)
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Accuracy       : {accuracy:.2f}%")
    print(f"  Parameters     : {total:,} (non-zero: {nonzero:,})")
    print(f"  Sparsity       : {sparsity:.1f}%")
    print(f"  Model Size     : {size:.2f} MB")
    print(f"{'='*60}\n")


# =============================================================================
# Step 1: Teacher Training (WideResNet-28-10)
# =============================================================================
def train_teacher(trainloader, testloader):
    print("\n" + "#" * 60)
    print("# Step 1: Training Teacher (WideResNet-28-10)")
    print("#" * 60)

    model = WideResNet(depth=28, widen_factor=10, dropout_rate=0.3, num_classes=10).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=TEACHER_LR,
                          momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TEACHER_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, TEACHER_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        scheduler.step()

        train_acc = 100.0 * correct / total
        test_acc = evaluate(model, testloader)
        print(f"  Epoch {epoch:3d}/{TEACHER_EPOCHS} | "
              f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.5f}")
        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs(SAVE_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "teacher_best.pt"))

    # Load best
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "teacher_best.pt"),
                                     map_location=DEVICE, weights_only=True))
    final_acc = evaluate(model, testloader)
    print_model_stats("Teacher (WideResNet-28-10)", model, final_acc)
    return model, final_acc


# =============================================================================
# Step 2: Knowledge Distillation (Teacher -> Student)
# =============================================================================
def distill_loss(student_logits, teacher_logits, labels, T, alpha):
    """KD loss = alpha * KL(soft) + (1-alpha) * CE(hard)."""
    soft_student = F.log_softmax(student_logits / T, dim=1)
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    kl = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T * T)
    ce = F.cross_entropy(student_logits, labels)
    return alpha * kl + (1 - alpha) * ce


def train_distillation(teacher, trainloader, testloader):
    print("\n" + "#" * 60)
    print("# Step 2: Knowledge Distillation (Teacher -> ResNet-20)")
    print("#" * 60)

    teacher.eval()
    student = ResNet20(num_classes=10).to(DEVICE)

    optimizer = optim.SGD(student.parameters(), lr=DISTILL_LR,
                          momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=DISTILL_EPOCHS)

    best_acc = 0.0
    for epoch in range(1, DISTILL_EPOCHS + 1):
        student.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.no_grad():
                teacher_logits = teacher(images)
            student_logits = student(images)

            loss = distill_loss(student_logits, teacher_logits, labels,
                                T=DISTILL_T, alpha=DISTILL_ALPHA)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        scheduler.step()

        train_acc = 100.0 * correct / total
        test_acc = evaluate(student, testloader)
        print(f"  Epoch {epoch:3d}/{DISTILL_EPOCHS} | "
              f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.5f}")
        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs(SAVE_DIR, exist_ok=True)
            torch.save(student.state_dict(),
                       os.path.join(SAVE_DIR, "student_distilled.pt"))

    # Load best
    student.load_state_dict(torch.load(os.path.join(SAVE_DIR, "student_distilled.pt"),
                                       map_location=DEVICE, weights_only=True))
    final_acc = evaluate(student, testloader)
    print_model_stats("Student (ResNet-20, Distilled)", student, final_acc)
    return student, final_acc


# =============================================================================
# Step 3: Iterative Magnitude Pruning
# =============================================================================
def get_prunable_layers(model):
    """Get all Conv2d and Linear layers for pruning."""
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layers.append((module, "weight"))
    return layers


def apply_global_pruning(model, target_sparsity):
    """Apply global unstructured L1 pruning to reach target sparsity."""
    layers = get_prunable_layers(model)
    prune.global_unstructured(
        layers,
        pruning_method=prune.L1Unstructured,
        amount=target_sparsity,
    )


def remove_pruning_reparametrization(model):
    """Make pruning permanent by removing forward hooks."""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            try:
                prune.remove(module, "weight")
            except ValueError:
                pass


def finetune(model, trainloader, testloader, epochs, lr):
    """Fine-tune a pruned model."""
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch % 5 == 0 or epoch == epochs:
            test_acc = evaluate(model, testloader)
            print(f"    Fine-tune epoch {epoch:3d}/{epochs} | Test Acc: {test_acc:.2f}%")


def train_pruning(student, trainloader, testloader):
    print("\n" + "#" * 60)
    print("# Step 3: Iterative Magnitude Pruning")
    print("#" * 60)

    model = copy.deepcopy(student)

    for target_sparsity in PRUNE_STAGES:
        print(f"\n  --- Pruning to {target_sparsity*100:.0f}% sparsity ---")

        # Remove existing parametrizations before re-applying
        remove_pruning_reparametrization(model)

        # Apply global pruning
        apply_global_pruning(model, target_sparsity)

        actual_sparsity = compute_sparsity(model)
        test_acc = evaluate(model, testloader)
        print(f"  After pruning  : Sparsity={actual_sparsity:.1f}%, Acc={test_acc:.2f}%")

        # Fine-tune
        finetune(model, trainloader, testloader, PRUNE_FINETUNE_EPOCHS, PRUNE_LR)

        test_acc = evaluate(model, testloader)
        print(f"  After fine-tune: Sparsity={compute_sparsity(model):.1f}%, Acc={test_acc:.2f}%")

    # Make pruning permanent
    remove_pruning_reparametrization(model)

    final_acc = evaluate(model, testloader)
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "student_pruned.pt"))
    print_model_stats("Student (Pruned 80%)", model, final_acc)
    return model, final_acc


# =============================================================================
# Step 4: 1.58-bit Quantization-Aware Training (BitNet b1.58)
# =============================================================================
def train_qat(pruned_model, trainloader, testloader):
    print("\n" + "#" * 60)
    print("# Step 4: 1.58-bit QAT (BitNet b1.58) + LR Warmup")
    print("#" * 60)

    # Build sparsity mask from pruned model BEFORE converting to BitNet
    sparsity_masks = {}
    for name, param in pruned_model.named_parameters():
        if 'weight' in name:
            sparsity_masks[name] = (param.data != 0).float().to(DEVICE)

    model = copy.deepcopy(pruned_model)
    model = convert_to_bitnet(model)
    model = model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=QAT_LR, weight_decay=1e-5)
    # Warmup + Cosine: linear warmup for QAT_WARMUP epochs, then cosine decay
    warmup_sched = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=QAT_WARMUP)
    cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=QAT_EPOCHS - QAT_WARMUP)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[QAT_WARMUP])
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, QAT_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Re-apply sparsity mask to keep pruned weights at zero
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in sparsity_masks:
                        param.data.mul_(sparsity_masks[name])

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        scheduler.step()

        train_acc = 100.0 * correct / total
        if epoch % 5 == 0 or epoch == 1:
            test_acc = evaluate(model, testloader)
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:3d}/{QAT_EPOCHS} | "
                  f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | "
                  f"LR: {lr_now:.6f}")
            if test_acc > best_acc:
                best_acc = test_acc
                os.makedirs(SAVE_DIR, exist_ok=True)
                torch.save(model.state_dict(),
                           os.path.join(SAVE_DIR, "student_bitnet.pt"))

    # Load best
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "student_bitnet.pt"),
                                     map_location=DEVICE, weights_only=True))
    final_acc = evaluate(model, testloader)
    print_model_stats("Student (Pruned + 1.58bit QAT)", model, final_acc)
    return model, final_acc


# =============================================================================
# Final Summary
# =============================================================================
def print_summary(results):
    print("\n" + "=" * 85)
    print("  FINAL COMPARISON SUMMARY")
    print("=" * 85)
    header = (f"  {'Stage':<35} {'Acc':>7} {'Params':>12} "
              f"{'Sparsity':>9} {'Size(MB)':>9} {'Gzip(MB)':>9}")
    print(header)
    print("  " + "-" * 81)
    for name, acc, model in results:
        nonzero, total = count_nonzero_parameters(model)
        sparsity = compute_sparsity(model)
        size = model_size_mb(model)
        gz_size = model_size_gz_mb(model)
        print(f"  {name:<35} {acc:>6.2f}% {total:>12,} "
              f"{sparsity:>8.1f}% {size:>8.2f} {gz_size:>8.2f}")
    print("=" * 85)


# =============================================================================
# Main
# =============================================================================
def main():
    print(f"Device: {DEVICE}")
    start_time = time.time()

    # Data
    trainloader, testloader = get_dataloaders(TEACHER_BATCH)

    results = []

    # Step 1: Teacher
    teacher_ckpt = os.path.join(SAVE_DIR, "teacher_best.pt")
    if os.path.exists(teacher_ckpt):
        print("\n[INFO] Found existing teacher checkpoint, loading...")
        teacher = WideResNet(depth=28, widen_factor=10, num_classes=10).to(DEVICE)
        teacher.load_state_dict(torch.load(teacher_ckpt, map_location=DEVICE, weights_only=True))
        teacher_acc = evaluate(teacher, testloader)
        print_model_stats("Teacher (WideResNet-28-10) [loaded]", teacher, teacher_acc)
    else:
        teacher, teacher_acc = train_teacher(trainloader, testloader)
    results.append(("Teacher (WRN-28-10)", teacher_acc, teacher))

    # Step 2: Distillation
    student_ckpt = os.path.join(SAVE_DIR, "student_distilled.pt")
    if os.path.exists(student_ckpt):
        print("\n[INFO] Found existing distilled student checkpoint, loading...")
        student = ResNet20(num_classes=10).to(DEVICE)
        student.load_state_dict(torch.load(student_ckpt, map_location=DEVICE, weights_only=True))
        student_acc = evaluate(student, testloader)
        print_model_stats("Student (ResNet-20, Distilled) [loaded]", student, student_acc)
    else:
        student, student_acc = train_distillation(teacher, trainloader, testloader)
    results.append(("Student (Distilled)", student_acc, student))

    # Step 3: Pruning
    pruned_ckpt = os.path.join(SAVE_DIR, "student_pruned.pt")
    if os.path.exists(pruned_ckpt):
        print("\n[INFO] Found existing pruned student checkpoint, loading...")
        pruned_model = ResNet20(num_classes=10).to(DEVICE)
        pruned_model.load_state_dict(torch.load(pruned_ckpt, map_location=DEVICE, weights_only=True))
        pruned_acc = evaluate(pruned_model, testloader)
        print_model_stats("Student (Pruned 80%) [loaded]", pruned_model, pruned_acc)
    else:
        pruned_model, pruned_acc = train_pruning(student, trainloader, testloader)
    results.append(("Student (Pruned 80%)", pruned_acc, pruned_model))

    # Step 4: 1.58-bit QAT
    bitnet_ckpt = os.path.join(SAVE_DIR, "student_bitnet.pt")
    if os.path.exists(bitnet_ckpt):
        print("\n[INFO] Found existing BitNet checkpoint, loading...")
        bitnet_model = ResNet20(num_classes=10)
        bitnet_model = convert_to_bitnet(bitnet_model).to(DEVICE)
        bitnet_model.load_state_dict(torch.load(bitnet_ckpt, map_location=DEVICE, weights_only=True))
        bitnet_acc = evaluate(bitnet_model, testloader)
        print_model_stats("Student (Pruned + 1.58bit QAT) [loaded]", bitnet_model, bitnet_acc)
    else:
        bitnet_model, bitnet_acc = train_qat(pruned_model, trainloader, testloader)
    results.append(("Student (Pruned + 1.58bit)", bitnet_acc, bitnet_model))

    # Summary
    print_summary(results)

    # Export: 全部入り .pt（モデル構造 + 重み + メタ情報）
    export_path = os.path.join(SAVE_DIR, "cifar10_compressed_all_in_one.pt")
    nonzero, total = count_nonzero_parameters(bitnet_model)
    payload = {
        "model_state_dict": bitnet_model.state_dict(),
        "architecture": "ResNet-20 + BitNet b1.58",
        "compression": {
            "distillation": {"teacher": "WideResNet-28-10", "T": DISTILL_T, "alpha": DISTILL_ALPHA},
            "pruning": {"method": "global L1 unstructured", "stages": PRUNE_STAGES},
            "quantization": {"method": "BitNet b1.58", "weight_bits": 1.58, "activation_bits": 8},
        },
        "accuracy": bitnet_acc,
        "params_total": total,
        "params_nonzero": nonzero,
        "sparsity": compute_sparsity(bitnet_model),
    }
    # 非圧縮で保存
    torch.save(payload, export_path)
    raw_size = os.path.getsize(export_path) / (1024 * 1024)
    # gzip圧縮版も保存
    gz_path = export_path + ".gz"
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    with gzip.open(gz_path, "wb") as f:
        f.write(buffer.getvalue())
    gz_size = os.path.getsize(gz_path) / (1024 * 1024)
    print(f"\n[EXPORT] All-in-one model saved: {export_path}")
    print(f"         Raw size : {raw_size:.2f} MB")
    print(f"         Gzip size: {gz_size:.2f} MB ({raw_size/gz_size:.1f}x compression)")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/3600:.1f} hours ({elapsed:.0f} seconds)")


if __name__ == "__main__":
    main()
