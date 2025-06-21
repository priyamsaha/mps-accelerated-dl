"""
MPS/GPU-Accelerated Deep Learning Model Training
============================================
A comprehensive implementation for comparing CPU vs MPS/GPU performance in deep learning training.

Requirements:
- torch>=2.0.0
- torchvision>=0.15.0
- matplotlib>=3.5.0
- numpy>=1.21.0
- tqdm>=4.64.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
import json
from datetime import datetime

class CNNModel(nn.Module):
    """
    Convolutional Neural Network for CIFAR-10 classification
    """
    def __init__(self, num_classes=10):
        super(CNNModel, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))

        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 128 * 4 * 4)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class DeepLearningBenchmark:
    """
    Main class for benchmarking CPU vs MPS/GPU performance in deep learning training
    """
    
    def __init__(self, dataset='cifar10', batch_size=128, num_epochs=10):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.results = {}

        self.mps_available = torch.backends.mps.is_available()
        if self.mps_available:
            self.mps_device = torch.device("mps")
            print(f"MPS Available: {torch.mps.device_count()} MPS devices detected")
            print(f"MPS Memory: {torch.mps.recommended_max_memory() / 1024**3:.3f} GB")
        else:
            print("MPS not available, will only run CPU benchmarks")
        
        self.device_cpu = torch.device('cpu')

        self.train_loader, self.test_loader, self.num_classes = self._setup_data()
        
    def _setup_data(self):
        """Setup data loaders for the chosen dataset"""
        if self.dataset == 'cifar10':
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            
            trainset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=transform_train
            )
            testset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transform_test
            )
            
            num_classes = 10
            
        elif self.dataset == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            trainset = torchvision.datasets.MNIST(
                root='./data', train=True, download=True, transform=transform
            )
            testset = torchvision.datasets.MNIST(
                root='./data', train=False, download=True, transform=transform
            )
            
            num_classes = 10

        train_loader = DataLoader(
            trainset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )
        test_loader = DataLoader(
            testset, batch_size=self.batch_size, shuffle=False, num_workers=2
        )
        
        print(f"Dataset: {self.dataset.upper()}")
        print(f"Training samples: {len(trainset)}")
        print(f"Test samples: {len(testset)}")
        print(f"Batch size: {self.batch_size}")
        
        return train_loader, test_loader, num_classes
    
    def train_model(self, device, device_name):
        """Train the model on specified device"""
        print(f"\n{'='*50}")
        print(f"Training on {device_name}")
        print(f"{'='*50}")

        model = CNNModel(num_classes=self.num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        train_losses = []
        train_accuracies = []
        epoch_times = []

        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            model.train()
            
            running_loss = 0.0
            correct = 0
            total = 0

            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}')
            
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                pbar.set_postfix({
                    'Loss': f'{running_loss/(batch_idx+1):.3f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })

            epoch_loss = running_loss / len(self.train_loader)
            epoch_acc = 100. * correct / total
            epoch_time = time.time() - epoch_start_time
            
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc)
            epoch_times.append(epoch_time)

            scheduler.step()
            
            print(f'Epoch {epoch+1}: Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%, Time: {epoch_time:.2f}s')
        
        total_training_time = time.time() - start_time

        test_accuracy = self._test_model(model, device, device_name)

        self.results[device_name] = {
            'total_training_time': total_training_time,
            'average_epoch_time': np.mean(epoch_times),
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'epoch_times': epoch_times,
            'test_accuracy': test_accuracy,
            'final_train_accuracy': train_accuracies[-1]
        }
        
        print(f'\nTraining completed in {total_training_time:.2f} seconds')
        print(f'Average epoch time: {np.mean(epoch_times):.2f} seconds')
        print(f'Final test accuracy: {test_accuracy:.2f}%')
        
        return model
    
    def _test_model(self, model, device, device_name):
        """Test the trained model"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc=f'Testing on {device_name}'):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy
    
    def run_benchmark(self):
        """Run the complete benchmark comparing CPU and MPS"""
        print("Starting Deep Learning Performance Benchmark")
        print(f"Dataset: {self.dataset.upper()}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Batch Size: {self.batch_size}")
        
        # Train on CPU
        print("\n" + "="*60)
        print("PHASE 1: CPU TRAINING")
        print("="*60)
        cpu_model = self.train_model(self.device_cpu, 'CPU')
        
        # Train on MPS
        if self.mps_available:
            print("\n" + "="*60)
            print("PHASE 2: MPS TRAINING")
            print("="*60)
            mps_model = self.train_model(self.mps_device, 'MPS')
        
        # Generate comparison report
        self._generate_report()
        self._plot_results()
        
        return self.results
    
    def _generate_report(self):
        """Generate a comprehensive performance report"""
        report = []
        report.append("="*80)
        report.append("DEEP LEARNING MODEL PERFORMANCE BENCHMARK REPORT")
        report.append("="*80)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Dataset: {self.dataset.upper()}")
        report.append(f"Number of epochs: {self.num_epochs}")
        report.append(f"Batch size: {self.batch_size}")
        report.append("")
        
        report.append("SYSTEM INFORMATION:")
        report.append("-" * 40)
        report.append(f"PyTorch version: {torch.__version__}")
        if self.mps_available:
            report.append(f"MPS Available: {torch.mps.device_count()}")
            report.append(f"MPS Memory: {torch.mps.recommended_max_memory() / 1024**3:.1f} GB")
        report.append("")

        report.append("PERFORMANCE COMPARISON:")
        report.append("-" * 40)
        
        for device_name, results in self.results.items():
            report.append(f"\n{device_name} Results:")
            report.append(f"  Total training time: {results['total_training_time']:.2f} seconds")
            report.append(f"  Average epoch time: {results['average_epoch_time']:.2f} seconds")
            report.append(f"  Final training accuracy: {results['final_train_accuracy']:.2f}%")
            report.append(f"  Test accuracy: {results['test_accuracy']:.2f}%")

        if len(self.results) > 1:
            cpu_time = self.results['CPU']['total_training_time']
            mps_time = self.results['MPS']['total_training_time']
            speedup = cpu_time / mps_time
            
            report.append(f"\nSPEEDUP ANALYSIS:")
            report.append(f"  MPS is {speedup:.2f}x faster than CPU")
            report.append(f"  Time saved: {cpu_time - mps_time:.2f} seconds ({((cpu_time - mps_time) / cpu_time * 100):.1f}%)")

        report_text = "\n".join(report)
        print("\n" + report_text)
        
        with open(f'benchmark_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt', 'w') as f:
            f.write(report_text)

        with open(f'benchmark_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def _plot_results(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Deep Learning Performance Benchmark Results', fontsize=16)

        if len(self.results) > 1:
            devices = list(self.results.keys())
            times = [self.results[device]['total_training_time'] for device in devices]
            
            axes[0, 0].bar(devices, times, color=['skyblue', 'lightcoral'])
            axes[0, 0].set_title('Total Training Time Comparison')
            axes[0, 0].set_ylabel('Time (seconds)')
            for i, v in enumerate(times):
                axes[0, 0].text(i, v + max(times)*0.01, f'{v:.1f}s', ha='center')

        for device_name, results in self.results.items():
            epochs = range(1, len(results['epoch_times']) + 1)
            axes[0, 1].plot(epochs, results['epoch_times'], marker='o', label=device_name)
        
        axes[0, 1].set_title('Epoch Training Time Progression')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Time (seconds)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Training Loss Progression
        for device_name, results in self.results.items():
            epochs = range(1, len(results['train_losses']) + 1)
            axes[1, 0].plot(epochs, results['train_losses'], marker='o', label=device_name)
        
        axes[1, 0].set_title('Training Loss Progression')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Training Accuracy Progression
        for device_name, results in self.results.items():
            epochs = range(1, len(results['train_accuracies']) + 1)
            axes[1, 1].plot(epochs, results['train_accuracies'], marker='o', label=device_name)
        
        axes[1, 1].set_title('Training Accuracy Progression')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'benchmark_plots_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', dpi=300, bbox_inches='tight')

def main():
    """Main function to run the benchmark"""
    print("MPS-Accelerated Deep Learning Model Training Benchmark")
    print("=" * 60)
    
    config = {
        'dataset': 'cifar10',
        'batch_size': 128,
        'num_epochs': 10
    }
    
    benchmark = DeepLearningBenchmark(**config)

    results = benchmark.run_benchmark()
    
    print("\nBenchmark completed successfully!")
    print("Check the generated files for detailed results and plots.")

if __name__ == "__main__":
    main()
