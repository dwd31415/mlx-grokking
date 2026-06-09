import argparse
import random
import numpy as np
import mlx.nn as nn
import mlx.core as mx
import mlx.optimizers as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from functools import partial

from models import Transformer
from data import grokking_data
from mlx.utils import tree_flatten, tree_unflatten

parser = argparse.ArgumentParser(add_help=True)
# data args
parser.add_argument('--p', type=int, default=97, help='prime number')
parser.add_argument('--op', type=str, default='/',
                    help='operation', choices=['*', '/', '+', '-'])
parser.add_argument('--train-fraction', type=float,
                    default=0.5, help='train fraction')
# model args
parser.add_argument('--depth', type=int, default=2, help='depth')
parser.add_argument('--dim', type=int, default=128, help='dimension')
parser.add_argument('--heads', type=int, default=1, help='heads')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
# optimizer args
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--weight-decay', type=float,
                    default=1, help='weight decay')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1')
parser.add_argument('--beta2', type=float, default=0.98, help='beta2')
parser.add_argument('--warmup', type=int, default=10, help='warmup steps')
# training args
parser.add_argument('-b', '--batch_size', type=int,
                    default=512, help='batch size')
parser.add_argument('-e', '--epochs', type=int,
                    default=150, help='number of epochs')
# misc args
parser.add_argument('--seed', type=int, default=random.randint(0,1000), help='random seed')
parser.add_argument('--cpu', action='store_true', help='use cpu only')


def padded_access(x, idx):
    if idx < 0:
        if len(x) + idx < 0:
            return 0
        return x[len(x) + idx]
    elif idx >= len(x):
        return 0
    else:
        return x[idx]

class NeuralNetwork:
    def __init__(self,
                 model: nn.Module,
                 optimizer: optim.Optimizer,
                 classification: bool = False,
                 batch_size: int = 64):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = nn.losses.cross_entropy if classification else nn.losses.mse_loss
        self.batch_size = batch_size

        # bookkeeping
        self.train_error_trace = []
        self.train_acc_trace = []
        self.val_error_trace = []
        self.val_acc_trace = []
        self.memorization_epoch = None
        self.memorization_hessian_eigenvalues = None
        self.memorization_hessian_negative_eigenvalues = None
        self.generalization_epochs = []
        self.generalization_hessian_negative_eigenvalues = []

    def _make_batches(self, X, T):
        bs = self.batch_size if self.batch_size != -1 else X.shape[0]
        for i in range(0, X.shape[0], bs):
            yield X[i:i+bs], T[i:i+bs]

    def eval_fn(self, X, T):
        Y = self.model(X)
        loss = self.loss_fn(Y, T, reduction='mean')
        correct = mx.sum(mx.argmax(Y, axis=1) == T)
        return loss, correct

    def _flatten_parameters(self):
        flat_params = tree_flatten(self.model.parameters())
        spec = [(path, leaf.shape, leaf.size) for path, leaf in flat_params]
        theta = mx.concatenate([mx.reshape(leaf, (-1,)) for _, leaf in flat_params])
        return theta, spec

    @staticmethod
    def _unflatten_parameters(theta, spec):
        params = {}
        offset = 0
        for path, shape, size in spec:
            params[path] = theta[offset:offset + size].reshape(shape)
            offset += size
        return tree_unflatten(params)

    def _top_hessian_eigenvalues(self, X, T, top_k=100):
        theta0, spec = self._flatten_parameters()

        def loss_from_theta(theta):
            self.model.update(self._unflatten_parameters(theta, spec))
            Y = self.model(X)
            return self.loss_fn(Y, T, reduction='mean')

        grad_fn = mx.grad(loss_from_theta)

        def hvp(vec):
            return mx.grad(lambda th: mx.sum(grad_fn(th) * vec))(theta0)

        n = theta0.size
        k = min(top_k, n)
        q_prev = mx.zeros_like(theta0)
        q = mx.random.normal(shape=theta0.shape)
        q_norm = mx.sqrt(mx.sum(q * q))
        if float(q_norm) == 0.0:
            q = mx.ones_like(theta0) / mx.sqrt(float(n))
        else:
            q = q / q_norm

        alphas = []
        betas = []

        for i in range(k):
            z = hvp(q)
            if i > 0:
                z = z - betas[-1] * q_prev

            alpha = float(mx.sum(q * z))
            z = z - alpha * q

            beta = float(mx.sqrt(mx.sum(z * z)))
            alphas.append(alpha)
            if i < k - 1:
                betas.append(beta)

            if beta == 0.0:
                break

            q_prev, q = q, z / beta

        m = len(alphas)
        tridiag = np.zeros((m, m), dtype=np.float64)
        np.fill_diagonal(tridiag, alphas)
        if m > 1:
            idx = np.arange(m - 1)
            tridiag[idx, idx + 1] = betas[:m - 1]
            tridiag[idx + 1, idx] = betas[:m - 1]

        eigvals = np.linalg.eigvalsh(tridiag)[::-1]
        return eigvals[:min(top_k, eigvals.shape[0])]

    def train(self, train_data, val_data, epochs=5, shuffle=True):
        state = [self.model.state, self.optimizer.state, mx.random.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def step(X, T):
            train_step_fn = nn.value_and_grad(self.model, self.eval_fn)
            (loss, correct), grads = train_step_fn(X, T)
            self.optimizer.update(self.model, grads)
            return loss, correct

        epoch_bar = tqdm(range(epochs), desc='Training', unit='epoch')
        for epoch in epoch_bar:
            self.model.train()
            if shuffle:
                inds = mx.array(np.random.permutation(train_data[0].shape[0]))
                train_data = [v[inds] for v in train_data]

            total_loss, total_correct = 0, 0
            for X, T in self._make_batches(*train_data):
                loss, correct = step(X, T)
                mx.eval(state)
                total_loss += loss.item() * X.shape[0]
                total_correct += correct.item()

            avg_train_loss = total_loss / train_data[0].shape[0]
            avg_train_acc = total_correct / train_data[0].shape[0]

            self.train_error_trace.append(avg_train_loss)
            self.train_acc_trace.append(avg_train_acc)
 
            postfix = {'train_loss': f'{avg_train_loss:.3f}',
                       'train_acc': f'{avg_train_acc:.3f}'}

            # eval on validation data
            avg_val_loss, avg_val_acc = self.evaluate(val_data)
            self.val_error_trace.append(avg_val_loss)
            self.val_acc_trace.append(avg_val_acc)
            postfix.update({'val_loss': f'{avg_val_loss:.3f}',
                            'val_acc': f'{avg_val_acc:.3f}'})

            if avg_train_acc > 0.97 and avg_val_acc < 0.5 and padded_access(self.train_error_trace, -2) - padded_access(self.train_error_trace, -1) < 0.005 and self.memorization_epoch is None:
                print(f"Memorization happened at epoch {epoch}!")
                self.memorization_epoch = epoch
                if self.memorization_hessian_eigenvalues is None:
                    probe_size = min(256, train_data[0].shape[0])
                    probe_data = (train_data[0][:probe_size], train_data[1][:probe_size])
                    self.model.eval()
                    try:
                        self.memorization_hessian_eigenvalues = self._top_hessian_eigenvalues(
                            *probe_data, top_k=200
                        )
                    finally:
                        self.model.train()
                    top = self.memorization_hessian_eigenvalues
                    print(np.sum(top * (top < 0)))
                    self.memorization_hessian_negative_eigenvalues = int(np.sum(top < 0))
                    print(
                        f"Negative Hessian eigenvalues among top {len(top)} at memorization: "
                        f"{self.memorization_hessian_negative_eigenvalues}"
                    )
            if avg_train_acc > 0.97 and avg_val_acc > 0.97 and padded_access(self.train_error_trace, -2) - padded_access(self.train_error_trace, -1) < 0.005 and len(self.generalization_epochs) == 0:
                print(f"Generalization happened at epoch {epoch}!")
                self.generalization_epochs.append(epoch)
                probe_size = min(256, train_data[0].shape[0])
                probe_data = (train_data[0][:probe_size], train_data[1][:probe_size])
                self.model.eval()
                try:
                    hessian_eigenvalues = self._top_hessian_eigenvalues(*probe_data, top_k=200)
                finally:
                    self.model.train()
                negative_count = int(np.sum(hessian_eigenvalues < 0))
                print(np.sum(hessian_eigenvalues * (hessian_eigenvalues < 0)))
                self.generalization_hessian_negative_eigenvalues.append(negative_count)
                print(
                    f"Negative Hessian eigenvalues among top {len(hessian_eigenvalues)} at generalization: "
                    f"{negative_count}"
                )

            epoch_bar.set_postfix(postfix)

    def evaluate(self, test_data):
        self.model.eval()
        total_loss, total_correct = 0, 0
        for X, T in self._make_batches(*test_data):
            Y = self.model(X)
            loss = self.loss_fn(Y, T, reduction='mean')
            correct = mx.sum(mx.argmax(Y, axis=1) == T)

            total_loss += loss.item() * X.shape[0]
            total_correct += correct.item()

        avg_loss = total_loss / test_data[0].shape[0]
        avg_acc = total_correct / test_data[0].shape[0]

        return avg_loss, avg_acc


def main(args):
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    Xtrain, Ttrain, Xtest, Ttest = grokking_data(
        args.p, op=args.op, train_fraction=args.train_fraction)

    kwargs = {
        'depth': args.depth,
        'dim': args.dim,
        'heads': args.heads,
        'n_tokens': args.p + 2,
        'seq_len': Xtrain.shape[1],
        'dropout': args.dropout
    }
    model = Transformer(**kwargs)
    model.summary()

    warmup = optim.linear_schedule(0, args.lr, args.warmup)
    optimizer = optim.AdamW(learning_rate=warmup,
                            betas=(args.beta1, args.beta2),
                            weight_decay=args.weight_decay)
    net = NeuralNetwork(model, optimizer, classification=True,
                        batch_size=args.batch_size)
    net.train((Xtrain, Ttrain), (Xtest, Ttest),
              epochs=args.epochs, shuffle=True)

    # !plotting
    fig, ax = plt.subplots(figsize=(5, 3.5))
    lw = 2
    ax.plot(np.array(net.train_acc_trace) * 100, 
            label='train', color='#1b9e77', lw=lw)
    ax.plot(np.array(net.val_acc_trace) * 100, 
            label='val', color='#d95f02', lw=lw)
    plt.axvline(x=net.memorization_epoch, color='red', linestyle='--', linewidth=1.5, label='memorization')
    for gen_epoch in net.generalization_epochs:
        plt.axvline(x=gen_epoch, color='green', linestyle='--', linewidth=1.5, label='generalization')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.legend()
    fig.tight_layout()
    fig.savefig('media/grokking.png', dpi=300)
    plt.show()

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(np.array(net.train_error_trace),
            label='train', color='#1b9e77', lw=lw)
    ax.plot(np.array(net.val_error_trace),
            label='val', color='#d95f02', lw=lw)
    plt.axvline(x=net.memorization_epoch, color='red', linestyle='--', linewidth=1.5, label='memorization')
    for gen_epoch in net.generalization_epochs:
        plt.axvline(x=gen_epoch, color='green', linestyle='--', linewidth=1.5, label='generalization')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    fig.tight_layout()
    fig.savefig('media/grokking_loss.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    args = parser.parse_args()
    if args.cpu:
        mx.set_default_device(mx.cpu)
    main(args)
