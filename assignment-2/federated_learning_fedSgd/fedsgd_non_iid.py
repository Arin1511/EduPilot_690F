# FedSGD Non-IID implementation (self-contained)
import os, pickle, numpy as np, pandas as pd
from pathlib import Path

from sklearn.metrics import accuracy_score

SEED = 42
BASE_DIR = Path(__file__).resolve().parent
ART = "artifacts_centralized"
NUM_CLIENTS  = 5
ROUNDS       = 100
LR_GLOBAL    = 0.05
HIDDEN_UNITS = 64
LAMBDA       = 1e-4
MODE = "NON_IID"

def make_label_skew_splits(y, k=5, seed=42, labels_per_client=2):
	rng = np.random.default_rng(seed)
	y = np.asarray(y); classes = np.unique(y)
	buckets = {c: list(np.where(y==c)[0]) for c in classes}
	for b in buckets.values(): rng.shuffle(b)
	splits = [[] for _ in range(k)]
	for i in range(k):
		chosen = rng.choice(classes, size=min(labels_per_client, len(classes)), replace=False)
		for c in chosen:
			take = max(1, len(buckets[c]) // k)
			splits[i].extend(buckets[c][:take])
			buckets[c] = buckets[c][take:]
	leftovers = [idx for v in buckets.values() for idx in v]
	rng.shuffle(leftovers)
	for i, idx in enumerate(leftovers):
		splits[i % k].append(idx)
	return [sorted(s) for s in splits]

class NeuralNetwork:
	def __init__(self, layer_sizes, seed=42, lambd=0.0):
		np.random.seed(seed)
		self.layer_sizes = layer_sizes
		self.num_layers = len(layer_sizes)
		self.lambd = lambd
		self.weights = []
		for i in range(self.num_layers - 1):
			w = np.random.randn(layer_sizes[i + 1], layer_sizes[i] + 1) * np.sqrt(2.0 / layer_sizes[i])
			self.weights.append(w)

	def sigmoid_function(self, z):
		z = np.array(z, dtype=np.float64)
		return 1.0 / (1.0 + np.exp(-z))

	def sigmoid_derivative_function(self, a):
		return a * (1.0 - a)

	def softmax_function(self, z):
		z = z - z.max(axis=0, keepdims=True)
		ez = np.exp(z)
		return ez / np.clip(ez.sum(axis=0, keepdims=True), 1e-12, None)

	def forward_pass(self, X, use_softmax_output=True):
		pre_acts, acts = [], []
		A = X
		for l, W in enumerate(self.weights[:-1]):
			A_bias = np.vstack([np.ones((1, A.shape[1])), A])
			Z = W @ A_bias
			A = self.sigmoid_function(Z)
			pre_acts.append(Z); acts.append(A)
		Wout = self.weights[-1]
		A_bias = np.vstack([np.ones((1, A.shape[1])), A])
		ZL = Wout @ A_bias
		if use_softmax_output:
			AL = self.softmax_function(ZL)
		else:
			AL = self.sigmoid_function(ZL)
		pre_acts.append(ZL); acts.append(AL)
		return pre_acts, acts

	def backward_pass_multiclass(self, acts, X, Y_onehot):
		N = X.shape[1]
		grads = [np.zeros_like(W) for W in self.weights]
		H_list = acts[:-1]
		AL = acts[-1]
		dZ = AL - Y_onehot
		A_prev = X if len(H_list) == 0 else H_list[-1]
		Ab_prev = np.vstack([np.ones((1, A_prev.shape[1])), A_prev])
		grads[-1] = (dZ @ Ab_prev.T) / N
		if self.lambd > 0:
			reg = self.weights[-1].copy(); reg[:, 0] = 0
			grads[-1] += (self.lambd / N) * reg
		dA = self.weights[-1][:, 1:].T @ dZ
		for l in reversed(range(len(H_list))):
			A_hidden = H_list[l]
			A_prev = X if l == 0 else H_list[l - 1]
			dZ = dA * self.sigmoid_derivative_function(A_hidden)
			Ab_prev = np.vstack([np.ones((1, A_prev.shape[1])), A_prev])
			grads[l] = (dZ @ Ab_prev.T) / N
			if self.lambd > 0:
				reg = self.weights[l].copy(); reg[:, 0] = 0
				grads[l] += (self.lambd / N) * reg
			if l > 0:
				dA = self.weights[l][:, 1:].T @ dZ
		return grads

	def get_params_vector(self):
		flats = [W.reshape(-1) for W in self.weights]
		return np.concatenate(flats, axis=0)

	def set_params_vector(self, vec):
		offset = 0
		for l in range(len(self.weights)):
			shape = self.weights[l].shape
			size = shape[0] * shape[1]
			self.weights[l] = vec[offset:offset+size].reshape(shape)
			offset += size

	def predict_multiclass(self, X):
		_, acts = self.forward_pass(X, use_softmax_output=True)
		return np.argmax(acts[-1], axis=0)

	def compute_multiclass_gradient(self, X, Y_labels):
		C = int(np.max(Y_labels)) + 1
		Y_onehot = np.zeros((C, Y_labels.shape[0]), dtype=np.float64)
		Y_onehot[Y_labels, np.arange(Y_labels.shape[0])] = 1.0
		_, acts = self.forward_pass(X, use_softmax_output=True)
		grads = self.backward_pass_multiclass(acts, X, Y_onehot)
		return np.concatenate([g.flatten() for g in grads])

def main():
	np.random.seed(SEED)
	with open(os.path.join(ART,"tfidf_vectorizer.pkl"),"rb") as f: vec = pickle.load(f)
	with open(os.path.join(ART,"label_encoder.pkl"),"rb") as f:  le  = pickle.load(f)
	tr = pd.read_csv(os.path.join(ART,"centralized_train_text_labels.csv"))
	te = pd.read_csv(os.path.join(ART,"centralized_test_text_labels.csv"))

	Xtr = vec.transform(tr["text"].astype(str)).toarray().astype(np.float64)
	ytr = le.transform(tr["label"].astype(str).values)
	Xte = vec.transform(te["text"].astype(str)).toarray().astype(np.float64)
	yte = le.transform(te["label"].astype(str).values)
	D, C = Xtr.shape[1], len(le.classes_)
	Xte_T = Xte.T
	print(f"[{MODE}] data  Ntr={len(Xtr)} Nte={len(Xte)} D={D} C={C}")

	splits = make_label_skew_splits(ytr, k=NUM_CLIENTS, seed=SEED, labels_per_client=2)

	import collections
	for i, idxs in enumerate(splits,1):
		cnt = collections.Counter(ytr[idxs])
		print(f"[{MODE}] client {i}: N={len(idxs)} hist={dict(cnt)}")

	def client_data(idxs):
		Xi = Xtr[idxs]; yi = ytr[idxs]
		return Xi.T, yi

	layer_sizes = [D, HIDDEN_UNITS, C]
	global_model = NeuralNetwork(layer_sizes=layer_sizes, lambd=LAMBDA)
	global_vec = global_model.get_params_vector()

	print(f"\n=== FL (FedSGD | {MODE}) START === clients={NUM_CLIENTS} rounds={ROUNDS} lr={LR_GLOBAL}")
	base_pred = global_model.predict_multiclass(Xte_T)
	print(f"[{MODE}] pre-round acc={accuracy_score(yte, base_pred):.4f}")

	acc_hist = []
	for r in range(1, ROUNDS+1):
		grads = []
		for idxs in splits:
			Xi, yi = client_data(idxs)
			local = NeuralNetwork(layer_sizes=layer_sizes, lambd=LAMBDA)
			local.set_params_vector(global_vec.copy())
			grad = local.compute_multiclass_gradient(Xi, yi)
			grads.append(grad)
		avg_grad = np.mean(np.stack(grads, axis=0), axis=0)
		global_vec -= LR_GLOBAL * avg_grad
		global_model.set_params_vector(global_vec.copy())
		acc = accuracy_score(yte, global_model.predict_multiclass(Xte_T))
		acc_hist.append(float(acc))
		print(f"[FedSGD | {MODE} | Round {r:02d}] acc={acc:.4f}")

	pd.DataFrame({"round": np.arange(1, len(acc_hist)+1), "acc": acc_hist}).to_csv("../fl_non_iid_fedsgd_accuracy.csv", index=False)
	print("Saved: fl_non_iid_fedsgd_accuracy.csv")

if __name__ == "__main__":
	main()