import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from collections import defaultdict

def positional_encoding(pos, d_model):
    def get_angles(position, i):
        return position / np.power(10000., 2. * (i // 2.) / float(d_model))
    angle_rates = get_angles(np.arange(pos)[:, np.newaxis], np.arange(d_model)[np.newaxis, :])
    pe_sin = np.sin(angle_rates[:, 0::2])
    pe_cos = np.cos(angle_rates[:, 1::2])
    pos_encoding = np.concatenate([pe_sin, pe_cos], axis=-1)
    return pos_encoding


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src


class Encoder(nn.Module):
    def __init__(self, d_model=24, nhead=4, num_encoder_layers=3, dff=32):
        super(Encoder, self).__init__()
        self.layer_stack = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dff)
            for _ in range(num_encoder_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layer_stack:
            output = layer(output, mask, src_key_padding_mask)
        return self.norm(output)



def calculate_pse_dpc(sequence, lambda_val=5, w=0.05, properties=None):
    """计算PseDPC特征（二肽组成+耦合因子）"""
    if len(sequence) < 2:
        return np.zeros(400 + lambda_val)

    amino_acids = sorted('ACDEFGHIKLMNPQRSTVWY')

    dpc_pairs = [aa1 + aa2 for aa1 in amino_acids for aa2 in amino_acids]

    total_pairs = len(sequence) - 1
    dpc_counts = defaultdict(int)

    valid_aa_sequence = [aa for aa in sequence if aa in amino_acids]
    for i in range(len(valid_aa_sequence) - 1):
        aa_pair = valid_aa_sequence[i] + valid_aa_sequence[i + 1]
        dpc_counts[aa_pair] += 1

    dpc_vector = np.zeros(400)
    if total_pairs > 0:
        total_valid_pairs = sum(dpc_counts.values())
        for idx, pair in enumerate(dpc_pairs):
            dpc_vector[idx] = dpc_counts.get(pair, 0) / total_valid_pairs if total_valid_pairs > 0 else 0

    if properties is None:
        properties = {
            'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29,
            'Q': -0.85, 'E': -0.74, 'G': 0.48, 'H': -0.40, 'I': 1.38,
            'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12,
            'S': -0.18, 'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08
        }

    n = len(valid_aa_sequence)
    theta = []
    for lag in range(1, lambda_val + 1):
        sum_corr = 0.0
        valid_pairs = n - lag
        if valid_pairs <= 0:
            theta.append(0)
            continue
        for i in range(valid_pairs):
            aa1 = valid_aa_sequence[i]
            aa2 = valid_aa_sequence[i + lag]
            sum_corr += (properties.get(aa1, 0) - properties.get(aa2, 0)) ** 2
        theta.append(sum_corr / valid_pairs)
    theta_vector = np.array(theta)

    return np.concatenate([dpc_vector, w * theta_vector])


amino_acid_dict = {
    'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
    'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
    'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
    'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20
}

amino_acid_to_vector = {
    'A': [0.96, 16.00, 1.43, 89.30, 9.36, 7.90, 0.92, -0.04, 0.50, 0.00, 9.25, 154.33],
    'C': [0.42, 168.00, 0.94, 102.50, 2.56, 1.90, 1.16, -0.38, 0.00, 0.00, 1.07, 219.79],
    'D': [0.42, -78.00, 0.92, 114.40, 0.94, 5.50, 0.48, 0.19, 0.00, -1.00, 3.89, 194.91],
    'E': [0.53, -106.00, 1.67, 138.80, 0.94, 7.10, 0.61, 0.23, 0.00, -1.00, 4.80, 223.16],
    'F': [0.59, 189.00, 1.19, 190.80, 10.99, 3.90, 1.25, -0.38, 2.50, 0.00, 6.36, 204.74],
    'G': [0.00, -13.00, 0.46, 63.80, 6.17, 7.10, 0.61, 0.09, 0.00, 0.00, 8.51, 127.90],
    'H': [0.57, 50.00, 0.98, 157.50, 0.47, 2.10, 0.93, -0.04, 0.50, 0.00, 1.88, 242.54],
    'I': [0.84, 151.00, 1.04, 163.00, 13.73, 5.20, 1.81, -0.34, 1.80, 0.00, 6.47, 233.21],
    'K': [0.73, -141.00, 1.27, 165.10, 0.58, 6.70, 0.70, 0.33, 0.00, 1.00, 3.50, 300.46],
    'L': [0.92, 145.00, 1.36, 163.10, 16.64, 8.60, 1.30, -0.37, 1.80, 0.00, 10.94, 232.30],
    'M': [0.86, 124.00, 1.53, 165.80, 3.93, 2.40, 1.19, -0.30, 1.30, 0.00, 3.14, 202.65],
    'N': [0.39, -74.00, 0.64, 122.40, 2.31, 4.00, 0.60, 0.13, 0.00, 0.00, 3.71, 207.90],
    'P': [-2.50, -20.00, 0.49, 121.60, 1.96, 5.30, 0.40, 0.19, 0.00, 0.00, 4.36, 179.93],
    'Q': [0.80, -73.00, 1.22, 146.90, 1.14, 4.40, 0.95, 0.14, 0.00, 0.00, 3.17, 235.51],
    'R': [0.77, -70.00, 1.18, 190.30, 0.27, 4.90, 0.93, 0.07, 0.00, 1.00, 3.96, 341.01],
    'S': [0.53, -70.00, 0.70, 94.20, 5.58, 6.60, 0.82, 0.12, 0.00, 0.00, 6.26, 174.06],
    'T': [0.54, -38.00, 0.78, 119.60, 4.68, 5.30, 1.12, 0.03, 0.40, 0.00, 5.66, 205.80],
    'V': [0.63, 123.00, 0.98, 138.20, 12.43, 6.80, 1.81, -0.29, 1.50, 0.00, 7.55, 207.60],
    'W': [0.58, 145.00, 1.01, 226.40, 2.20, 1.20, 1.54, -0.33, 3.40, 0.00, 2.22, 237.01],
    'Y': [0.72, 53.00, 0.69, 194.60, 3.13, 3.10, 1.53, -0.29, 2.30, 0.00, 3.28, 229.15]
}


def process_peptide_sequence(peptide_sequence):
    """使用Transformer Encoder提取特征"""
    # 序列填充
    peptide_list = list(peptide_sequence)
    padded_sequence = peptide_list[:50] if len(peptide_list) >= 50 else peptide_list + [0] * (50 - len(peptide_list))

    extended_array = []
    for aa in padded_sequence:
        if aa == 0:
            extended_array.append([0] * 12)
        else:
            try:
                aa_char = list(amino_acid_dict.keys())[list(amino_acid_dict.values()).index(aa)]
                vector = amino_acid_to_vector[aa_char]
                extended_array.append(vector)
            except:
                extended_array.append([0] * 12)
    extended_array = np.array(extended_array, dtype=np.float32)  # (50, 12)

    pos_encoding = positional_encoding(50, 12)
    encoded_array = extended_array + pos_encoding
    encoder = Encoder(d_model=12, nhead=2, num_encoder_layers=2, dff=16)
    encoded_tensor = encoder(torch.tensor(encoded_array, dtype=torch.float32).unsqueeze(0))  # (1, 50, 12)
    feature_vector = torch.mean(encoded_tensor, dim=1).squeeze(0)  # (12,)
    return feature_vector.detach().numpy()

def encode_peptide(sequence, max_len=50):
    """将肽序列编码为固定长度的数字向量（单氨基酸）"""
    truncated = sequence[:max_len]
    encoded = [amino_acid_dict.get(aa, 0) for aa in truncated]
    if len(encoded) < max_len:
        encoded = [0] * (max_len - len(encoded)) + encoded
    return np.array(encoded)


def encode_dipeptide(sequence, max_len=50):
    """将肽序列编码为固定长度的二肽向量"""
    encoded = []
    for i in range(len(sequence) - 1):
        if i >= max_len:
            break
        aa1 = sequence[i]
        aa2 = sequence[i + 1]
        code = (amino_acid_dict.get(aa1, 0) - 1) * 20 + amino_acid_dict.get(aa2, 0)
        encoded.append(code)
    if len(encoded) < max_len:
        encoded = [0] * (max_len - len(encoded)) + encoded
    else:
        encoded = encoded[:max_len]
    return np.array(encoded)


def load_sequences(file_path):
    """返回：原始序列（截断50）、单氨基酸编码、二肽编码"""
    raw_sequences = []
    sequences_single = []
    sequences_dipeptide = []
    with open(file_path, 'r') as fp:
        for line in fp:
            if not line.startswith('>'):
                seq = line.strip().upper()[:50]
                raw_sequences.append(seq)
                sequences_single.append(encode_peptide(seq))
                sequences_dipeptide.append(encode_dipeptide(seq))
    return raw_sequences, np.array(sequences_single), np.array(sequences_dipeptide)


def load_labels(file_path):
    """加载标签数据"""
    labels = []
    with open(file_path, 'r') as fp:
        for line in fp:
            if line.startswith('>'):
                values = line[1:].strip().split('|')
                labels.append(int(values[1]))
    return np.array(labels)

def build_cnn_feature_extractor():

    feature_extractor_single = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=21, output_dim=128, input_length=50),
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(50, activation='linear')
    ])
    feature_extractor_dipeptide = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=401, output_dim=128, input_length=50),
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(50, activation='linear')
    ])
    return feature_extractor_single, feature_extractor_dipeptide



raw_sequences, X_single, X_dipeptide = load_sequences('ACP530.txt')
y = load_labels('ACP530.txt')


pse_dpc_features = []
for seq in raw_sequences:
    valid_seq = ''.join([aa for aa in seq if aa in amino_acid_dict])
    feat = calculate_pse_dpc(valid_seq)
    pse_dpc_features.append(feat)
pse_dpc_features = np.array(pse_dpc_features)

feature_extractor_single, feature_extractor_dipeptide = build_cnn_feature_extractor()

single_features = feature_extractor_single.predict(X_single)
dipeptide_features = feature_extractor_dipeptide.predict(X_dipeptide)
cnn_features = np.concatenate([single_features, dipeptide_features], axis=1)

transformer_features = []
for seq in X_single:
    peptide_sequence = ''.join([list(amino_acid_dict.keys())[i - 1] if i != 0 else '0' for i in seq])
    feature = process_peptide_sequence(peptide_sequence)
    transformer_features.append(feature)
transformer_features = np.array(transformer_features)

combined_features = np.hstack([transformer_features, cnn_features, pse_dpc_features])
scaler = StandardScaler()
combined_features = scaler.fit_transform(combined_features)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(combined_features, y)):
    print(f"\n===== 第 {fold + 1}/5 折 =====")
    X_train, X_val = combined_features[train_idx], combined_features[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((combined_features.shape[1], 1)),
        tf.keras.layers.Conv1D(filters=256, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.0005),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=50,
                        batch_size=128,
                        verbose=1)

    y_pred_prob = model.predict(X_val).ravel()
    y_pred = (y_pred_prob > 0.5).astype(int)
    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "Accuracy": accuracy_score(y_val, y_pred),
        "Sensitivity": recall_score(y_val, y_pred),
        "Specificity": tn / (tn + fp),
        "Precision": precision_score(y_val, y_pred),
        "F1 Score": f1_score(y_val, y_pred),
        "MCC": matthews_corrcoef(y_val, y_pred),
        "ROC-AUC": roc_auc_score(y_val, y_pred_prob)
    }
    fold_scores.append(metrics)

    print(f"第 {fold + 1} 折结果:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

print("\n===== 5折交叉验证平均结果 =====")
for metric in fold_scores[0].keys():
    scores = [fold[metric] for fold in fold_scores]
    print(f"{metric}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")