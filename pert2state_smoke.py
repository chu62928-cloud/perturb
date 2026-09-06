import json
from pathlib import Path

import numpy as np
import pandas as pd
from pert2state_model import Perturb2StateModel

rng = np.random.default_rng(19)
genes = [f'G{i}' for i in range(6)]
X = pd.DataFrame(rng.normal(size=(6, 3)), index=genes, columns=['P1', 'P2', 'P3'])
y = pd.Series(0.7 * X['P1'] - 0.2 * X['P2'] + 0.1 * X['P3'], index=genes, name='target')
model = Perturb2StateModel(n_splits=1, random_state=19, alpha=0.1, l1_ratio=0.5)
model.fit(X, y, model_id='tiny')
pred = model.get_prediction(X)
out = Path('/root/autodl-tmp/CRISPR_perturb_runtime/smoke/pert2state_tiny_predictions.csv')
pred.to_csv(out)
print(json.dumps({'model': 'pert2state-model', 'version': '0.0.1', 'input_shape': list(X.shape), 'prediction_shape': list(pred.shape), 'prediction_file': str(out), 'finite': bool(np.isfinite(pred.to_numpy()).all()), 'mean_prediction': float(pred['pred_mean'].mean())}))
