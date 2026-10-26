"""Additional end-to-end isolation and independent saved-weight replay tests.

Kept separate from the frozen v2 execution files: these checks do not modify
the protocol, selectors, optimizers, or any ongoing source training.
"""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.mixed95.c3_che_data import read, sequences
from experiments.mixed95.c3_che_v2_data import (
    BASE, DEFAULT_SUITE, PROTOCOL, GROUPS, load_candidates, select_features, selected_arrays,
)
from experiments.mixed95.c3_che_v2_model import fit_source_run, train_grid, svd_dense
from experiments.mixed95 import test_c3_che_v2 as v2_tests


class AdditionalIsolationTests(unittest.TestCase):
    def test_hidden_labels_cannot_change_source_arrays_or_weights(self):
        arrays,meta,labels=v2_tests.V2Tests().fixture()
        calibration=[r for r in labels if r['cycle']!=101]
        first=select_features(arrays,meta,calibration)
        changed=[dict(r,soh=99.) if r['cycle']==101 else dict(r) for r in labels]
        second=select_features(arrays,meta,[r for r in changed if r['cycle']!=101])
        cells,state=selected_arrays(arrays,first);other,other_state=selected_arrays(arrays,second)
        self.assertEqual(state,other_state)
        for a,b in zip(cells,other):
            for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        with tempfile.TemporaryDirectory() as directory, patch.dict(PROTOCOL,{'max_epochs':8,'patience':3}):
            folders=[Path(directory)/'first',Path(directory)/'mutated']
            fit_source_run(cells,state,4,1,2021,folders[0],'isolation')
            fit_source_run(other,other_state,4,1,2021,folders[1],'isolation')
            weights=[torch.load(p/'source.pt',map_location='cpu',weights_only=True)['state_dict'] for p in folders]
            for key in weights[0]:self.assertTrue(torch.equal(weights[0][key],weights[1][key]))

    def test_source_grid_cannot_accept_unconverged_configurations(self):
        arrays,meta,labels=v2_tests.V2Tests().fixture();selection=select_features(arrays,meta,labels)
        def fake(cells,state,hidden,history,seed,folder,context):
            return dict(checkpoint_sha256='mock',converged=False,validation_mse=.001,elapsed_seconds=0.)
        with tempfile.TemporaryDirectory() as directory, patch.dict(PROTOCOL,{'hidden_sizes':[8],'history_lengths':[1]}), \
                patch('experiments.mixed95.c3_che_v2_model.fit_source_run',side_effect=fake):
            with self.assertRaisesRegex(ValueError,'failed convergence'):
                train_grid(arrays,selection,Path(directory),'test')

    def test_grid_selection_uses_three_seed_source_validation_only(self):
        arrays,meta,labels=v2_tests.V2Tests().fixture();selection=select_features(arrays,meta,labels)
        def fake(cells,state,hidden,history,seed,folder,context):
            return dict(checkpoint_sha256='mock',converged=True,
                        validation_mse=.001 if hidden==16 else .01,elapsed_seconds=0.)
        with tempfile.TemporaryDirectory() as directory, patch.dict(PROTOCOL,{'hidden_sizes':[8,16],'history_lengths':[1]}), \
                patch('experiments.mixed95.c3_che_v2_model.fit_source_run',side_effect=fake):
            chosen=train_grid(arrays,selection,Path(directory),'test')
            self.assertEqual(chosen['hidden_size'],16)
            self.assertEqual(len(chosen['runs']),3)
            self.assertTrue(all(r['converged'] for r in chosen['runs']))


@unittest.skipUnless((BASE/DEFAULT_SUITE/'reconstruction.json').exists(),'v2 outputs not yet complete')
class IndependentArtifactReplayTests(unittest.TestCase):
    def test_numpy_replay_all_folds_all_404_cycles(self):
        out=BASE/DEFAULT_SUITE;arrays,meta=load_candidates(out)
        for context in [*(f'holdout_{g[0]}' for g in GROUPS),'full']:
            r=read(out/context/'calibration.json');s=read(out/context/'selection.json')
            state=r['source_transform']
            x=(arrays['target_x'][:,s['selected_indices']]-state['mean'])/state['scale']
            # Match the documented FP32 feature sequence -> FP64 model input.
            x=sequences(x,arrays['target_cycles'],r['history']).astype(np.float64)
            with np.load(out/context/'predictions.npz') as d:expected=d['predictions'].copy()
            for k,record in enumerate(r['per_seed']):
                weights=torch.load(out/context/f"C3_seed_{record['seed']}.pt",map_location='cpu',weights_only=True)['state_dict']
                w={key:value.numpy() for key,value in weights.items()}
                h=np.zeros((len(x),r['hidden_size']));c=h.copy()
                sigmoid=lambda a:1/(1+np.exp(-a))
                for step in range(x.shape[1]):
                    gates=x[:,step]@w['kernel'].T+h@w['recurrent_kernel'].T+w['bias']
                    ig,fg,candidate,og=np.split(gates,4,axis=1)
                    c=sigmoid(fg)*c+sigmoid(ig)*np.maximum(candidate,0)
                    h=sigmoid(og)*np.maximum(c,0)
                prediction=(h@w['dense.weight'].T+w['dense.bias']).ravel()
                np.testing.assert_allclose(prediction,expected[k],atol=1e-8,rtol=0)


if __name__=='__main__':unittest.main()
