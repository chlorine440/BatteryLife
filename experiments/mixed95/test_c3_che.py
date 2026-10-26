"""Independent numerical/data-isolation checks for the neural reconstruction."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.mixed95.c3_che_data import (
    OUT, WINDOW, TRAIN_CELLS, VAL_CELL, charge_increments, fit_indicators,
    transform_indicators, sequences,
)
from experiments.mixed95.c3_che_model import ReLULSTMCapacity, encoder_hash, fit_dense, CONFIG
from experiments.mixed95.c3_che import GROUPS, metrics, crossing, audit_artifacts, reconstruct


class NeuralReconstructionTests(unittest.TestCase):
    def trace(self):
        v=np.linspace(2.8,4.2,1001)
        return np.c_[np.arange(len(v)),45800+np.arange(len(v))/86400,
                     v,np.full(len(v),90.),(v-2.8)*100]

    def test_record_order_corrects_one_second_time_typo_without_capacity_edit(self):
        a=self.trace();a[400,1]=a[398,1]
        original=a.copy();q,stats=charge_increments(a,target=True)
        np.testing.assert_allclose(q,np.diff(WINDOW)*100,atol=1e-10)
        self.assertEqual(stats['timestamp_reversals'],1)
        np.testing.assert_array_equal(a,original)

    def test_old_pause_outside_window_is_not_fake_capacity_reset(self):
        a=self.trace();a[300,3]=0
        q,stats=charge_increments(a,target=True)
        self.assertEqual(stats['noncharging_records_outside_window'],1)
        np.testing.assert_allclose(q,np.diff(WINDOW)*100,atol=1e-10)

    def test_pause_inside_window_is_rejected(self):
        a=self.trace();a[700,3]=0
        with self.assertRaisesRegex(ValueError,'interrupted'):charge_increments(a,target=True)

    def test_large_time_reversal_is_rejected(self):
        a=self.trace();a[400,1]-=10/86400
        with self.assertRaisesRegex(ValueError,'timestamp reversal'):charge_increments(a,target=True)

    def test_no_counter_repair_or_voltage_extrapolation(self):
        a=self.trace();a[600,4]-=3
        with self.assertRaisesRegex(ValueError,'counter reset'):charge_increments(a)
        with self.assertRaisesRegex(ValueError,'not covered'):charge_increments(self.trace()[:600])

    def test_conflicting_duplicate_is_rejected(self):
        a=self.trace();extra=a[500].copy();extra[4]+=1
        with self.assertRaisesRegex(ValueError,'conflicting'):charge_increments(np.r_[a,extra[None]])

    def test_hi_fit_does_not_use_target(self):
        q=np.array([[1.,2.,3.],[1.1,2.3,3.4],[1.4,2.7,4.1],[1.3,2.4,3.7]])
        state=fit_indicators(q);original=copy.deepcopy(state)
        x=transform_indicators(q,state)
        np.testing.assert_allclose(x.mean(axis=0),0,atol=1e-12)
        transform_indicators(q*50,state)
        self.assertEqual(state,original)

    def test_sequence_is_causal_and_does_not_bridge_missing_cycle(self):
        x=np.arange(18).reshape(6,3)
        original=sequences(x,[1,2,3,5,6,7],3)
        changed=x.copy();changed[-1]=10000
        updated=sequences(changed,[1,2,3,5,6,7],3)
        np.testing.assert_array_equal(original[:-1],updated[:-1])
        np.testing.assert_array_equal(original[3],np.repeat(x[3:4],3,axis=0))
        np.testing.assert_array_equal(original[0],np.repeat(x[:1],3,axis=0))

    def test_relu_lstm_independent_two_step_hand_calculation(self):
        model=ReLULSTMCapacity(1)
        with torch.no_grad():
            model.kernel.zero_();model.recurrent_kernel.zero_();model.bias.zero_()
            model.bias[2]=1.;model.dense.weight.fill_(1.);model.dense.bias.zero_()
        value=float(model(torch.zeros(1,2,3)).detach())
        # First c=.5,h=.25; second c=.75,h=.375 with all gates=.5.
        self.assertAlmostEqual(value,.375,places=7)

    def test_only_dense_updated_and_hidden_labels_cannot_affect_fit(self):
        torch.manual_seed(1);base=ReLULSTMCapacity(4)
        x=torch.randn(8,3,3)
        with torch.no_grad():z=base.encode(x)
        before=encoder_hash(base);labels=np.linspace(1.,.92,8)
        with patch.dict(CONFIG,{'fine_tune_epochs':20}):
            head,pred,fit=fit_dense(base,z,[0,4,7],labels[[0,4,7]])
            labels[2]=1000
            _,replay,_=fit_dense(base,z,[0,4,7],labels[[0,4,7]])
        np.testing.assert_array_equal(pred,replay)
        self.assertEqual(before,encoder_hash(base))
        self.assertEqual(fit['training_indices'],[0,4,7])
        self.assertFalse(torch.equal(head.weight,base.dense.weight))

    def test_group_and_source_split_isolation(self):
        self.assertNotIn(VAL_CELL,TRAIN_CELLS)
        self.assertEqual(GROUPS[-1],(401,402))
        self.assertEqual(sorted(n for g in GROUPS for n in g),[1,101,201,301,401,402])

    def test_metric_units_and_threshold_boundary(self):
        m=metrics([1.,.95],[.99,.96])
        self.assertAlmostEqual(m['mae_pp'],1)
        self.assertAlmostEqual(m['rmse_pp'],1)
        self.assertAlmostEqual(m['mae_Ah'],1.83)
        self.assertEqual(crossing([1,2,3],[.96,.95,.94]),2)
        self.assertIsNone(crossing([1,2],[1.,.96]))

    def test_existing_target_weight_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);weight=folder/'C3_seed_2021.pt'
            weight.write_bytes(b'keep this existing artifact')
            with self.assertRaises(FileExistsError):reconstruct(folder)
            self.assertEqual(weight.read_bytes(),b'keep this existing artifact')


@unittest.skipUnless((OUT/'reconstruction.json').exists(),'neural artifacts not generated yet')
class NeuralArtifactTests(unittest.TestCase):
    def test_independent_metric_and_integrity_recheck(self):
        checks=audit_artifacts(OUT)
        self.assertTrue(checks['encoder_frozen'])

    def test_all_cycles_are_neural_estimates_not_relabelled_truth(self):
        r=json.loads((OUT/'reconstruction.json').read_text())
        self.assertTrue(r['source_pretraining'] and r['lstm_model'] and r['dense_only_finetuning'])
        self.assertFalse(r['suitable_as_measured_benchmark_truth'])
        self.assertFalse(r['target_label_div094'])
        self.assertEqual(sum(row['measured_rpt_soh'] is not None for row in r['rows']),6)
        self.assertTrue(all(row['label_kind']=='neural_estimate_not_measured_truth' for row in r['rows']))
        self.assertTrue(all(np.isfinite(row['estimated_soh_mean']) for row in r['rows']))
        self.assertEqual([row['physical_cycle'] for row in r['rows'] if row['beyond_last_calibration_checkpoint']],[403,404])

    def test_numpy_replay_is_independent_of_torch_model_forward(self):
        state=json.loads((OUT/'source_transform.json').read_text())
        with np.load(OUT/'target_features.npz') as data:
            x=sequences(transform_indicators(data['delta_q'],state),data['cycles'],CONFIG['history_cycles'])
        for seed in CONFIG['seeds']:
            checkpoint=torch.load(OUT/f'C3_seed_{seed}.pt',map_location='cpu',weights_only=True)
            weights={k:v.numpy() for k,v in checkpoint['state_dict'].items()}
            h=np.zeros((len(x),CONFIG['hidden_size']),dtype=np.float32);c=h.copy()
            sigmoid=lambda a:1/(1+np.exp(-a))
            for step in range(x.shape[1]):
                gates=x[:,step]@weights['kernel'].T+h@weights['recurrent_kernel'].T+weights['bias']
                ig,fg,candidate,og=np.split(gates,4,axis=1)
                c=sigmoid(fg)*c+sigmoid(ig)*np.maximum(candidate,0)
                h=sigmoid(og)*np.maximum(c,0)
            pred=(h@weights['dense.weight'].T+weights['dense.bias']).ravel()
            rows=json.loads((OUT/'reconstruction.json').read_text())['rows']
            expected=np.array([r[f'estimated_soh_seed_{seed}'] for r in rows])
            np.testing.assert_allclose(pred,expected,rtol=0,atol=2e-5)

    def test_source_partial_charge_less_than_full_discharge(self):
        with np.load(OUT/'source_features.npz') as d:
            for name in (*TRAIN_CELLS,VAL_CELL):
                key=name[:-4]
                self.assertTrue((d[key+'_q'].sum(axis=1)<=d[key+'_y']*1.05).all())


if __name__=='__main__':unittest.main()
