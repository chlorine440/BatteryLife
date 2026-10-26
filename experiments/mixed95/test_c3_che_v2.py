"""Numerical, data-isolation, and protocol regression tests for C3 v2."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.mixed95.c3_che_data import sequences, read, save
from experiments.mixed95.c3_che_model import encoder_hash
from experiments.mixed95.c3_che_v2_data import (
    PROTOCOL, GROUPS, BASE, DEFAULT_SUITE, windows, prepare_trace, window_q,
    scalar_features, grouped_values, select_features, selected_arrays,
    extract_candidates, load_candidates,
)
from experiments.mixed95.c3_che_v2_model import CapacityModel, svd_dense, train_grid
from experiments.mixed95.c3_che_v2 import suite_path, audit, select_all


class V2Tests(unittest.TestCase):
    def trace(self):
        v=np.linspace(2.8,4.2,1001)
        return np.c_[np.arange(len(v)),45800+np.arange(len(v))/86400,v,
                     np.full(len(v),90.),(v-2.8)*100]

    def fixture(self):
        x=np.linspace(1.,.8,180)
        arrays={'target_x':np.linspace(1.,.8,404)[:,None],
                'target_current_A':np.linspace(90,80,404), 'target_cycles':np.arange(1,405)}
        for c in range(4):
            arrays[f'c{c}_x']=x[:,None].copy();arrays[f'c{c}_soh']=x.copy()
            arrays[f'c{c}_cycles']=np.arange(1,181)
        meta={'descriptors':[dict(id='test:total',window='test',kind='total')]}
        labels=[dict(cycle=n,soh=float(arrays['target_x'][n-1,0])) for g in GROUPS for n in g]
        return arrays,meta,labels

    def test_window_enumeration_exact(self):
        w=windows();self.assertEqual(len(w),18)
        self.assertEqual(len(set(w)),18)
        self.assertTrue(all(3.2<=lo<hi<=4 and round((hi-lo)*10) in (2,3,4) for lo,hi in w))

    def test_record_order_timestamp_and_no_extrapolation(self):
        t=self.trace();t[400,1]=t[398,1];before=t.copy()
        q=window_q(prepare_trace(t,target=True),3.6,3.8)
        self.assertTrue((q>=0).all());np.testing.assert_array_equal(t,before)
        with self.assertRaisesRegex(ValueError,'covered'):window_q(prepare_trace(t[:500]),3.6,3.8)

    def test_inside_pause_rejected_outside_pause_allowed(self):
        t=self.trace();t[490,3]=0  # 3.486 V: before 3.6 but inside 3.4..3.7.
        trace=prepare_trace(t)
        window_q(trace,3.6,3.8)
        with self.assertRaisesRegex(ValueError,'interruption'):window_q(trace,3.4,3.7)

    def test_capacity_reset_is_not_repaired(self):
        t=self.trace();t[500,4]-=1
        with self.assertRaisesRegex(ValueError,'counter reset'):prepare_trace(t)

    def test_scalar_feature_definitions(self):
        q=np.array([[1.,2.,3.,4.],[2.,3.,4.,5.]])
        result=scalar_features(q,np.zeros(4),np.ones(4)/2)
        np.testing.assert_allclose(result[:,0],[10,14]);np.testing.assert_allclose(result[:,1],[2.5,3.5])
        np.testing.assert_allclose(result[:,3:6],np.quantile(q,[.25,.5,.75],axis=1).T)
        self.assertAlmostEqual(result[0,7],-sum(v*np.log(v) for v in [.1,.2,.3,.4]))
        np.testing.assert_allclose(result[:,-1],[5,7])

    def test_group_endpoint_equal_weight(self):
        a,m,labels=self.fixture();gx,gy=grouped_values(a['target_x'],labels)
        self.assertEqual(len(gy),5)
        self.assertAlmostEqual(gy[-1],(labels[-2]['soh']+labels[-1]['soh'])/2)
        with self.assertRaisesRegex(ValueError,'401/402'):grouped_values(a['target_x'],labels[:-1])

    def test_hidden_label_perturbation_has_no_selection_effect(self):
        a,m,labels=self.fixture()
        for hidden in GROUPS:
            cal=[r for r in labels if r['cycle'] not in hidden]
            first=select_features(a,m,cal)
            mutated=[dict(r,soh=1000) if r['cycle'] in hidden else dict(r) for r in labels]
            second=select_features(a,m,[r for r in mutated if r['cycle'] not in hidden])
            self.assertEqual(first,second);self.assertTrue(first['selected_indices'])

    def test_direction_mismatch_and_no_qualified_feature_stop(self):
        a,m,labels=self.fixture();a['c0_x']=-a['c0_x']
        r=select_features(a,m,labels)
        self.assertEqual(r['status'],'stopped_no_qualified_features')
        self.assertEqual(r['candidate_audit'][0]['reason'],'source/target correlation direction differs')
        with self.assertRaisesRegex(ValueError,'no qualified'):selected_arrays(a,r)

    def test_redundant_features_are_not_counted_twice(self):
        a,m,labels=self.fixture()
        for c in range(4):a[f'c{c}_x']=np.tile(a[f'c{c}_x'],(1,2))
        a['target_x']=np.tile(a['target_x'],(1,2))
        m['descriptors'].append(dict(id='test:mean',window='test',kind='mean'))
        r=select_features(a,m,labels)
        self.assertEqual(len(r['selected_indices']),1)
        self.assertTrue(any('redundant' in q['reason'] for q in r['candidate_audit']))

    def test_validation_and_target_not_in_scaler(self):
        a,m,labels=self.fixture();s=select_features(a,m,labels)
        _,first=selected_arrays(a,s)
        a['c3_x']*=100;a['c3_soh']*=.5;a['target_x']*=1000
        _,second=selected_arrays(a,s)
        self.assertEqual(first,second)
        self.assertEqual(first['n_training_cycles'],540)

    def test_varying_input_width_and_relu_equations(self):
        for dim in (1,2,3):
            model=CapacityModel(1,dim)
            with torch.no_grad():
                model.kernel.zero_();model.recurrent_kernel.zero_();model.bias.zero_()
                model.bias[2]=1;model.dense.weight.fill_(1);model.dense.bias.zero_()
                pred=model(torch.zeros(1,2,dim)).item()
            self.assertAlmostEqual(pred,.375)

    def test_svd_calibration_exact_and_frozen(self):
        rng=np.random.default_rng(3);z=rng.normal(size=(20,8));base=CapacityModel(8,2).double()
        truth=np.linspace(1.,.91,6);idx=[0,3,6,10,14,19];before=encoder_hash(base)
        head,pred,c=svd_dense(base,z,idx,truth)
        np.testing.assert_allclose(pred[idx],truth,atol=1e-11)
        self.assertEqual(encoder_hash(base),before);self.assertTrue(c['accepted']);self.assertEqual(c['rank'],6)
        with torch.no_grad():replay=head(torch.from_numpy(z)).numpy().ravel()
        np.testing.assert_allclose(replay,pred,atol=1e-12)

    def test_svd_rank_deficiency_and_nonzero_irreducible_error(self):
        base=CapacityModel(4,1).double();z=np.zeros((10,4));truth=[1.,.9,.8]
        _,pred,c=svd_dense(base,z,[0,4,9],truth)
        self.assertEqual(c['rank'],1);self.assertTrue(c['accepted'])
        np.testing.assert_allclose(pred,.9,atol=1e-12)
        self.assertGreater(c['after_rmse_pp'],0)

    def test_svd_minimum_source_weight_update_nullspace(self):
        base=CapacityModel(4,1).double();z=np.zeros((4,4))
        original=base.dense.weight.detach().numpy().copy()
        head,_,_=svd_dense(base,z,[0,3],[.98,.96])
        np.testing.assert_array_equal(head.weight.detach().numpy(),original)

    def test_calibration_rejects_nonfinite_labels(self):
        base=CapacityModel(2,1).double()
        with self.assertRaisesRegex(ValueError,'nonfinite'):svd_dense(base,np.ones((4,2)),[0,3],[1.,np.nan])

    def test_causal_history_resets_at_gaps(self):
        x=np.arange(12).reshape(6,2);cycles=[1,2,4,5,6,7]
        seq=sequences(x,cycles,3)
        np.testing.assert_array_equal(seq[2],np.repeat(x[2:3],3,axis=0))
        changed=x.copy();changed[-1]=1000
        np.testing.assert_array_equal(seq[:-1],sequences(changed,cycles,3)[:-1])

    def test_suite_path_cannot_target_old_artifacts(self):
        self.assertEqual(suite_path(DEFAULT_SUITE),BASE/DEFAULT_SUITE)
        for value in ('../bad_v2','c3_che_20261026_v1','/tmp/test_v2'):
            with self.assertRaises(ValueError):suite_path(value)


@unittest.skipUnless((BASE/DEFAULT_SUITE/'candidates.json').exists(),'candidate artifacts not generated')
class CandidateArtifactTests(unittest.TestCase):
    def test_candidate_pca_is_source_only_and_404_covered(self):
        arrays,meta=load_candidates(BASE/DEFAULT_SUITE)
        self.assertEqual(arrays['target_x'].shape[0],404)
        self.assertTrue(np.isfinite(arrays['target_x']).all())
        self.assertFalse(meta['target_labels_used_for_extraction'])
        self.assertTrue(all(len(s['fitted_cells'])==3 for s in meta['pca'].values()))

    def test_all_six_selections_hidden_label_isolation(self):
        arrays,meta=load_candidates(BASE/DEFAULT_SUITE)
        for hidden in (*GROUPS,()):
            calibration=[r for r in meta['rpt'] if r['cycle'] not in hidden]
            s=select_features(arrays,meta,calibration)
            self.assertEqual(s['status'],'selected')
            self.assertFalse(set(hidden)&set(s['calibration_cycles']))
            self.assertLessEqual(len(s['selected_indices']),3)


@unittest.skipUnless((BASE/DEFAULT_SUITE/'reconstruction.json').exists(),'v2 reconstruction not generated')
class V2ArtifactTests(unittest.TestCase):
    def test_full_audit_and_exactly_six_measured_points(self):
        out=BASE/DEFAULT_SUITE;checks=audit(out)
        self.assertTrue(checks['hidden_label_perturbation'])
        r=read(out/'reconstruction.json')
        self.assertEqual(len(r['rows']),404)
        self.assertEqual(sum(q['measured_rpt_soh'] is not None for q in r['rows']),6)
        self.assertEqual([q['physical_cycle'] for q in r['rows'] if q['beyond_last_calibration_checkpoint']],[403,404])


if __name__=='__main__':unittest.main()
