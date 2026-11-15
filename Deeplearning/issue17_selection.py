"""Objective-based archive selection, distinct from post-hoc common-error selection."""
import numpy as np


def selection_curve(records,weight):
    if weight not in (1.,.5) or not records:raise ValueError('Expected nonempty records and lambda=1 or .5')
    output=[];selected=common=None
    for index,original in enumerate(records):
        p=dict(original)
        components=p['acoustic_components']
        objective=1e6 if p['failed'] else components['log_spectral']+weight*components['envelope']
        if not np.isfinite(objective) or not np.isfinite(p['acoustic_error']):raise FloatingPointError('Nonfinite candidate')
        p.update(weighted_loss=float(objective),envelope_weight=float(weight))
        if selected is None or objective<output[selected]['weighted_loss']:selected=index
        if common is None or p['acoustic_error']<output[common]['acoustic_error']:common=index
        # Strict comparisons mean ties are resolved by first observation, never
        # by quietly choosing the tied candidate with a better common metric.
        jpoint=p if selected==index else output[selected]
        epoint=p if common==index else output[common]
        p.update(objective_selected_index=selected,objective_selected_trial=jpoint['episodes'],
                 objective_selected_loss=jpoint['weighted_loss'],
                 objective_selected_common_error=jpoint['acoustic_error'],
                 common_posthoc_index=common,common_posthoc_trial=epoint['episodes'],
                 common_posthoc_error=epoint['acoustic_error'])
        output.append(p)
    return output
