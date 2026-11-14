"""Instance-local HF generation-argument compatibility; no precision changes."""
def install_logit_argument_forwarding(model):
    original = model.prepare_inputs_for_generation
    def prepare(*args, **kwargs):
        inputs = original(*args, **kwargs)
        # Retain the documented API value, including explicit0 or index tensor.
        if 'logits_to_keep' in kwargs:
            inputs['logits_to_keep'] = kwargs['logits_to_keep']
        return inputs
    model.prepare_inputs_for_generation = prepare
    return dict(adapter='instance_prepare_preserves_explicit_logits_to_keep',
                ordinary_forward_default_unchanged=True, precision_changed=False)
