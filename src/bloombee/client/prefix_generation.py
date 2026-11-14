"""Explicit public adapter for ordinary distributed model.generate over real RPC."""
import uuid
import torch


class DistributedPrefixGenerator:
    def __init__(self, model, *, max_context, record=None):
        self.model, self.max_context, self.record = model, max_context, record
        sequence = getattr(getattr(model, 'transformer', None), 'h', None)
        if sequence is not None:
            config = sequence.sequence_manager.config
            if type(config.max_retries) is not int or config.max_retries < 1:
                raise ValueError('configure a finite positive max_retries on the distributed model')

    @torch.inference_mode()
    def generate(self, token_ids, blocks, *, mode='prefetch', request_id=None,
                 max_new_tokens, **generate_options):
        if (not token_ids or any(type(t) is not int or t < 0 for t in token_ids)
                or type(max_new_tokens) is not int or max_new_tokens <= 0
                or len(token_ids) + max_new_tokens > self.max_context):
            raise ValueError('input/output exceed explicit model context; no implicit truncation')
        if mode not in ('prefetch', 'demand', 'cold'):
            raise ValueError('unknown mode')
        if generate_options.get('do_sample', False):
            raise ValueError('finite deterministic integration uses greedy generation')
        if generate_options.get('num_beams', 1) != 1:
            raise ValueError('native prefix adapter requires batch1 greedy generation')
        request_id = request_id or uuid.uuid4().hex
        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.model.device)
        options = dict(generate_options)
        options.setdefault('do_sample', False)
        options.setdefault('num_beams', 1)
        failures = []
        for attempt_mode in ([mode, 'cold'] if mode != 'cold' else ['cold']):
            try:
                # Each attempt owns NEW native session handles. No partly imported cache is reused.
                with self.model.inference_session(max_length=len(token_ids) + max_new_tokens) as session:
                    session.remote_prefix = dict(request_id=request_id + '-' + attempt_mode,
                        token_ids=list(token_ids), blocks=list(blocks), mode=attempt_mode)
                    output = self.model.generate(input_tensor, session=session,
                        max_new_tokens=max_new_tokens, **options)
                receipt = dict(request_id=request_id, mode=attempt_mode,
                    input_length=len(token_ids), requested_output_length=max_new_tokens,
                    actual_output_length=int(output.shape[-1]) - len(token_ids), failures=failures,
                    transport='ordinary model.generate -> inference_session -> rpc_inference')
                if self.record:
                    self.record(receipt)
                return output, receipt
            except Exception as error:
                failures.append(dict(mode=attempt_mode, error_type=type(error).__name__))
                if attempt_mode == 'cold':
                    if self.record:
                        self.record(dict(request_id=request_id, failed=True, failures=failures))
                    raise
        raise RuntimeError('unreachable')


def validate_prefix_receipt(receipt, *, rpc_session_id, envelope, input_tokens, layer_start, layer_end):
    """Validate an explicit native-import RPC acknowledgment before accepting suffix output."""
    import hashlib
    import json
    if (not isinstance(receipt, dict) or receipt.get('schema') != 1
            or receipt.get('rpc_session_id') != rpc_session_id
            or receipt.get('request_id') != envelope.get('request_id')
            or not isinstance(receipt.get('native_session_id'), str) or not receipt['native_session_id']
            or receipt.get('input_tokens') != input_tokens
            or receipt.get('layer_start') != layer_start or receipt.get('layer_end') != layer_end
            or receipt.get('origin') not in ('prefetch', 'demand')
            or type(receipt.get('skipped_tokens')) is not int
            or not 0 < receipt['skipped_tokens'] < input_tokens):
        raise RuntimeError('invalid native prefix RPC receipt')
    tokens = envelope.get('token_ids')
    digest = hashlib.sha256(json.dumps(tokens, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if receipt.get('token_ids_digest') != digest or len(tokens) != input_tokens:
        raise RuntimeError('native prefix RPC receipt token identity mismatch')
    return dict(receipt)
