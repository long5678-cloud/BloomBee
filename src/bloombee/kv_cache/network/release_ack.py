"""Verify a server release against the owner identity frozen at terminal."""
import math
import re
import time


class ReleasePending(TimeoutError):
    """The same terminal can resume without replaying the model."""


def _validate_returned_pages(owner, attempt, pid, handles):
    if (owner.get('status') != 'released' or owner.get('attempt_id') != attempt or
            type(owner.get('runtime_pid')) is not int or owner['runtime_pid'] != pid):
        raise ValueError('release owner has no matching released receipt')
    counts = owner.get('page_counts')
    digests = owner.get('slot_generation_sha256')
    if (not isinstance(counts, list) or len(counts) != len(handles) or
            not isinstance(digests, list) or len(digests) != len(handles)):
        raise ValueError('expected one page count and one slot generation digest per handle')
    # The service emits these parallel arrays in handle order before normalization.
    for handle, count, digest in zip(handles, counts, digests):
        if type(count) is not int or count <= 0:
            raise ValueError(f'invalid released page count for handle {handle}')
        if not isinstance(digest, str) or re.fullmatch(r'[0-9a-f]{64}', digest) is None:
            raise ValueError(f'invalid slot generation SHA256 for handle {handle}')


def identity(value, attempt, *, released=False):
    if not isinstance(value, dict) or value.get('attempt_id') != attempt:
        raise ValueError('release attempt mismatch')
    pid, clock = value.get('runtime_pid'), value.get('clock_domain')
    if type(pid) is not int or pid <= 0 or not isinstance(clock, str) or not clock:
        raise ValueError('release runtime identity missing')
    owners = value.get('owners')
    if not isinstance(owners, list) or not owners:
        raise ValueError('release owners missing')
    result = []
    used = set()
    layers = set()
    for owner in owners:
        if not isinstance(owner, dict):
            raise ValueError('release owner must be an object')
        instance = owner.get('runtime_instance')
        generation = owner.get('allocation_generation')
        handles = owner.get('handles')
        uids = owner.get('backend_uids')
        if not isinstance(instance, str) or not instance or type(generation) is not int:
            raise ValueError('invalid release owner')
        if (not isinstance(handles, list) or not handles or
                any(type(h) is not int or h < 0 for h in handles) or
                generation != min(handles) or len(set(handles)) != len(handles)):
            raise ValueError('invalid allocation generation or handles')
        if (not isinstance(uids, list) or not uids or
                any(not isinstance(uid, str) or not uid for uid in uids) or
                len(set(uids)) != len(uids) or layers.intersection(uids)):
            raise ValueError('invalid or duplicate backend identity')
        if len(handles) != len(uids):
            raise ValueError('release owner handle/backend count mismatch')
        if released:
            _validate_returned_pages(owner, attempt, pid, handles)
        keys = {(instance, h) for h in handles}
        if used.intersection(keys):
            raise ValueError('duplicate release handles')
        used.update(keys)
        layers.update(uids)
        result.append((instance, generation, tuple(sorted(zip(handles, uids)))))
    return pid, clock, tuple(sorted(result))


def wait_released(attempt, expected, reservation, post, budget):
    frozen = identity(expected, attempt)
    if (reservation.get('attempt_id') != attempt or
            reservation.get('runtime_pid') != frozen[0] or
            reservation.get('clock_domain') != frozen[1]):
        raise ValueError('release terminal differs from demand reservation')
    if type(budget) not in (int, float) or not math.isfinite(budget) or budget <= 0:
        raise ValueError('release_ack_timeout_s must be positive and finite')
    deadline = time.monotonic() + budget
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ReleasePending('release acknowledgement timed out')
        try:
            ack = post(attempt, remaining)
        except TimeoutError as error:
            raise ReleasePending('release acknowledgement transport timeout') from error
        if (not isinstance(ack, dict) or ack.get('attempt_id') != attempt or
                type(ack.get('runtime_pid')) is not int or ack['runtime_pid'] != frozen[0] or
                ack.get('clock_domain') != frozen[1]):
            raise ValueError('release response identity mismatch')
        if time.monotonic() >= deadline:
            raise ReleasePending('release acknowledgement arrived after timeout')
        status = ack.get('status')
        if status == 'released':
            if identity(ack, attempt, released=True) != frozen:
                raise ValueError('release owner differs from terminal')
            return ack
        if status not in ('pending', 'unknown'):
            raise ValueError('release not confirmed: ' + str(status))
        time.sleep(min(0.05, max(0, deadline - time.monotonic())))
