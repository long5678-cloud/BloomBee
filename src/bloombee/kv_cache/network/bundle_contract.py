"""Content-bound page-bundle metadata and bounded local handoff framing."""
import base64,hashlib,json,struct

FORMAT='bb_joint_page_bundle_v1'
HEADER_LIMIT=1024**2

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def sha(blob):return hashlib.sha256(blob).hexdigest()
def identity_dict(identity):return identity.to_dict() if hasattr(identity,'to_dict') else dict(identity)
def value_sha(tensor):
    import torch
    raw=tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()

def layout_metadata(layout):
    if layout.block_size!=16 or layout.blocks!=1 or layout.logical_layers!=32 or layout.storage_layers!=64 or tuple(layout.shape)!=(2,64,16,4096):raise ValueError('frozen joint scope requires 32 model layers /64 storage slots, one16token page')
    return dict(model_name=layout.model_name,cache_salt=layout.cache_salt,tokens=layout.tokens,block_size=layout.block_size,logical_layers=layout.logical_layers,storage_layers=layout.storage_layers,shape=list(layout.shape),dtype=str(layout.dtype),identities_by_block=[[identity_dict(i) for i in row] for row in layout.identities_by_block])

def object_id(manifest):
    return digest({k:manifest[k] for k in ('format','layout','sha_by_block','native_value_sha256','wire_bytes_by_block')})

def make_manifest(layout,blobs,hashes,native,*,source_attempt_id,run_id,source_instance):
    layout_metadata(layout);layout._check_grid(blobs);layout._check_grid(hashes)
    for row,shas in zip(blobs,hashes):
        for blob,expected in zip(row,shas):
            if not isinstance(blob,bytes) or sha(blob)!=expected:raise ValueError('source object SHA mismatch')
    manifest=dict(format=FORMAT,layout=layout_metadata(layout),sha_by_block=[list(row) for row in hashes],wire_bytes_by_block=[[len(blob) for blob in row] for row in blobs],native_value_sha256=value_sha(native),native_tensor_bytes=native.numel()*native.element_size(),source_attempt_id=source_attempt_id,run_id=run_id,source_instance=source_instance)
    manifest['object_id']=object_id(manifest);return manifest

def validate_manifest(manifest,layout=None):
    if manifest.get('format')!=FORMAT or manifest.get('object_id')!=object_id(manifest):raise ValueError('full bundle object digest mismatch')
    info=manifest['layout']
    if (info['logical_layers'],info['storage_layers'],info['shape'])!=(32,64,[2,64,16,4096]):raise ValueError('model/storage layer geometry mismatch')
    if layout is not None and layout_metadata(layout)!=info:raise ValueError('codec namespace/full page identity differs from manifest')
    if len(manifest['sha_by_block'])!=1 or len(manifest['sha_by_block'][0])!=32:raise ValueError('incomplete original page SHA grid')
    return manifest

def ticket_header(manifest,ticket_id,expires_at,transfer,decision_link):
    validate_manifest(manifest)
    return dict(version=1,ticket_id=ticket_id,selection_id=ticket_id,object_id=manifest['object_id'],model_name=manifest['layout']['model_name'],cache_salt=manifest['layout']['cache_salt'],identities_by_block=manifest['layout']['identities_by_block'],sha_by_block=manifest['sha_by_block'],tokens=manifest['layout']['tokens'],block_size=manifest['layout']['block_size'],expires_at=expires_at,transfer=transfer,native_value_sha256=manifest['native_value_sha256'],source_manifest=manifest,wire_bytes_by_block=manifest['wire_bytes_by_block'],decision_link=decision_link)

def frame_parts(header,blobs):
    raw=json.dumps(header,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
    if len(raw)>HEADER_LIMIT:raise ValueError('handoff metadata exceeds bound')
    return [struct.pack('>I',len(raw)),raw,*[blob for row in blobs for blob in row]]

def unpack_ticket_frame(data,*,max_bytes=64*1024**2):
    if len(data)>max_bytes+HEADER_LIMIT+4 or len(data)<4:raise ValueError('local handoff frame exceeds bound')
    n=struct.unpack('>I',data[:4])[0]
    if not 0<n<=HEADER_LIMIT or n+4>len(data):raise ValueError('truncated handoff header')
    header=json.loads(data[4:4+n]);manifest=validate_manifest(header['source_manifest'])
    if header['model_name']!=manifest['layout']['model_name'] or header['cache_salt']!=manifest['layout']['cache_salt'] or header['native_value_sha256']!=manifest['native_value_sha256']:raise ValueError('handoff namespace/value hash mismatch')
    if header['object_id']!=manifest['object_id'] or header['ticket_id']!=header['selection_id']:raise ValueError('ticket identity mismatch')
    for key in ('identities_by_block','sha_by_block','tokens','block_size'):
        expected=manifest[key] if key in manifest else manifest['layout'][key]
        if header[key]!=expected:raise ValueError('handoff header differs from full source bundle')
    at=4+n;grid=[];total=0
    for sizes,hashes in zip(manifest['wire_bytes_by_block'],manifest['sha_by_block']):
        if len(sizes)!=32 or len(hashes)!=32:raise ValueError('incomplete page frame')
        row=[]
        for size,expected in zip(sizes,hashes):
            if type(size)is not int or size<=0 or at+size>len(data):raise ValueError('bad page length')
            blob=bytes(data[at:at+size]);at+=size;total+=size
            if total>max_bytes or sha(blob)!=expected:raise ValueError('local handoff object checksum/budget failure')
            row.append(blob)
        grid.append(row)
    if at!=len(data):raise ValueError('trailing handoff payload')
    return header,grid

def page_ready_payload(header,blobs,*,attempt_id,runtime_pid,clock_domain):
    """A-local provider control copy; not a B-to-A network receipt."""
    return dict(version=1,phase='prepare',attempt_id=attempt_id,object_id=header['object_id'],selection_id=header['selection_id'],identities_by_block=header['identities_by_block'],tokens=header['tokens'],block_size=header['block_size'],sha_by_block=header['sha_by_block'],blobs_b64_by_block=[[base64.b64encode(blob).decode('ascii') for blob in row] for row in blobs],expires_at=header['expires_at'],runtime_pid=runtime_pid,clock_domain=clock_domain,transfer=header['transfer'])
