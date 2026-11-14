import hashlib,json
from pathlib import Path
import pytest
from bloombee.flexgen_utils.conversion_reuse import configured_conversion,_verified,_identity

def fixture(tmp_path,monkeypatch):
    src=tmp_path/'model';src.mkdir();(src/'config.json').write_text('{}')
    root=tmp_path/'converted';out=root/'model-np';out.mkdir(parents=True)
    for name in ['embed_tokens.weight','.bloombee_np_converted']:(out/name).write_bytes(b'original')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    source_hashes={'config.json':sha(src/'config.json')}
    converted_hashes={str(p.relative_to(root)):sha(p) for p in out.iterdir()}
    att=tmp_path/'att.json';att.write_text(json.dumps(dict(model_id='model',revision='a'*40,files=source_hashes)))
    binding=root/'binding.json';binding.write_text(json.dumps(dict(schema=1,files=converted_hashes)))
    proof=tmp_path/'proof.json';proof.write_text(json.dumps(dict(schema=1,source_hashes=source_hashes,conversion_hashes=converted_hashes,
        source={name:_identity(src/name) for name in source_hashes},conversion={name:_identity(root/name) for name in converted_hashes},
        attestation_sha256=sha(att),binding_sha256=sha(binding))))
    for key,value in dict(BLOOMBEE_PRECONVERTED_ROOT=root,BLOOMBEE_ASSET_REUSE_RECEIPT=proof,BLOOMBEE_SOURCE_ATTESTATION=att).items():monkeypatch.setenv(key,str(value))
    _verified.cache_clear();return src,out,proof

def test_reuses_verified_immutable_assets_without_conversion(tmp_path,monkeypatch):
    src,out,_=fixture(tmp_path,monkeypatch)
    assert configured_conversion(str(src),'a'*40)==str(out)

def test_changed_conversion_is_not_silently_reconverted(tmp_path,monkeypatch):
    src,out,_=fixture(tmp_path,monkeypatch);(out/'embed_tokens.weight').write_bytes(b'changed')
    with pytest.raises(ValueError,match='asset changed'):configured_conversion(str(src),'a'*40)

def test_wrong_revision_and_partial_settings_are_rejected(tmp_path,monkeypatch):
    src,_,_=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='revision'):configured_conversion(str(src),'b'*40)
    monkeypatch.delenv('BLOOMBEE_SOURCE_ATTESTATION')
    with pytest.raises(ValueError,match='all three'):configured_conversion(str(src),'a'*40)
