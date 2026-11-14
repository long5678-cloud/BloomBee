"""Optional unsupported architecture must not prevent Llama registration."""
import builtins,types,warnings
from pathlib import Path
import pytest
SOURCE=Path(__file__).resolve().parents[1]/'src/bloombee/models/__init__.py'

@pytest.mark.parametrize('missing',[None,ImportError('missing Gemma4 output type',name='transformers.models.gemma4.modeling_gemma4'),ModuleNotFoundError('missing Gemma4',name='transformers.models.gemma4')])
def test_optional_gemma4_does_not_block_llama(missing):
    def imported(name,*args,**kwargs):
        if name=='bloombee.models.gemma4' and missing is not None:raise missing
        if name=='bloombee.utils.auto_config':
            return types.SimpleNamespace(register_unavailable_model=lambda *args:None)
        if name.startswith('bloombee.models.'):
            value=types.ModuleType(name);value.__all__=['registered_'+name.rsplit('.',1)[1]]
            setattr(value,value.__all__[0],True);return value
        return builtins.__import__(name,*args,**kwargs)
    scope={'__builtins__':dict(vars(builtins),__import__=imported)}
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter('always');exec(compile(SOURCE.read_text(),str(SOURCE),'exec'),scope)
    assert scope['registered_llama'] is True
    assert ('registered_gemma4' in scope)==(missing is None)
    assert len(emitted)==(missing is not None)

def test_unrelated_gemma4_dependency_error_is_not_swallowed():
    error=ImportError('broken unrelated dependency',name='numpy.core')
    def imported(name,*args,**kwargs):
        if name=='bloombee.models.gemma4':raise error
        if name=='bloombee.utils.auto_config':
            return types.SimpleNamespace(register_unavailable_model=lambda *args:None)
        if name.startswith('bloombee.models.'):
            value=types.ModuleType(name);value.__all__=[];return value
        return builtins.__import__(name,*args,**kwargs)
    scope={'__builtins__':dict(vars(builtins),__import__=imported)}
    with pytest.raises(ImportError) as caught:exec(compile(SOURCE.read_text(),str(SOURCE),'exec'),scope)
    assert caught.value is error


def test_requested_unavailable_gemma4_has_actionable_error(tmp_path,monkeypatch):
    from bloombee.utils import auto_config
    monkeypatch.delitem(auto_config._CLASS_MAPPING,'gemma4',raising=False)
    monkeypatch.setitem(auto_config._UNAVAILABLE_MODEL_REASONS,'gemma4','missing Gemma4TextModelOutputWithPast')
    monkeypatch.setattr(auto_config.AutoConfig,'from_pretrained',lambda *args,**kwargs:types.SimpleNamespace(model_type='gemma4'))
    with pytest.raises(ValueError,match='gemma4 is unavailable.*Transformers build.*Gemma4TextModelOutputWithPast'):
        auto_config.AutoDistributedModelForCausalLM.from_pretrained(str(tmp_path))
