"""Public constructors for explicitly different retention lifecycles.

The factory selects an implementation; it does not load a model, allocate CUDA,
fetch a prefix or silently translate between byte and token budgets.
"""
import inspect
from copy import deepcopy

from .history import HistoryCache
from .placement import Placement
from .tiered import TieredCache



def _spec(name, aliases, family, selector, title, source, contract, validation, variants=()):
    return dict(name=name, aliases=list(aliases), family=family, selector=selector,
                title=title, source=source, data_contract=contract,
                experimental=family != "group-bytes" or name not in ("lru", "hotness", "decay", "remote-only"),
                validation=validation, variants=list(variants))


_BYTE_CONTRACT = dict(
    input="完整且兼容的32层Group；KVSLAB01序列化bytes；capacity_bytes是真实载荷字节",
    lifecycle=["begin(group, request_id)", "request.bind_fetcher(source)", "request.finish(reuse_success=bool)"],
    output="Demand与完成事件；命中、接纳、源取回字节和成功访问计数",
    protection="单活动请求；必须在真实消费成功后确认；源回调/模型工作内存不在保留预算内",
)
_TIER_CONTRACT = dict(
    input="完整不可变快照及兼容键；size/move/validate回调；fast_bytes与slow_bytes独立计量",
    lifecycle=["acquire(key, loader, priority=None)", "lease.value / lease.origin", "lease.finish(success)"],
    output="Lease；origin为0快层、1慢层、2源；完成返回admitted/origin",
    protection="保留对象不能原地续写；活动租约受保护；实际device取决于move回调",
)
_TOKEN_CONTRACT = dict(
    input="真实token IDs及完整expected_layers张量；time-major (tokens,...)；context绑定模型/执行兼容身份",
    lifecycle=["put_completed(ids, layers, now=...)", "get(ids, now=..., target_device=...)", "release() on a hit in finally"],
    output="get返回(layers或None, 可复用token数, 事件)；必须按返回长度重建导入身份",
    protection="块容量不等于字节；真实字节见resident_bytes；活动租约期间禁止put；仅保留成功完成的快照",
)
_WHOLE_PASS = dict(prior_model="既有完整32层/127-token研究接线通过；每配置24次；不等于当前源码的设备验收",
                   current_scope="本地公共接口与功能检查见随附测试/示例；未自动接入普通客户端")
_TIER_PASS = dict(prior_model="既有研究适配器每配置24次，旧完整gate通过；分层项有真实D2H/H2D与释放证据",
                  current_scope="本版是包内迁移；设备、字节和内容验证由调用方回调确定")
_TOKEN_LIMIT = dict(prior_model="既有48/112-token切片命中共45次未通过旧逐位logits gate；token及原容差通过；失败未修复",
                    current_scope="实验选项；动态层数/形状的CPU功能通过不自动证明任意模型或设备正确")
_DYNAMO_CONTRACT = dict(
    input="从请求起点累积的已计算token IDs及time-major张量；每块显式priority缺失为0；块容量",
    lifecycle=["put_completed(ids,layers,priorities=...,request_id=...,final=...)",
               "get(ids,target_device=...) -> layers,n,event", "命中后finally release()；取消用finish_request(id)"],
    output="独立CPU副本；实际连续可跳过长度；截断/复制/回收事件；GPU原块仍由推理端持有",
    protection="同request_id首次低于门槛永久停止后续分块卸载；仅闲置CPU块可回收；get不删除CPU副本",
)

_SPECS = (
    _spec("dynamo", ("Dynamo", "dynamo-native", "native-dynamo"), "dynamo-host-prefix", None,
          "Dynamo本体GPU/CPU规则的同步适配",
          "ai-dynamo/dynamo v1.6.2 @28f9765d98ded77149a94d0defee6d54e97f6741 的连续前缀门槛与闲置块回收规则；这些规则与初次验收的 v1.6.1 逐字节一致",
          _DYNAMO_CONTRACT,
          dict(scope="GPU/CPU配置子集；不是Rust/NIXL、pinned allocator或G3/G4完整服务",
               numerical="本选项的同轮冷算/严格门槛另行验证；旧45/61失败保留，不能改名为本体结果"),
          (dict(name="缺省门槛", parameters=dict(host_offload_min_priority=0)),
           dict(name="正门槛及相等边界", parameters=dict(host_offload_min_priority=50)),
           dict(name="高门槛", parameters=dict(host_offload_min_priority=80)))),
    _spec("lru", ("LRU", "least-recently-used"), "group-bytes", "lru", "原LRU",
          "现有Placement，不重写", _BYTE_CONTRACT, _WHOLE_PASS),
    _spec("hotness", ("window-frequency", "windowed-frequency"), "group-bytes", "hotness", "原窗口访问频次",
          "现有Placement；window_s窗口内成功访问计数", _BYTE_CONTRACT, _WHOLE_PASS),
    _spec("decay", ("decayed-frequency",), "group-bytes", "decay", "原衰减访问频次",
          "现有Placement；保留原60秒/每4次事件减半定义", _BYTE_CONTRACT, _WHOLE_PASS),
    _spec("remote-only", ("disabled", "no-retention"), "group-bytes", "remote-only", "关闭本地保留",
          "现有Placement控制项", _BYTE_CONTRACT,
          dict(prior_model="关闭保留控制项；不宣称本地缓存命中", current_scope="原接口兼容")),
    _spec("tiered-lru", ("gpu-lru", "tensor-lru"), "tiered-snapshot", "LRU", "快照LRU参照",
          "已测分层研究适配器的单层参照；device由move指定", _TIER_CONTRACT, _TIER_PASS),
    _spec("A", ("a", "tiered"), "tiered-snapshot", "A", "分层LRU",
          "借鉴Dynamo v1.6.1配套分层概念的研究适配，非运行NVIDIA服务", _TIER_CONTRACT, _TIER_PASS),
    _spec("B", ("b", "priority"), "tiered-snapshot", "B", "显式优先级，同级LRU",
          "TensorRT-LLM v1.3.0rc29保留优先级概念的完整前缀适配", _TIER_CONTRACT, _TIER_PASS),
    _spec("A+B", ("a+b", "ab", "tiered-priority"), "tiered-snapshot", "A+B", "分层加显式优先级",
          "A和B的组合；额外慢层容量明确分列", _TIER_CONTRACT, _TIER_PASS),
    _spec("lfu", ("C", "c", "cumulative-lfu"), "group-bytes", "lfu", "C 累计LFU",
          "累计成功访问频次，同频LRU；继承原Group/生命周期", _BYTE_CONTRACT, _WHOLE_PASS),
    _spec("slru", ("D", "d", "segmented-lru"), "group-bytes", "slru", "D 分段LRU",
          "试用/保护组研究变体，保护份额和晋升阈值显式可选", _BYTE_CONTRACT, _WHOLE_PASS),
    _spec("fifo", ("FIFO",), "group-bytes", "fifo", "先进先出",
          "按当前驻留批次的插入顺序", _BYTE_CONTRACT, _WHOLE_PASS),
    _spec("filo", ("FILO",), "group-bytes", "filo", "先进后出",
          "按当前驻留批次的反向插入顺序", _BYTE_CONTRACT, _WHOLE_PASS),
    _spec("mru", ("MRU",), "group-bytes", "mru", "最近使用者优先淘汰",
          "按已成功使用的最近顺序", _BYTE_CONTRACT, _WHOLE_PASS),
    _spec("E", ("e", "segment-priority"), "token-tensors", "E", "区段优先级与到期",
          "TensorRT-LLM v1.3.0rc29概念；显式已知且块对齐的RetentionRange", _TOKEN_CONTRACT, _TOKEN_LIMIT,
          (dict(name="默认同级", parameters=dict(e_ranges=[])),
           dict(name="在线提示", parameters="e_ranges=(RetentionRange(start,end,priority,duration_s),...)；范围不得超过已知前缀"))),
    _spec("F", ("f", "offload-threshold"), "token-tensors", "F", "整对齐前缀卸载门槛",
          "TensorRT-LLM v1.3.0rc29的secondary offload门槛概念；非token后缀淘汰", _TOKEN_CONTRACT, _TOKEN_LIMIT,
          (dict(name="门槛35", parameters=dict(threshold=35, priority=35)),
           dict(name="门槛60", parameters=dict(threshold=60, priority=35), prior_coverage="既有24次全部冷回退，零命中；不算命中复用通过"))),
    _spec("G", ("g", "suffix-first"), "token-tensors", "G", "同批释放后缀优先",
          "vLLM v0.31.0概念，附加叶子/前缀闭包保护的研究适配", _TOKEN_CONTRACT, _TOKEN_LIMIT),
    _spec("H", ("h", "tail-optimized-lru"), "token-tensors", "H", "Tail-optimized LRU",
          "SGLang v0.5.21公式适配；历史高水位/预算/增长估计显式给定，不读未来", _TOKEN_CONTRACT, _TOKEN_LIMIT),
)

# English public metadata; preserve implementations, aliases, budgets and options.
_TITLES = {
    'dynamo':'Dynamo-derived CPU prefix retention', 'lru':'Least recently used',
    'hotness':'Windowed successful-use frequency', 'decay':'Event-decayed successful-use frequency',
    'remote-only':'Disable serialized-group retention', 'tiered-lru':'Snapshot LRU reference',
    'A':'Tiered LRU research adapter', 'B':'Explicit priority with LRU ties',
    'A+B':'Tiered explicit priority', 'lfu':'Cumulative successful-use frequency',
    'slru':'Segmented LRU', 'fifo':'First in first out', 'filo':'First in last out',
    'mru':'Most recently used victim', 'E':'Explicit segment priority and expiry',
    'F':'Aligned-prefix offload threshold', 'G':'Suffix-first research adapter',
    'H':'Tail-optimized LRU research adapter',
}
_SOURCES = {
    'dynamo':'NVIDIA Dynamo KVBM v1.6.2 @28f9765d98ded77149a94d0defee6d54e97f6741; prefix threshold and idle-block rules match the previously audited v1.6.1 subset',
    'lru':'Original unchanged Placement', 'hotness':'Original Placement successful uses within window_s',
    'decay':'Original Placement: 60-second window, half-life four later successful reuse events',
    'remote-only':'Original Placement control',
    'A':'Dynamo v1.6.1 tiering concept; BloomBee research adapter, not NVIDIA runtime',
    'B':'TensorRT-LLM v1.3.0rc29 priority concept adapted to complete prefixes',
    'A+B':'Combined tier/priority research adapter; separate slow-tier budget',
    'E':'TensorRT-LLM v1.3.0rc29 explicit aligned RetentionRange concept',
    'F':'TensorRT-LLM v1.3.0rc29 secondary offload threshold concept',
    'G':'vLLM v0.31.0 concept with prefix/leaf protection',
    'H':'SGLang v0.5.21 formula with explicit prior high-water/growth assumptions',
}
_ENGLISH_CONTRACTS = {
 'group-bytes':dict(input='32 compatible ordered KVSLAB01 byte layers; capacity_bytes is payload bytes',
    lifecycle=['begin(group,request_id)','bind_fetcher(source)','finish(reuse_success=bool)'],
    output='Demand and completed-use receipt',protection='One active demand; real consumer confirms success; model working memory is separate'),
 'tiered-snapshot':dict(input='Immutable snapshots and compatible keys; size/move/validate callbacks; independent fast_bytes/slow_bytes',
    lifecycle=['acquire(key,loader,priority=None)','lease.value / lease.origin','lease.finish(success)'],
    output='Lease; origin 0=fast, 1=slow, 2=source',protection='Active leases protected; device determined by move callback'),
 'token-tensors':dict(input='Actual token IDs and complete time-major tensors; compatible execution context; block capacity',
    lifecycle=['put_completed(ids,layers)','get(ids,target_device=...)','release() in finally after hit'],
    output='layers or None, reusable token count, event',protection='No write during active lease; retain completed snapshots only; blocks are not bytes'),
 'dynamo-host-prefix':dict(input='Cumulative computed tokens; time-major tensors; priorities 0..100, missing=0; block capacity',
    lifecycle=['put_completed(ids,layers,priorities=...,request_id=...,final=...)','get/get_range','release/finish_request'],
    output='Independent CPU replica, contiguous reusable tokens and copy/reclaim events',
    protection='First below-threshold block stops later offload for that request; reclaim idle blocks only; read retains CPU copy'),
}
for _entry in _SPECS:
    _entry['title'] = _TITLES[_entry['name']]
    _entry['source'] = _SOURCES.get(_entry['name'], 'Historical BloomBee research adapter; implementation contract retained')
    _entry['data_contract'] = _ENGLISH_CONTRACTS[_entry['family']]
    _entry['validation'] = dict(current_scope='See current checkout validation; constructor does not run model acceptance',
        historical_scope='Historical research and model evidence is separate from this packaged revision',
        limitations=('Prior E/F/G/H slicing runs retained 45 strict-logit failures; not repaired by packaging'
                    if _entry['family']=='token-tensors' else 'CPU component tests do not certify arbitrary models or GPUs'))
    for _index,_variant in enumerate(_entry['variants']):
        _variant['name'] = 'Explicit configuration '+str(_index+1)
        if isinstance(_variant.get('parameters'),str):
            _variant['parameters']='Explicit RetentionRange tuples within the known aligned prefix'
        if 'prior_coverage' in _variant:
            _variant['prior_coverage']='Historical higher-threshold configuration had 24 cold fallbacks and zero reuse hits'

_ALIASES = {}
for _entry in _SPECS:
    for _alias in (_entry["name"], *_entry["aliases"]):
        _normalized = _alias.casefold()
        if _normalized in _ALIASES and _ALIASES[_normalized] is not _entry:
            raise RuntimeError("ambiguous cache option alias: " + _alias)
        _ALIASES[_normalized] = _entry


def _constructor(spec):
    if spec["family"] == "dynamo-host-prefix":
        from .dynamo import DynamoCache
        return DynamoCache
    if spec["family"] == "tiered-snapshot":
        return TieredCache
    if spec["family"] == "token-tensors":
        from .token_cache import TokenCache
        return TokenCache
    return Placement if spec["name"] in ("lru", "hotness", "decay", "remote-only") else HistoryCache


def create_cache(option, **kwargs):
    """Select an implementation and forward every constructor keyword intact.

    lru/LRU always selects the unchanged bytes Placement. tiered-lru/gpu-lru
    selects the snapshot API; the alias does not allocate or choose a GPU.
    Unknown options or unsupported/conflicting kwargs fail explicitly.
    """
    if not isinstance(option, str):
        raise TypeError("cache option must be a string")
    spec = _ALIASES.get(option.strip().casefold())
    if spec is None:
        raise ValueError("unknown cache option {!r}; choose one of {}".format(
            option, ", ".join(entry["name"] for entry in _SPECS)))
    if spec["family"] == "dynamo-host-prefix":
        from .dynamo import DynamoCache
        return DynamoCache(**kwargs)
    controlled = "method" if spec["family"] in ("tiered-snapshot", "token-tensors") else "policy"
    if controlled in kwargs:
        raise TypeError("{} is selected by option; do not also pass {}".format(controlled, controlled))
    return _constructor(spec)(spec["selector"], **kwargs)


def _default(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if callable(value):
        return value.__module__ + "." + value.__name__
    return repr(value)


def describe_options():
    """Return fresh JSON-serializable descriptors, signatures and limitations.

    This is capability/provenance metadata, not a claim that calling it ran
    tests. Prior model evidence is separated from current implementation scope.
    """
    result = []
    for spec in _SPECS:
        row = deepcopy(spec)
        constructor = _constructor(spec)
        row["implementation"] = constructor.__module__ + "." + constructor.__name__
        row["parameters"] = []
        parameters = list(inspect.signature(constructor).parameters.values())
        if spec["family"] != "dynamo-host-prefix":
            parameters = parameters[1:]
        for parameter in parameters:
            item = dict(name=parameter.name, required=parameter.default is inspect.Parameter.empty,
                        kind=parameter.kind.name)
            if not item["required"]:
                item["default"] = _default(parameter.default)
            row["parameters"].append(item)
        result.append(row)
    return result
