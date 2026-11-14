from bloombee.models.bloom import *
from bloombee.models.falcon import *
from bloombee.models.llama import *
from bloombee.models.mixtral import *
from bloombee.models.deepseekv3 import *

try:
    from bloombee.models.gpt_oss import *
except ModuleNotFoundError as exc:
    if not str(exc.name).startswith("transformers.models.gpt_oss"):
        raise

from bloombee.models.nemotron_h import *

try:
    from bloombee.models.gemma4 import *
except ImportError as exc:
    # The supported minimum Transformers may contain Gemma4 but lack a
    # newer model-output type. Keep unrelated models usable; do not replace
    # Gemma4 classes or hide failures from unrelated dependencies.
    if not str(exc.name).startswith("transformers.models.gemma4"):
        raise
    from bloombee.utils.auto_config import register_unavailable_model
    register_unavailable_model("gemma4", str(exc))
    import warnings
    warnings.warn("Gemma4 registration unavailable in this Transformers build: " + str(exc), RuntimeWarning)

try:
    from bloombee.models.qwen3 import *
except ModuleNotFoundError as exc:
    if not str(exc.name).startswith("transformers.models.qwen3"):
        raise

try:
    from bloombee.models.glm4_moe_lite import *
except ModuleNotFoundError as exc:
    if not str(exc.name).startswith("transformers.models.glm4_moe_lite"):
        raise
