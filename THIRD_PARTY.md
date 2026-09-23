# Third-party code and license status

The proposed MSF-ST research and its SST experiment design should be distinguished from the baseline implementations included for comparison.

`model/models/__init__.py` and `model/modules/__init__.py` retain CAIRI AI Lab copyright notices. Several baseline components are associated with the OpenSTL codebase. The following source links are also present in the retained layer files:

| Local file | Source identified in the file |
|---|---|
| `model/modules/layers/hornet.py` | [HorNet](https://github.com/raoyongming/HorNet) |
| `model/modules/layers/moganet.py` | [MogaNet](https://github.com/Westlake-AI/MogaNet/blob/main/models/moganet.py) |
| `model/modules/layers/poolformer.py` | [PoolFormer](https://github.com/sail-sg/poolformer/blob/main/models/poolformer.py) |
| `model/modules/layers/uniformer.py` | [UniFormer](https://github.com/Sense-X/UniFormer/blob/main/image_classification/models/uniformer.py) |
| `model/modules/layers/van.py` | [Visual Attention Network](https://github.com/Visual-Attention-Network/VAN-Classification) |

This is an inventory of visible attribution, not a complete origin or license audit. Exact upstream revisions, applicable license texts, and any additional attribution requirements still need to be recorded for redistributed baseline files. Existing notices are preserved.

No repository-wide LICENSE file is currently present. The earlier `license: MIT` citation field has been removed because a citation entry alone does not establish the licensing of this mixed-origin code. No replacement blanket license is asserted here. Dataset terms and external model/artifact terms must be recorded separately.
