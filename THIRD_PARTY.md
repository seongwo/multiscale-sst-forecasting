# Third-party code and attribution

MSF-ST and its SST experiment pipeline are the proposed research. Comparison models and imported layer implementations retain their upstream attribution and license terms. This file does not assign a blanket license to the mixed-origin repository.

| Component | Local source | License text |
| --- | --- | --- |
| [OpenSTL](https://github.com/chengtan9907/OpenSTL) | Adapted recurrent/SimVP models and `model/modules/` | [Apache-2.0](licenses/OpenSTL.txt) |
| [HorNet](https://github.com/raoyongming/HorNet) | `model/modules/layers/hornet.py` | [MIT](licenses/HorNet.txt) |
| [MogaNet](https://github.com/Westlake-AI/MogaNet) | `model/modules/layers/moganet.py` | [Apache-2.0](licenses/MogaNet.txt) |
| [PoolFormer](https://github.com/sail-sg/poolformer) | `model/modules/layers/poolformer.py` | [Apache-2.0](licenses/PoolFormer.txt) |
| [UniFormer](https://github.com/Sense-X/UniFormer) | `model/modules/layers/uniformer.py` | [Apache-2.0](licenses/UniFormer.txt) |
| [VAN](https://github.com/Visual-Attention-Network/VAN-Classification) | `model/modules/layers/van.py` | [Apache-2.0](licenses/VAN.txt) |
| [Earthformer](https://github.com/amazon-science/earth-forecasting-transformer) | Unmodified core in `model/vendor/earthformer/` | [Apache-2.0](model/vendor/earthformer/LICENSE), [NOTICE](model/vendor/earthformer/NOTICE) |

Original notices, including CAIRI AI Lab notices, are retained. Adapted files carry a source/alteration notice. Retrieved license source URLs and Git blob identifiers are in [licenses/sources.json](licenses/sources.json); the complete MIT copyright/permission text and Apache license texts accompany the corresponding code. Any upstream NOTICE retrieved with a component is kept alongside its license.

## Added model sources

Earthformer is pinned to commit `7732b03bdb366110563516c3502315deab4c2026`. Required core source files are unchanged; their hashes are in [source.json](model/vendor/earthformer/source.json). The separate SST layout adapter is `model/models/earthformer_model.py`.

PiFormer is pinned to commit `fc922dca8199eafe062e7b24cb09913aeaffc77c`. Redistribution permission for upstream source was not established, so no upstream PiFormer code is included. The SST adapter loads and verifies the required files from an external official checkout. See [setup instructions](docs/additional_models.md#piformer).

The proposed MSF-ST model, its SST code and the torchvision-based ViT-GRU implementation remain separately identified. The repository does not redistribute torchvision/timm packages or their weights; those dependencies retain their own licenses. Third-party paper PDFs, datasets, external checkpoints and raw prediction arrays are not included in this release.
