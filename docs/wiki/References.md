# References and attribution

Each source supports a specific mechanism; its reported gains are not results
from this repository.

- [Burn 0.22 migration](https://burn.dev/books/burn/migrating-to-0.22.html): runtime
  device selection and current tensor API. Burn is the tensor substrate.
- [BitNet b1.58 2B4T](https://arxiv.org/abs/2504.12285): trained ternary model recipe;
  not evidence that post-hoc ternarizing an arbitrary dense model is lossless.
- [Muon, Keller Jordan](https://kellerjordan.github.io/posts/muon/): matrix momentum
  update orthogonalization. Attribute source and implementation adaptations.
- [Muon scaling](https://arxiv.org/abs/2502.16982): update scaling, weight decay,
  and distributed implementation are existing research topics.
- [DMuon](https://arxiv.org/abs/2606.27153): distributed optimizer prior work.
- [MuonQ](https://arxiv.org/abs/2605.11396): optimizer-state quantization;
  distinguish this from ternary model weights.
- [Ternary NeoMuon recipe](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-24_74M_Ternary_UNet_FP8_10L_8192BPE_YaRN_NeoMuon/README.md):
  existing ternary training combination on different hardware and objectives.
- [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/): established state
  partitioning/offload ideas; our future scheduler needs its own measured benefit.
- [GitHub rulesets](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets):
  required checks, branches, and integration-bound check sources.
- [GitHub App tokens](https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/generating-an-installation-access-token-for-a-github-app):
  repository/permission narrowing and expiring installation tokens.
- [GitHub Wiki publishing](https://docs.github.com/en/communities/documenting-your-project-with-wikis/adding-or-editing-wiki-pages):
  separate wiki Git repository and initialization requirements.
