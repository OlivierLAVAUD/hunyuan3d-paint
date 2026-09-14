# Licensing

**This file describes what the repository does about licensing. It is not legal
advice, and it does not replace the license texts it points to.** Where this
summary and a license text disagree, the license text governs.

Two bodies of code live here, under two different licenses. The license of one
does not apply to the other.

## Scope

| Path | Work | License |
|---|---|---|
| `app/`, `ui/`, `scripts/`, `tests/`, `main.py`, the Dockerfiles | original | MIT — [`LICENSE`](LICENSE) |
| `third_party/Hunyuan3D-2/` | Tencent's `hy3dgen`, redistributed | Tencent Hunyuan 3D 2.0 Community License Agreement — [`third_party/Hunyuan3D-2/LICENSE`](third_party/Hunyuan3D-2/LICENSE) |
| the model checkpoints | not distributed here | same agreement; downloaded at first start |

The MIT license covers **only** the first row. It cannot cover the vendored code
— that code is not ours to relicense — and nothing in this repository claims
otherwise.

## The agreement that governs `third_party/`

The vendored `hy3dgen` package is covered by the **Tencent Hunyuan 3D 2.0
Community License Agreement** (Release Date: January 21, 2025). Its Section 1.j
defines "Tencent Hunyuan 3D 2.0" as including *"machine-learning model code,
inference-enabling code"* and the rest of what is published at
`https://github.com/Tencent/Hunyuan3D-2` — so the code is in scope, not just the
weights. The `LICENSE` and `NOTICE` files shipped inside
`third_party/Hunyuan3D-2/` are byte-for-byte upstream's.

Four things in it are worth knowing before you build on this repository.

**Commercial use is allowed.** The license is non-exclusive, non-transferable
and royalty-free for use, reproduction, modification and distribution. Section 4
adds a single threshold: if the monthly active users of all products or services
made available by or for you exceeded **1 million** in the preceding calendar
month, you must request a license from Tencent (`hunyuan3d@tencent.com`) and may
not exercise the rights in the agreement until they grant one.

**You keep your outputs.** Section 6.d: Tencent claims no rights in outputs you
generate, and Section 6.a makes you the owner of your own derivative works and
modifications. So the textured GLBs this service returns are yours.

**The territory is limited.** The agreement's own header states that it *"does
not apply in the European Union, United Kingdom and South Korea"*, Section 1.l
defines the licensed "Territory" as worldwide minus those three, and Section 5.c
provides that use, reproduction, modification, distribution or display outside
the Territory is *"unlicensed and unauthorized"*. Section 2 accordingly grants
its rights *"for the Territory only"*. If you are located in the EU, the UK or
South Korea, the agreement does not grant you anything, and the rest of this
file does not describe a right you hold — that is a question for Tencent or for
counsel, not for a README. Note also that publishing the code on a public host
distributes it worldwide, which sits awkwardly with a territory-limited grant
wherever you are.

**Two restrictions travel with the code.** Section 5.b forbids using the works
or their outputs to improve any other AI model (other than Hunyuan 3D 2.0 or its
own derivatives), and Section 5.a requires the use restrictions of Sections 5(a)
and 5(b) to be carried as an enforceable provision in any agreement governing
use or distribution of the works, with notice passed on to subsequent users.
That is why they are restated here rather than left to the links.

### A note on the per-file headers

The files under `hy3dgen/` carry a comment header reading *"Hunyuan 3D is
licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT"*. That text
is upstream's — verified unchanged against
`Tencent-Hunyuan/Hunyuan3D-2@main` — and it contradicts the repository-level
`LICENSE`, which is the 3D 2.0 Community License, not a non-commercial one. The
`LICENSE` is the agreement; the comment is a leftover from the earlier Hunyuan3D
release. If that discrepancy matters to you, raise it upstream and take advice:
it is Tencent's inconsistency, not something this repository introduced and not
something it can settle.

Tencent's `NOTICE` also lists components inside the distribution that carry
their own terms (Stable Diffusion under MIT and CreativeML Open RAIL++-M,
HunyuanDiT under the Tencent Hunyuan Community License Agreement). Those apply
on top, for the parts concerned.

## Files modified from upstream

Section 3.b of the agreement requires that modified files *"carry prominent
notices stating that You changed the files"*. Five files were changed; each
carries a modification notice immediately below its upstream header, and the
`third_party/Hunyuan3D-2/` root files (`LICENSE`, `NOTICE`, `README.md`,
`setup.py`, `requirements.txt`) are untouched. Every change is a small,
commented patch — mostly memory behaviour on a small GPU, plus the seed knob:

| File | Change |
|---|---|
| `hy3dgen/shapegen/pipelines.py` | checkpoint loading is device- and dtype-aware: weights map straight to CUDA, modules are instantiated in the checkpoint dtype, checkpoint tensors are freed before the move |
| `hy3dgen/texgen/pipelines.py` | device read from `HY3DGEN_TEXGEN_DEVICE`; `enable_sequential_cpu_offload()` added, which first turns the paint UNet's `learned_text_clip_gen` / `learned_text_clip_ref` parameters into buffers; `__call__` takes an optional `seed` |
| `hy3dgen/texgen/utils/multiview_utils.py` | optional `seed` instead of a hardcoded 0; the generator is built from `_execution_device`; the pipeline loads with `low_cpu_mem_usage=True` |
| `hy3dgen/texgen/utils/dehighlight_utils.py` | the delight pipeline loads with `low_cpu_mem_usage=True` |
| `hy3dgen/texgen/differentiable_renderer/mesh_render.py` | `set_default_render_resolution()` recomputes `bake_unreliable_kernel_size` |

Note that `hy3dgen/shapegen/` is not used by this service — the paint half only
needs `texgen/`. It is vendored because the package is imported as a whole. The
two `shapegen` changes stay for whoever vendors this tree next.

## What distribution requires, and what this repository does

| Section | Obligation | Status here |
|---|---|---|
| 3.a | give recipients of the works a copy of the agreement | [`third_party/Hunyuan3D-2/LICENSE`](third_party/Hunyuan3D-2/LICENSE), verbatim |
| 3.b | modified files carry a notice that they were changed | notice in each of the five headers, listed above |
| 3.c | *encouraged*: a public statement of use, and/or marking products "Powered by Tencent Hunyuan" | not done — it is optional |
| 3.d | distributions other than via a hosted service come with a `NOTICE` file containing a specific sentence | [`NOTICE`](NOTICE), with the sentence reproduced verbatim |
| 3.e | if you serve third parties with this, state who actually provides it and that Tencent is not affiliated | this is on you if you deploy it |
| 5.a | carry the Section 5(a)/5(b) use restrictions as an enforceable provision, and pass them on | restated in this file; add them to your own terms if you distribute |
| 5.b | do not use the works or their outputs to improve another AI model | a usage rule you must observe |
| 6.b | no trademark rights beyond customary descriptive use | "Tencent Hunyuan 3D" is used here to describe the vendored code, nothing more; no logos, no implied endorsement |

The agreement is governed by the law of the Hong Kong SAR, with exclusive venue
in its courts (Section 9).

## Practical summary

- Using this repository as it stands, in the Territory, is within the license.
- Commercial use is fine below the Section 4 threshold.
- Redistributing the vendored tree means carrying its agreement, `NOTICE`, and
  the change notices — which is what the files above are for.
- Shipping the model as a service to third parties adds the Section 3.e
  disclosure duty.
- Being in the EU, the UK or South Korea is a different conversation.
