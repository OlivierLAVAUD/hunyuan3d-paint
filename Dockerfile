# syntax=docker/dockerfile:1
#
# Hunyuan3D texture (paint) API.
#
# Base image: python:3.13-slim, not an nvidia/cuda image.
#   * The PyTorch cuXXX wheels bundle the whole CUDA runtime (cudart, cublas,
#     cudnn, ...) as pip dependencies, so a CUDA base image would be dead
#     weight in the image (~3 GB) and a second source of CUDA versions to keep
#     in sync.
#   * The GPU is still used: `--gpus all` / the compose `deploy.resources`
#     block injects libcuda and the device nodes, and torch finds them.
#   * Python 3.13 matches the environment this pipeline was validated in.
#
# The one thing the slim image cannot do is *compile* the paint model's two
# extensions, so that happens in the `texture-ext` stage below (a CUDA devel
# image) and only the built artifacts are copied in. See that stage's comment.
#
# The CUDA wheel index is a build arg so bumping CUDA is a one-line change:
#   docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 .
# Check torch.cuda.get_device_capability() support: cu130 wheels need a
# driver >= 580.


# =========================================================================== #
# Stage: texture extensions
# =========================================================================== #
#
# This service *only* runs the paint model, so the extensions below are not
# optional here as they were in the shape repo — without them there is no
# service at all. Two pieces that nothing else builds, because `hy3dgen`'s own
# setup.py declares neither and the slim runtime image has no compiler:
#
#   * `custom_rasterizer_kernel` - a CUDA extension (nvcc + the CUDA headers);
#   * `mesh_processor`           - a pybind11 C++ extension (g++).
#
# They are compiled here, in a throw-away CUDA *devel* image that carries nvcc,
# the CUDA headers and a C++ compiler, and only the built artifacts are copied
# into the runtime below.
#
# No GPU is needed to build: PyTorch normally asks the driver which architecture
# to target, which is impossible at build time, so TORCH_CUDA_ARCH_LIST names
# the targets explicitly.
ARG CUDA_VERSION=13.0.2
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS texture-ext

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH}

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
ARG TORCH_VERSION=2.14.0
ARG TORCHVISION_VERSION=0.29.0
# Matches the CUDA flavour of the wheels above. sm_86 (RTX 30xx), sm_89 (40xx)
# and sm_120 (50xx) cover current consumer cards; adjust for another target and
# expect the build to grow with every architecture listed.
ARG TORCH_CUDA_ARCH_LIST="8.6;8.9;12.0"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# Python 3.13 exactly: a compiled module is ABI-locked to the interpreter that
# imports it, which is the runtime image's python:3.13.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.13 \
        python3.13-dev \
        python3.13-venv \
        g++ \
    && rm -rf /var/lib/apt/lists/* \
    && python3.13 -m ensurepip --upgrade

# The build tools come from PyPI, the CUDA-enabled torch (which provides
# `torch.utils.cpp_extension`) from the wheel index - same version as the
# runtime, so the compiled modules load there.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3.13 -m pip install --no-cache-dir ninja setuptools wheel pybind11
RUN --mount=type=cache,target=/root/.cache/pip \
    python3.13 -m pip install --no-cache-dir \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
        --index-url "${TORCH_INDEX_URL}"

COPY third_party/Hunyuan3D-2 /opt/Hunyuan3D-2

# `--no-build-isolation` so the build reuses the torch and pybind11 installed
# above instead of resolving (and downloading) its own. MAX_JOBS bounds nvcc's
# memory: it compiles one kernel per architecture in TORCH_CUDA_ARCH_LIST.
#
# custom_rasterizer is installed into a prefix of its own (`/opt/texture-ext`),
# which the runtime copies over site-packages; upstream's own instructions run
# `setup.py install` for it, and the layout it expects is `custom_rasterizer`
# (the package) next to a top-level `custom_rasterizer_kernel` .so.
#
# differentiable_renderer is built *in place* instead: `mesh_render.py` imports
# it as `from .mesh_processor import ...`, i.e. as a submodule of the installed
# package, so its .so has to end up inside
# `hy3dgen/texgen/differentiable_renderer/` - not next to it in site-packages.
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /opt/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer \
    && MAX_JOBS=4 python3.13 -m pip install --no-build-isolation --no-deps \
         --target /opt/texture-ext . \
    && cd ../differentiable_renderer \
    && MAX_JOBS=4 python3.13 setup.py build_ext --inplace


# =========================================================================== #
# Stage: runtime
# =========================================================================== #
FROM python:3.13-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
ARG TORCH_VERSION=2.14.0
ARG TORCHVISION_VERSION=0.29.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Weights live on a mounted volume, never in the image layer.
    HF_HOME=/data/models \
    HF_HUB_CACHE=/data/models/hub \
    HUGGINGFACE_HUB_CACHE=/data/models/hub \
    # Reduces allocator fragmentation over a long-lived server.
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Runtime shared libraries:
#   libgl1 / libglib2.0-0 / libegl1  -> opencv-python loads libGL
#   libglx0 / libxrender1 / libx11-6 -> the GLVND dispatch library and the
#     X11-free GL stack those link against
#   libgomp1                         -> scikit-image (OpenMP)
#   libexpat1                        -> the vendored `mesh_processor` extension
#     links against it via the glib stack it pulls in.
# nvidia-smi is *not* installed here on purpose: it comes from the driver
# mounted in by the container runtime, and a stale copy in the image would
# report the wrong card (see app/services/cuda.py).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libegl1 \
        libglx0 \
        libxrender1 \
        libx11-6 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# --- PyTorch (CUDA build) ---------------------------------------------------
# Installed first and separately: it is by far the largest layer and the
# requirements below must never be able to change the CUDA flavour.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
        --index-url "${TORCH_INDEX_URL}"

# --- Python dependencies ----------------------------------------------------
# Copied on their own so editing the application code reuses this layer.
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

# --- hy3dgen (vendored upstream model code) ---------------------------------
# Installed with --no-deps: upstream's setup.py would drag gradio and a few
# other demo-only packages into a container that has no UI.
COPY third_party/Hunyuan3D-2 /opt/Hunyuan3D-2
RUN pip install --no-deps /opt/Hunyuan3D-2

# --- Texture extensions, built in the devel stage above ---------------------
# custom_rasterizer (package + top-level kernel .so) lands in site-packages;
# mesh_processor has to go *inside* the installed differentiable_renderer
# package, because that is how mesh_render.py imports it (see the build stage).
COPY --from=texture-ext /opt/texture-ext/ /usr/local/lib/python3.13/site-packages/
COPY --from=texture-ext \
    /opt/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer/mesh_processor*.so \
    /usr/local/lib/python3.13/site-packages/hy3dgen/texgen/differentiable_renderer/

# --- Application ------------------------------------------------------------
WORKDIR /app
COPY . /app

# Output/upload/model directories: bind-mounted in compose, created here so the
# image also works with `docker run` alone.
RUN mkdir -p /data/outputs /data/uploads /data/models/hub \
    && chmod +x /app/entrypoint.sh

EXPOSE 8082

# Liveness: /health answers as soon as the process is up. /ready (readiness)
# is what tells you the weights are loaded.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8082/health', timeout=5).status == 200 else 1)"

ENTRYPOINT ["/app/entrypoint.sh"]
