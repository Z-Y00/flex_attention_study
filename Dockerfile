ARG BASE_IMAGE=rocm/primus:v26.2
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="PyTorch FlexAttention small sparse-block study"
LABEL org.opencontainers.image.source="https://github.com/Z-Y00/flex_attention_study"
LABEL org.opencontainers.image.description="Backports compatible Triton choices for fine-grained ROCm FlexAttention BlockMasks"

# Pick the patch matching the base image's PyTorch:
#   rocm/primus:v26.2                  -> pytorch2.10-...
#   rocm/primus:v26.7-pytorch2.12-...  -> pytorch2.12-...
ARG PATCH=pytorch2.10-flexattention-small-sparse-blocks.patch

COPY patches/${PATCH} /tmp/flex-attention.patch

RUN patch --batch --forward -p1 \
      -d /opt/venv/lib/python3.12/site-packages \
      < /tmp/flex-attention.patch \
    && python3 -m py_compile \
      /opt/venv/lib/python3.12/site-packages/torch/_inductor/kernel/flex/flex_attention.py \
    && rm /tmp/flex-attention.patch
