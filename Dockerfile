ARG BASE_IMAGE=rocm/sgl-dev:v0.5.17-rocm720-mi30x-20260819
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="PyTorch FlexAttention small sparse-block study"
LABEL org.opencontainers.image.source="https://github.com/Z-Y00/flex_attention_study"
LABEL org.opencontainers.image.description="Backports compatible Triton choices for fine-grained ROCm FlexAttention BlockMasks"

COPY patches/pytorch-flexattention-small-sparse-blocks.patch /tmp/

RUN patch --batch --forward -p1 \
      -d /opt/venv/lib/python3.10/site-packages \
      < /tmp/pytorch-flexattention-small-sparse-blocks.patch \
    && python3 -m py_compile \
      /opt/venv/lib/python3.10/site-packages/torch/_inductor/kernel/flex/flex_attention.py \
    && rm /tmp/pytorch-flexattention-small-sparse-blocks.patch
