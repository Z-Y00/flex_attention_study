ARG BASE_IMAGE=rocm/primus:v26.7-pytorch2.12-te2.17
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="PyTorch FlexAttention small sparse-block study"
LABEL org.opencontainers.image.source="https://github.com/Z-Y00/flex_attention_study"
LABEL org.opencontainers.image.description="Backports compatible Triton choices for fine-grained ROCm FlexAttention BlockMasks"

COPY patches/pytorch2.12-flexattention-small-sparse-blocks.patch /tmp/

RUN patch --batch --forward -p1 \
      -d /opt/venv/lib/python3.12/site-packages \
      < /tmp/pytorch2.12-flexattention-small-sparse-blocks.patch \
    && python3 -m py_compile \
      /opt/venv/lib/python3.12/site-packages/torch/_inductor/kernel/flex/flex_attention.py \
    && rm /tmp/pytorch2.12-flexattention-small-sparse-blocks.patch
