FROM python:3.11-slim
WORKDIR /workspace
COPY . /workspace
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -e ".[dev]"
CMD ["python", "-m", "cfq.cli", "smoke", "--output-dir", "results/smoke"]
