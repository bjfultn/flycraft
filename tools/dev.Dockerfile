FROM python:3.11-slim
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir "numpy>=1.26" "pyyaml>=6" "pandas>=2.1" "pyarrow>=14" \
    "google-crc32c>=1.5" "matplotlib>=3.8" "websockets>=13" "msgpack>=1.0" \
    "pytest>=8" "ruff>=0.6"
# pysc2 for the game-side unit tests (no SC2 binary needed). The archive URL avoids needing git.
RUN pip install --no-cache-dir \
    "pysc2 @ https://github.com/google-deepmind/pysc2/archive/0df53d38c153972f1e368572ba65b1442a0fd41f.zip" \
    "protobuf==3.20.3"
WORKDIR /w
