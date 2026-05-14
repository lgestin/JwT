#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="${1:-data}"
URL="https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
ARCHIVE="LJSpeech-1.1.tar.bz2"
EXPECTED_SHA256="be1a30453f28eb8dd26af4101ae40cbf2c50413b1bb21936cbcdc6fae3de8aa5"

mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

if [[ -d "LJSpeech-1.1" ]]; then
    echo "LJSpeech-1.1 already extracted in $DEST_DIR, skipping."
    exit 0
fi

if [[ ! -f "$ARCHIVE" ]]; then
    echo "Downloading $URL ..."
    curl -L --fail --progress-bar -o "$ARCHIVE" "$URL"
fi

echo "Verifying checksum..."
echo "$EXPECTED_SHA256  $ARCHIVE" | sha256sum -c -

echo "Extracting $ARCHIVE ..."
tar -xjf "$ARCHIVE"

rm "$ARCHIVE"
echo "Done. Dataset at $(pwd)/LJSpeech-1.1"
