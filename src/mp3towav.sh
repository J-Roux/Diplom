#!/bin/bash
for f in *.au; do lame --decode  "$f" "${f%}.wav"; done
