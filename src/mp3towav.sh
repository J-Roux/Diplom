#!/bin/bash
for f in *.mp3; do lame --decode  "$f" "../wav/${f%.mp3}.wav"; done
