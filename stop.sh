#!/usr/bin/env bash
# stop.sh — stop only this kit's own V4.1 REAP-256E pair. Nothing else is touched.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start.sh" stop
