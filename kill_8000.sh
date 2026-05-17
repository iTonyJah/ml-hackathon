#!/bin/bash
echo "Killing process on port 8000..."
fuser -k 8000/tcp 2>/dev/null || true
sleep 1
echo "Port 8000 is free"
