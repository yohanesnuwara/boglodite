#!/usr/bin/env bash
# Mount the T7 drive (Windows letter $1, default E) read-only at /mnt/t7. Usage: ./mount-disk.sh [E]
sudo mkdir -p /mnt/t7 && sudo mount -t drvfs "${1:-E}:" /mnt/t7 -o ro,uid=$(id -u),gid=$(id -g) && ls "/mnt/t7/Dell Inspiron/Local Disk D/Full-Norne"
