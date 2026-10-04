#!/usr/bin/env bash
# Pull /workspace/results from every running RunPod pod to this Mac with rsync.
# Only new or changed files are copied, so running it daily is cheap.
#
# Pods are listed in ~/.config/runpod-sync/pods.txt, one per line:
#     <name> <public-ip> <ssh-port>          e.g.  shard0 213.173.108.12 17042
# (IP and port are the pod's "SSH over exposed TCP" details. They change when a
#  pod is re-created, so edit this file when that happens. Lines starting with # are ignored.)
#
# Usage:
#   bash pull_runpod_results.sh              # sync now
#   bash pull_runpod_results.sh --install    # also run it every day at 09:00 (launchd)
#   bash pull_runpod_results.sh --uninstall  # stop the daily run
#
# Results land in $RUNPOD_SYNC_DEST/<name>/ (default ~/runpod_results). That folder and
# the installed copy of this script live outside ~/Documents on purpose: macOS blocks
# background (launchd) jobs from reading or writing ~/Documents without Full Disk Access.
set -uo pipefail

CONF_DIR="$HOME/.config/runpod-sync"
PODS_FILE="$CONF_DIR/pods.txt"
DEST="${RUNPOD_SYNC_DEST:-$HOME/runpod_results}"
LOG="$DEST/sync.log"
LABEL="com.runpod-sync.results"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
INSTALLED="$CONF_DIR/pull_runpod_results.sh"

case "${1:-}" in
  --install)
    mkdir -p "$CONF_DIR" "$DEST" "$HOME/Library/LaunchAgents"
    cp "$0" "$INSTALLED" && chmod +x "$INSTALLED"
    [ -f "$PODS_FILE" ] || printf '# <name> <public-ip> <ssh-port>\n' > "$PODS_FILE"
    cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$INSTALLED</string></array>
  <key>EnvironmentVariables</key><dict><key>RUNPOD_SYNC_DEST</key><string>$DEST</string></dict>
  <key>StartCalendarInterval</key><dict><key>Hour</key><integer>9</integer><key>Minute</key><integer>0</integer></dict>
  <key>StandardOutPath</key><string>$DEST/launchd.log</string>
  <key>StandardErrorPath</key><string>$DEST/launchd.log</string>
</dict></plist>
EOF
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "$PLIST"
    echo "Installed: runs daily at 09:00 (or at next wake if the Mac was asleep)."
    echo "Add pods to $PODS_FILE; results go to $DEST"
    exit 0 ;;
  --uninstall)
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
    rm -f "$PLIST"
    echo "Removed the daily job. Synced results in $DEST are kept."
    exit 0 ;;
esac

mkdir -p "$DEST" "$CONF_DIR"
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

if ! grep -qv '^\s*\(#\|$\)' "$PODS_FILE" 2>/dev/null; then
  log "no pods listed in $PODS_FILE, nothing to do"; exit 0
fi

status=0
while read -r name ip port _; do
  [[ -z "${name:-}" || "$name" == \#* ]] && continue
  mkdir -p "$DEST/$name"
  # Pods are short-lived, so host keys go to a separate known_hosts instead of ~/.ssh/known_hosts.
  ssh_cmd="ssh -p $port -i $HOME/.ssh/id_ed25519 -o ConnectTimeout=20 -o BatchMode=yes \
-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$CONF_DIR/known_hosts"
  if rsync -az --timeout=300 -e "$ssh_cmd" "root@$ip:/workspace/results/" "$DEST/$name/" </dev/null >>"$LOG" 2>&1; then
    log "$name: ok ($(find "$DEST/$name" -name '*.json' | wc -l | tr -d ' ') json files)"
  else
    log "$name: FAILED ($ip:$port), is the pod running and the IP/port current?"; status=1
  fi
done < "$PODS_FILE"
exit $status
