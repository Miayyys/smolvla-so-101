#!/bin/bash
# GPU server (suanjiayun), started after the training process: when the run ends, copy best/ and the small run files
# (not last/, training_state.pt or image caches) to the network disk /root/sj-fs and verify, then shut the instance
# down through the platform API, retrying until it accepts.
#
#   nohup scripts/server_autoshutdown.sh so101_full_v5 > /root/sj-tmp/autoshutdown_so101_full_v5.nohup 2>&1 < /dev/null &
#
# The API URL and token are read from /root/sj-tmp/.suanjia_shutdown_url and .suanjia_token (chmod 600, never in git).
# Copy: cp ~7.5 MB/s to the network disk (dd 16 MB blocks ~6.3, rsync ~0.7). Shutdown: experiment 4 stage 2's single
# call got "502 Bad Gateway" during the platform's maintenance and the instance kept running for days, so a call counts
# only with HTTP 200 and the instance JSON (contains "projectId") in the reply; otherwise retry every 2 min for 2 h.
RUN=${1:?usage: server_autoshutdown.sh <run name under runs/>}
R=/root/sj-tmp/so101-vla-edge/runs/$RUN
DST=/root/sj-fs/$RUN
LOG=/root/sj-tmp/autoshutdown_$RUN.log
log() { echo "$(date '+%F %T') $*" >> $LOG; }
log "watching $R"
while true; do
  if grep -qE '"stage": "(completed|early_stop)"' $R/train.log; then s=DONE
  elif grep -qE 'Error|Traceback|Killed' $R/train.log; then s=FAILED
  elif ! pgrep -f '[s]tage train' >/dev/null; then s=DIED
  else sleep 120; continue; fi
  break
done
log "training state: $s"
sleep 30  # let a final checkpoint write finish
for try in 1 2 3; do
  mkdir -p $DST/best
  rsync -a --exclude '*.jpeg.bin' --exclude last --exclude training_state.pt --exclude 'best/model.safetensors' $R/ $DST/ >> $LOG 2>&1
  cp $R/best/model.safetensors $DST/best/model.safetensors 2>> $LOG
  a=$(cd $R && sha256sum best/model.safetensors); b=$(cd $DST && sha256sum best/model.safetensors)
  if [ "$a" = "$b" ]; then log "copied and verified (try $try): $b"; ok=1; break; fi
  log "verify failed (try $try)"; sleep 60
done
[ -n "$ok" ] || log "copy NOT verified; files remain on the data disk /root/sj-tmp"
cp $LOG $DST/autoshutdown.log 2>/dev/null
for try in $(seq 60); do
  reply=$(curl -s -m 60 -w '\n%{http_code}' -X POST "$(cat /root/sj-tmp/.suanjia_shutdown_url)" \
    -H "Authorization: $(cat /root/sj-tmp/.suanjia_token)" -H 'Content-Type: application/json')
  code=${reply##*$'\n'}; body=${reply%$'\n'*}
  if [ "$code" = 200 ] && [[ $body == *'"projectId"'* ]]; then log "shutdown accepted (try $try)"; break; fi
  log "shutdown failed (try $try): HTTP $code $(echo "$body" | tr -d '\n' | cut -c1-200)"
  cp $LOG $DST/autoshutdown.log 2>/dev/null
  sleep 120
done
cp $LOG $DST/autoshutdown.log 2>/dev/null
