# SO-101 device paths (by-id, stable across reboots/replugging). Usage: source scripts/so101_env.fish
# Leader/follower assignment verified with lerobot-find-port on 2026-10-21; see docs/hardware.md.
set -gx FOLLOWER /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14111527-if00
set -gx LEADER /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14031528-if00
set -gx TOP /dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB_2.0_Camera-video-index0
set -gx WRIST /dev/v4l/by-id/usb-ZSKJ-260319-K_Integrated_Webcam_HD-video-index0

for v in FOLLOWER LEADER TOP WRIST
    if not test -e $$v
        echo "so101_env: $v -> $$v not found (unplugged?)" >&2
    end
end
