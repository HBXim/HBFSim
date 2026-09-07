# Physical configuration profiles

The core engine accepts `key=value` configuration files through repeatable
`--system-config` options. Files are applied from left to right, so a later
overlay replaces only the keys it declares.

## Complete system

- `systems/server-hbm128-hbf512.cfg`: 128 GiB HBM with 512 GiB HBF, including
  HBM timing, HBF geometry, controller resources, base-die links, garbage
  collection, and write buffering.

## External backing overlays

- `overlays/backing/host-dram.cfg`
- `overlays/backing/cxl-memory.cfg`
- `overlays/backing/nvme-ssd.cfg`
- `overlays/backing/cxl-ssd.cfg`
- `overlays/backing/cxl-ssd-cached.cfg`
- `overlays/backing/on-package-lpddr.cfg`

Apply one backing profile after the complete system profile and start the
engine with `--enable-external true`. The CXL-SSD cache overlay is applied
after `cxl-ssd.cfg`.

## HBF mechanism overlays

- `overlays/hbf/cached-l2p-1-over-1000.cfg`: use a capacity-derived controller
  DRAM budget for cached logical-to-physical translation.
- `overlays/hbf/external-direct-lane.cfg`: enable the explicit direct link
  between each HBF base die and the external backing device.

All included numeric profiles state their modeling assumptions in comments.
