> [!WARNING]
> This is a planning document and should not be considered as a source of truth without verification.

# ActorII Post-1993 Technical Modernization Roadmap & Backlog

Tracks the modern OS, protocol, toolchain, codec, and hardware object layers added to `ActorII` so Bilby (`qwen3.8:latest`) reasons over first-class Actor 4.1 objects and deterministic Ring-0 bridges instead of improvising post-1993 infrastructure from raw shell strings.

---

## Completed Layers (`dev` / `v4.4.x`)

- [x] **1. Web & Media Streaming Protocol Suite (`Actor[#Protocols]` + `Actor[#Net]` Wire Probes — `v4.4.0`)**
  - Ring-1 `ProtocolSpec` & `ProtocolSuite` (`web_protocols.act`): `#http`/`#rest`, `#mjpeg`, `#rtsp` (`#send` & `#receive`, including Jetson TX1 `omxh264enc` vs `nvv4l2h264enc` SPS rules), `#hls`, `#sse`, `#ws` (RFC 6455).
  - Ring-0 active conformance probes: `probeHttp`, `probeMjpeg`, `probeRtsp`, `probeSse`, `probeWebSocket`, `probeHls` + `JetsonEdgeTwin` `deployService` / `serviceStatus` / `stopService`.
- [x] **2. Network Topology, Tailnet Identity & LAN->Tailscale Port Relay (`Actor[#Network]` + `Actor[#NetTopo]`)**
  - Ring-1 `NetworkTopology`, `NetworkNode`, `PortRelayRoute` (`network_topology.act`) + Ring-0 `NetworkTopologyBridge`: live host interface map, `tailscale status --json` peer ownership (`whoOwnsIp`), `explainReachability`, and guarded `exposeLanPort` (`8091..8999` user-space `socat` relays).
- [x] **3. Zero-Action Mutation / Telemetry Hallucination Guard (`UnverifiedExecutionClaimError`)**
  - Deterministic `VMSandboxBridge` check blocking Step-1 `COMPLETE:` replies that claim live installations, PIDs, restarts, or interface changes when `0` Actor 4.1 tool actions executed in the turn.

---

## Active Backlog (Priority Order)

### Priority 1 — Modern Data Formats, Schemas & Binary/Crypto Primitives (`Actor[#Codecs]`)
- [ ] **Ring-0 `CodecBridge` + Ring-1 `DataCodec` / `SchemaValidator` (`codecs.act`)**:
  - Native Tier-0 `parseJson` / `toJson` (mapping cleanly between JSON and Actor `Dictionary` / `Array`), `parseYaml`, `parseToml`, `parseCsv`.
  - Binary & crypto primitives inside pure Actor code: `base64Encode` / `base64Decode`, `sha256`, `hmacSha256`, `regexFind` / `regexReplace`, and `packStruct` / `unpackStruct` (so `.act` Engrams never have to shell out to `python3 -c` for JSON/header parsing).

### Priority 2 — Build Systems, Cross-Compilation & Native ABI Inspector (`Actor[#Toolchain]`)
- [ ] **Ring-0 `ToolchainBridge` + Ring-1 `BuildTarget` Hierarchy (`toolchain.act`)**:
  - `CudaTarget`: Hardware compute-capability profiles (`sm_53` + `gcc-7` on Jetson TX1 L4T R32.7.6 vs `sm_120` on `valkyrie` RTX 5070 Ti), `pkg-config` resolution, and preprocessor header-path verification (`resolveHeader` + `ldd` runtime `.so` ABI match check to catch `libjpeg`-style struct mismatches before runtime).
  - `CrossRustTarget`: First-class wrapper around `/home/skippy/jetson-rust/` (`jetson-tx1-rust:latest` `aarch64` `glibc 2.27` sysroot) so Bilby can compile fast `aarch64` binaries on `valkyrie`'s 28 threads and deploy them to `Actor[#Jetson]` in one message send.

### Priority 3 — Linux Daemons, `systemd` User Units & Container Orchestration (`Actor[#Containers]`)
- [ ] **Ring-0 `ContainerAndUnitBridge` + Ring-1 `ServiceUnit` / `ContainerSpec` (`containers.act`)**:
  - Safe user-space `systemd --user` unit generator/inspector (with hard allowlists so OS/network/desktop daemons remain untouchable) + structured `journalctl` JSON log reader.
  - OCI / Docker / Compose inspection & multi-arch (`linux/amd64`, `linux/arm64`) sandboxed build/test helpers.

### Priority 4 — Edge Hardware, Sensor Buses & Multimodal Vision/Audio Pipelines (`Actor[#Media]` / `Actor[#Sensors]`)
- [ ] **Ring-1 `MediaPipeline` & `EdgeSensor` Hierarchy (`media_sensors.act`)**:
  - V4L2 / MIPI-CSI camera capabilities, PipeWire/ALSA audio capture & TTS playback, generalized BLE/GATT scanner & characteristic reader (extending beyond `ThermoProTwin`), GPIO/I2C readouts on Jetson J21 header, and TensorRT (`sm_53` FP16) inference runner on `Actor[#Jetson]`.
