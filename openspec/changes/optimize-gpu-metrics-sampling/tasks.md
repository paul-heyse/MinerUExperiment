# Tasks: Optimize GPU metrics sampling via NVML with CLI fallback

- [ ] Prototype an NVML sampling helper that reports utilization, memory, and temperature without invoking `nvidia-smi`.
- [ ] Integrate the helper into `MetricsCollector`, falling back to the current subprocess path when NVML is unavailable.
- [ ] Ensure collector teardown calls NVML shutdown routines to prevent resource leaks.
- [ ] Update unit tests to cover both NVML-enabled and fallback paths (using fakes/mocks where necessary).
- [ ] Document the optional `pynvml` dependency and update dependency lists if needed.
