# Tasks: Add batch processor preflight validation and clearer failure handling

- [ ] Design a preflight validator that checks CLI availability, required Python modules, and output directory writability.
- [ ] Integrate the validator into the batch processor setup phase with descriptive error messages.
- [ ] Add integration tests covering missing CLI and missing torch/vllm scenarios to ensure the processor aborts early.
- [ ] Update documentation (README/CLI help) to mention the new validation behavior and guidance on resolving failures.
