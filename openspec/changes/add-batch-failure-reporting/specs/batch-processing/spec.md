# Spec Delta: Batch Processing System

## ADDED Requirements

### Requirement: Failure Reporting Artifacts

The batch processor SHALL produce a structured artifact summarizing permanent failures.

#### Scenario: Generate failure report

- **WHEN** one or more PDFs exhaust retries and are marked as permanently failed
- **THEN** the system SHALL write a JSON report in the output directory listing each failed PDF, the attempts made, and the final
  error message
- **AND** the final console summary SHALL reference the path to this report

#### Scenario: No failures

- **WHEN** all PDFs succeed
- **THEN** the batch processor SHALL NOT create the failure report artifact
- **AND** the summary SHALL explicitly state that no failures occurred
