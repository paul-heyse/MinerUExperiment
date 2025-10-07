## MODIFIED Requirements

### Requirement: Error Handling and Retry

#### Scenario: Requeue previously failed PDFs

- **GIVEN** a PDF has a `.failed` marker from a prior run
- **WHEN** batch processing is invoked with retry controls enabled
- **THEN** the system SHALL delete or ignore the marker
- **AND** it SHALL attempt to process the PDF again
- **AND** it SHALL record that the document was retried in the run summary

#### Scenario: Failed marker expiration

- **WHEN** a `.failed` marker exceeds the configured expiration window
- **THEN** the system SHALL treat the PDF as pending work
- **AND** it SHALL regenerate the failure marker if the PDF fails again

### Requirement: CLI Interface

#### Scenario: Retry failed flag

- **WHEN** `scripts/process_batch.py --retry-failed` is executed
- **THEN** the CLI SHALL enable automatic reprocessing of `.failed` PDFs
- **AND** it SHALL accept an optional `--failed-expiry-hours` override
- **AND** the help output SHALL describe both options
