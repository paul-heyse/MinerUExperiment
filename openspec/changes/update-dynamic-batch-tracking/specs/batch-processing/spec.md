## MODIFIED Requirements

### Requirement: PDF Directory Scanning

#### Scenario: Pick up PDFs arriving mid-run

- **GIVEN** the batch run has already started
- **WHEN** a new `.pdf` file appears under PDFsToProcess
- **THEN** the system SHALL discover the new file
- **AND** it SHALL enqueue the PDF for processing before the batch completes

### Requirement: Parallel Worker Execution

#### Scenario: Continue until queue drained

- **GIVEN** workers are running
- **AND** new PDFs or reclaimed stale locks increase the pending queue
- **WHEN** the worker pool finishes the original set of PDFs
- **THEN** the orchestrator SHALL keep workers alive until no pending PDFs remain for a configured quiet period
- **AND** it SHALL NOT signal shutdown while work is still queued

### Requirement: Progress Tracking

#### Scenario: Dynamic totals for late arrivals

- **WHEN** additional PDFs are discovered after processing begins
- **THEN** the progress display SHALL refresh the total/remaining counts to include them
- **AND** the final summary SHALL report how many PDFs were processed after startup
