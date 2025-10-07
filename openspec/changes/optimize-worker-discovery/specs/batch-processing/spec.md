# Spec Delta: Batch Processing Discovery Optimization

## MODIFIED Requirements

### Requirement: PDF Directory Scanning

#### Scenario: Efficient rescans on large corpora
- **WHEN** more than 1,000 PDFs exist in PDFsToProcess
- **THEN** the system SHALL avoid re-walking the entire tree for each worker claim
- **AND** it SHALL reuse an incremental index that only touches directories changed since the last scan
- **AND** the rescan interval SHALL be configurable without restarting the batch run

#### Scenario: Detect new PDFs during active run
- **WHEN** new PDFs are added to PDFsToProcess after batch processing has started
- **THEN** the system SHALL discover and enqueue them for processing within one rescan interval
- **AND** it SHALL skip any new files that already have `.lock`, `.done`, or `.failed` markers

### Requirement: Progress Tracking

#### Scenario: Dynamic totals reflect newly discovered work
- **WHEN** additional PDFs are discovered mid-run
- **THEN** the progress display SHALL update the total/remaining counts accordingly
- **AND** the completion summary SHALL include the total number of PDFs discovered during the run

## ADDED Requirements

### Requirement: Incremental Work Queue

The system SHALL maintain a shared queue of pending PDFs populated by the discovery index.

#### Scenario: Single-pass claims
- **WHEN** a worker requests the next PDF
- **THEN** it SHALL consume from the shared queue instead of scanning the filesystem itself
- **AND** it SHALL skip any PDFs already claimed or marked complete

#### Scenario: Requeue reclaimed locks
- **WHEN** a stale lock is cleaned up
- **THEN** the associated PDF SHALL be pushed back onto the shared queue for retry
- **AND** the system SHALL log the requeue event for observability
