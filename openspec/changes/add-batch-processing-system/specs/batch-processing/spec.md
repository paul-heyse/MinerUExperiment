# Spec Delta: Batch Processing System

## ADDED Requirements

### Requirement: PDF Directory Scanning

The system SHALL scan the PDFsToProcess directory to discover PDFs for batch processing.

#### Scenario: Find all PDFs

- **WHEN** batch processing starts
- **THEN** the system SHALL scan PDFsToProcess recursively
- **AND** it SHALL identify all files with .pdf extension (case-insensitive)
- **AND** it SHALL exclude hidden files and directories

#### Scenario: Empty directory

- **WHEN** PDFsToProcess contains no PDFs
- **THEN** the system SHALL log a warning
- **AND** it SHALL exit gracefully without error

#### Scenario: Nested directories

- **WHEN** PDFs are in subdirectories of PDFsToProcess
- **THEN** the system SHALL discover them recursively
- **AND** it SHALL preserve relative paths in output naming

### Requirement: Worker Coordination

The system SHALL coordinate multiple parallel workers to prevent duplicate processing of PDFs.

#### Scenario: Claim PDF for processing

- **WHEN** a worker attempts to claim a PDF
- **THEN** it SHALL create a .lock file atomically
- **AND** if the lock already exists, it SHALL skip that PDF
- **AND** if lock creation succeeds, it SHALL proceed with processing

#### Scenario: Release PDF after completion

- **WHEN** a worker completes processing a PDF
- **THEN** it SHALL remove the .lock file
- **AND** it SHALL mark the PDF as processed

#### Scenario: Stale lock detection

- **WHEN** a .lock file is older than 30 minutes
- **THEN** the system SHALL consider it stale
- **AND** it SHALL allow reclaiming that PDF
- **AND** it SHALL log the stale lock cleanup

#### Scenario: Concurrent worker safety

- **WHEN** two workers attempt to claim the same PDF simultaneously
- **THEN** only one worker SHALL succeed in creating the lock
- **AND** the other worker SHALL move to the next PDF

### Requirement: Parallel Worker Execution

The system SHALL spawn multiple parallel workers to process PDFs concurrently.

#### Scenario: Worker pool initialization

- **WHEN** batch processing starts
- **THEN** the system SHALL create a worker pool
- **AND** the pool size SHALL default to (CPU cores - 2)
- **AND** the pool size SHALL be configurable via CLI argument

#### Scenario: Worker task distribution

- **WHEN** workers are available
- **THEN** each worker SHALL claim an unprocessed PDF
- **AND** each worker SHALL process its PDF independently
- **AND** workers SHALL not block each other

#### Scenario: Maximum parallelism

- **WHEN** --workers=14 is specified on AMD 9950x (16 cores)
- **THEN** the system SHALL spawn 14 worker processes
- **AND** all workers SHALL run concurrently
- **AND** system resources SHALL be monitored to prevent overload

### Requirement: Output File Management

The system SHALL save all processed outputs to the MDFilesCreated directory.

#### Scenario: Save Markdown output

- **WHEN** a PDF is successfully processed
- **THEN** the Markdown file SHALL be saved to MDFilesCreated
- **AND** the filename SHALL match the original PDF name with .md extension
- **AND** subdirectory structure SHALL be preserved if PDFs were nested

#### Scenario: Preserve intermediate files

- **WHEN** MinerU generates intermediate files (content_list.json, middle.json)
- **THEN** these files SHALL also be saved to MDFilesCreated
- **AND** they SHALL be in a subdirectory named <pdf_name>_artifacts/

#### Scenario: Output directory creation

- **WHEN** MDFilesCreated does not exist
- **THEN** the system SHALL create it automatically
- **AND** it SHALL create necessary subdirectories

### Requirement: Progress Tracking

The system SHALL track and report batch processing progress.

#### Scenario: Display progress

- **WHEN** batch processing is running
- **THEN** the system SHALL display current progress (N/M PDFs)
- **AND** it SHALL update progress every 5 seconds
- **AND** it SHALL show estimated time remaining

#### Scenario: Summary report

- **WHEN** batch processing completes
- **THEN** the system SHALL display a summary
- **AND** the summary SHALL include total processed, failed, and skipped counts
- **AND** the summary SHALL include total time and average time per PDF

### Requirement: Error Handling and Retry

The system SHALL handle processing errors gracefully and retry transient failures.

#### Scenario: Transient failure retry

- **WHEN** a PDF processing fails with a transient error
- **THEN** the system SHALL retry up to 3 times
- **AND** it SHALL wait 10 seconds between retries
- **AND** it SHALL log each retry attempt

#### Scenario: Permanent failure handling

- **WHEN** a PDF fails after 3 retries
- **THEN** the system SHALL mark it as permanently failed
- **AND** it SHALL move to the next PDF
- **AND** it SHALL log the failure with full error details
- **AND** it SHALL continue processing remaining PDFs

#### Scenario: Individual failure isolation

- **WHEN** one PDF fails permanently
- **THEN** other PDFs SHALL continue processing
- **AND** the batch process SHALL not abort

### Requirement: Graceful Shutdown

The system SHALL support graceful shutdown to avoid corrupting in-progress work.

#### Scenario: SIGINT handling

- **WHEN** SIGINT (Ctrl+C) is received
- **THEN** the system SHALL stop accepting new PDFs
- **AND** it SHALL wait for current workers to complete
- **AND** it SHALL wait up to 60 seconds before forcing termination
- **AND** it SHALL log the shutdown reason

#### Scenario: Emergency termination

- **WHEN** a second SIGINT is received during graceful shutdown
- **THEN** the system SHALL terminate immediately
- **AND** it SHALL log incomplete work

### Requirement: Resource Monitoring

The system SHALL monitor system resources to prevent overload.

#### Scenario: Memory monitoring

- **WHEN** system memory usage exceeds 80%
- **THEN** the system SHALL log a warning
- **AND** it SHALL pause spawning new workers
- **AND** it SHALL resume when memory usage drops below 70%

#### Scenario: CPU load monitoring

- **WHEN** CPU load average exceeds 90%
- **THEN** the system SHALL log a warning
- **AND** it MAY reduce worker count dynamically

### Requirement: CLI Interface

The system SHALL provide a command-line interface for batch processing.

#### Scenario: Run with defaults

- **WHEN** `python scripts/process_batch.py` is run
- **THEN** it SHALL use PDFsToProcess as input directory
- **AND** it SHALL use MDFilesCreated as output directory
- **AND** it SHALL use (CPU cores - 2) workers

#### Scenario: Custom configuration

- **WHEN** `python scripts/process_batch.py --workers 10 --input-dir /custom/pdfs` is run
- **THEN** it SHALL use 10 workers
- **AND** it SHALL scan /custom/pdfs for input
- **AND** it SHALL respect all specified arguments

#### Scenario: Help documentation

- **WHEN** `python scripts/process_batch.py --help` is run
- **THEN** it SHALL display usage information
- **AND** it SHALL document all available arguments
